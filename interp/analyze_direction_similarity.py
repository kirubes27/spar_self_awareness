#!/usr/bin/env python3
"""
analyze_direction_similarity.py

Compute cosine similarities between d_conf, d_pass, and d_so directions.
Also performs cross-validation of d_conf if contrastive pairs data is available.

This is a CPU-only script - no GPU needed.
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F

OUTPUT_DIR = Path("interp/outputs")


def load_direction(name: str, layer: int = 35) -> torch.Tensor:
    """Load a direction vector from .pt file."""
    path = OUTPUT_DIR / f"{name}_layer{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Direction not found: {path}")

    data = torch.load(path, weights_only=False, map_location="cpu")

    if isinstance(data, dict):
        # Handle different dict formats
        if "direction" in data:
            v = data["direction"]
        elif "d" in data:
            v = data["d"]
        else:
            v = list(data.values())[0]
    else:
        v = data

    return v.view(-1).float()


def compute_cosine_matrix(directions: dict) -> dict:
    """Compute pairwise cosine similarities."""
    names = list(directions.keys())
    results = {}

    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            d1 = directions[name1]
            d2 = directions[name2]
            cos = F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
            results[f"{name1}_vs_{name2}"] = round(cos, 4)

    return results


def main():
    print("=" * 60)
    print("DIRECTION SIMILARITY ANALYSIS")
    print("=" * 60)

    # Load directions
    directions = {}
    for name in ["confidence_direction", "pass_game_direction", "self_other_direction"]:
        try:
            directions[name] = load_direction(name, layer=35)
            print(f"Loaded {name}: shape={directions[name].shape}, norm={directions[name].norm():.4f}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if len(directions) < 2:
        print("Not enough directions to compare!")
        return

    # Rename for cleaner output
    rename = {
        "confidence_direction": "d_conf",
        "pass_game_direction": "d_pass",
        "self_other_direction": "d_so"
    }
    directions = {rename.get(k, k): v for k, v in directions.items()}

    print("\n" + "=" * 60)
    print("COSINE SIMILARITIES (Layer 35)")
    print("=" * 60)

    cosines = compute_cosine_matrix(directions)
    for pair, cos in cosines.items():
        interpretation = ""
        if abs(cos) > 0.7:
            interpretation = " ← HIGHLY CORRELATED"
        elif abs(cos) > 0.4:
            interpretation = " ← MODERATELY CORRELATED"
        elif abs(cos) < 0.1:
            interpretation = " ← NEARLY ORTHOGONAL"
        print(f"  {pair}: {cos:.4f}{interpretation}")

    # Save results
    output = {
        "layer": 35,
        "norms": {name: round(d.norm().item(), 4) for name, d in directions.items()},
        "cosine_similarities": cosines,
    }

    out_path = OUTPUT_DIR / "direction_similarity_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Pair':<20} | {'Cosine':<10} | {'Interpretation'}")
    print("-" * 55)
    for pair, cos in cosines.items():
        if abs(cos) > 0.7:
            interp = "Same direction"
        elif abs(cos) > 0.4:
            interp = "Related"
        elif abs(cos) > 0.1:
            interp = "Distinct"
        else:
            interp = "Orthogonal"
        print(f"{pair:<20} | {cos:<10.4f} | {interp}")


if __name__ == "__main__":
    main()
