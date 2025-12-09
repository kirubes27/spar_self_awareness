#!/usr/bin/env python3
"""
generate_random_directions.py

Generate random unit vectors matched to d_conf dimensionality for baseline comparison.
"""

import os
from pathlib import Path

import torch


def generate_random_directions(n_directions: int = 10, seed: int = 42):
    """Generate random unit vectors matched to d_conf dimensionality."""

    # Infer hidden_dim from existing d_conf file
    reference_path = Path("interp/outputs/confidence_direction_layer35.pt")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference direction not found: {reference_path}")

    reference = torch.load(reference_path, weights_only=False)

    # Handle different formats (tensor or dict)
    if isinstance(reference, dict) and "direction" in reference:
        reference = reference["direction"]

    hidden_dim = reference.shape[-1]
    ref_norm = reference.norm().item()
    print(f"Hidden dim: {hidden_dim}")
    print(f"Reference norm: {ref_norm:.4f}")

    # Set seed for reproducibility
    torch.manual_seed(seed)

    output_dir = Path("interp/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_directions):
        # Generate random vector
        v = torch.randn(hidden_dim)
        # Normalize to unit norm, then scale to match d_conf norm
        v = v / v.norm()
        v = v * ref_norm  # Match d_conf magnitude for fair comparison

        save_path = output_dir / f"random_direction_{i}_layer35.pt"
        torch.save(v, save_path)
        print(f"Saved {save_path} (norm={v.norm():.4f})")

    print(f"\n✓ Generated {n_directions} random directions (all norm={ref_norm:.4f})")


if __name__ == "__main__":
    generate_random_directions(n_directions=10)
