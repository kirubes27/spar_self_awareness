#!/usr/bin/env python3
"""
pca_confidence_subspace.py

Minimal PCA analysis for confidence contrastive activations.
Goal: Determine if confidence signal is 1D (PC1 ≈ d_conf) or multi-dimensional.

Usage:
  python interp/pca_confidence_subspace.py --layer 35 --n 5 --k 5   # smoke test
  python interp/pca_confidence_subspace.py --layer 35 --k 10        # full run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add interp directory to path
sys.path.insert(0, str(Path(__file__).parent))
from prompt_utils import build_self_prompt

# --------- CONFIG ---------
MODEL_ID_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"

PAIRS_CSV = Path(f"contrastive_pairs/{MODEL_NAME}/{MODEL_NAME}_introspective_extremes_AB_train.csv")
COMPILED_JSON = Path(f"compiled_results_smc/{MODEL_NAME}_phase1_compiled.json")
DIRECTION_DIR = Path("interp/outputs")


# --------- Model Loading ---------
def load_model_and_tokenizer(model_id: str):
    """Load 4-bit quantized model."""
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


# --------- Data Loading ---------
def load_pairs(pairs_csv: Path, compiled_json: Path, n: int | None):
    """Load A/B pairs with question text from compiled JSON."""
    print(f"Loading pairs from {pairs_csv}...")

    # Load compiled JSON for question text
    with open(compiled_json) as f:
        compiled_data = json.load(f)

    def get_question_text(qid: str) -> str:
        """Get formatted MCQ text from compiled JSON."""
        res = compiled_data.get("results", {}).get(qid, {})
        q_data = res.get("question", {})
        if not isinstance(q_data, dict):
            return ""

        q_stem = (q_data.get("question") or "").strip()
        opts = q_data.get("options")

        if not q_stem:
            return ""

        # Format with options
        if isinstance(opts, dict) and opts:
            lines = [q_stem, ""]
            for k in ["A", "B", "C", "D"]:
                if k in opts:
                    lines.append(f"{k}) {opts[k]}")
            return "\n".join(lines).strip()

        if isinstance(opts, list) and len(opts) >= 2:
            lines = [q_stem, ""]
            for k, opt in zip(["A", "B", "C", "D"], opts[:4]):
                lines.append(f"{k}) {opt}")
            return "\n".join(lines).strip()

        return q_stem

    pairs = []
    with open(pairs_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a_qid = row.get("A_qid", "").strip()
            b_qid = row.get("B_qid", "").strip()

            if not a_qid or not b_qid:
                continue

            a_text = get_question_text(a_qid)
            b_text = get_question_text(b_qid)

            if not a_text or not b_text:
                continue

            pairs.append({
                "pair_id": row.get("pair_id", ""),
                "high_qid": a_qid,  # A = high confidence
                "low_qid": b_qid,   # B = low confidence
                "high_text": a_text,
                "low_text": b_text,
            })

    print(f"Loaded {len(pairs)} valid pairs.")

    if n is not None and len(pairs) > n:
        pairs = pairs[:n]
        print(f"Using first {n} pairs.")

    return pairs


# --------- Hidden State Extraction ---------
def extract_hidden_state(
    model, tokenizer, prompt: str, layer_idx: int
) -> torch.Tensor:
    """Extract last-token hidden state at specified layer."""

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt")
    target_device = model.model.embed_tokens.weight.device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    # Hook to capture hidden state
    captured = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        captured["hidden"] = h[:, -1, :].detach().clone()  # last token

    layer_module = model.model.layers[layer_idx]
    handle = layer_module.register_forward_hook(hook)

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return captured["hidden"].squeeze(0)  # (hidden_dim,)


# --------- PCA ---------
def run_pca(D: torch.Tensor, k: int):
    """Run PCA on difference vectors D using SVD."""
    # Move to CPU float32 for numerical stability (CUDA float16 SVD can fail)
    D0 = (D - D.mean(dim=0, keepdim=True)).float().cpu()

    # SVD: D0 = U @ S @ Vt
    U, S, Vt = torch.linalg.svd(D0, full_matrices=False)

    # Variance explained
    var_total = (S ** 2).sum()
    var_explained = (S ** 2) / var_total

    # PCs are rows of Vt
    pcs = Vt[:k]  # (k, hidden_dim)

    return pcs, var_explained[:k].tolist(), S[:k].tolist()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.view(-1).float()
    b = b.view(-1).float()
    return (a @ b / (a.norm() * b.norm())).item()


# --------- Main ---------
def main():
    parser = argparse.ArgumentParser(
        description="Minimal PCA analysis for confidence contrastive activations"
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--pairs-csv", type=str, default=str(PAIRS_CSV))
    parser.add_argument("--compiled-json", type=str, default=str(COMPILED_JSON))
    parser.add_argument("--k", type=int, default=5, help="Number of PCs to report")
    parser.add_argument("--n", type=int, default=None, help="Limit pairs (smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("PCA: CONFIDENCE SUBSPACE ANALYSIS")
    print("=" * 60)
    print(f"Layer: {args.layer}")
    print(f"Pairs CSV: {args.pairs_csv}")
    print(f"K (PCs): {args.k}")
    print(f"N limit: {args.n or 'all'}")
    print("=" * 60)

    # Load data
    pairs = load_pairs(Path(args.pairs_csv), Path(args.compiled_json), args.n)

    if len(pairs) < 3:
        print("ERROR: Need at least 3 pairs for PCA")
        return

    # Load model
    tokenizer, model = load_model_and_tokenizer(args.model_id)

    # Extract hidden states
    print(f"\nExtracting hidden states at layer {args.layer}...")
    D_list = []

    for pair in tqdm(pairs, desc="Pairs"):
        prompt_high = build_self_prompt(pair["high_text"])
        prompt_low = build_self_prompt(pair["low_text"])

        h_high = extract_hidden_state(model, tokenizer, prompt_high, args.layer)
        h_low = extract_hidden_state(model, tokenizer, prompt_low, args.layer)

        d = h_high - h_low
        D_list.append(d)

    D = torch.stack(D_list)  # (n_pairs, hidden_dim)
    print(f"D shape: {D.shape}")

    # Run PCA
    print("\nRunning PCA...")
    pcs, var_explained, singular_values = run_pca(D, args.k)

    # Load d_conf for comparison
    direction_path = DIRECTION_DIR / f"confidence_direction_layer{args.layer}.pt"
    if direction_path.exists():
        data = torch.load(direction_path, map_location="cpu", weights_only=False)
        if isinstance(data, torch.Tensor):
            d_conf = data
        elif isinstance(data, dict) and "direction" in data:
            d_conf = data["direction"]
        else:
            d_conf = None

        if d_conf is not None:
            d_conf = d_conf.view(-1).float()
            cos_pc1_dconf = cosine_similarity(pcs[0], d_conf)
            cos_pc2_dconf = cosine_similarity(pcs[1], d_conf) if len(pcs) > 1 else None
        else:
            cos_pc1_dconf = None
            cos_pc2_dconf = None
    else:
        print(f"WARNING: d_conf not found at {direction_path}")
        cos_pc1_dconf = None
        cos_pc2_dconf = None

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"N pairs: {len(pairs)}")
    print(f"\nVariance explained (PC1..PC{args.k}):")
    for i, v in enumerate(var_explained):
        bar = "█" * int(v * 50)
        print(f"  PC{i+1}: {v:.4f} ({v*100:.1f}%) {bar}")

    cumulative = sum(var_explained)
    print(f"\nCumulative (PC1..PC{args.k}): {cumulative:.4f} ({cumulative*100:.1f}%)")

    if cos_pc1_dconf is not None:
        print(f"\ncos(PC1, d_conf): {cos_pc1_dconf:.4f}")
        if cos_pc2_dconf is not None:
            print(f"cos(PC2, d_conf): {cos_pc2_dconf:.4f}")

    # Interpretation
    print("\n--- INTERPRETATION ---")
    if var_explained[0] > 0.8:
        print("✓ PC1 dominates (>80%) — confidence signal is essentially 1D")
    elif var_explained[0] > 0.5:
        print("⚠ PC1 explains 50-80% — some structure beyond PC1")
    else:
        print("⚠ PC1 < 50% — confidence signal is multi-dimensional")

    if cos_pc1_dconf is not None:
        if abs(cos_pc1_dconf) > 0.9:
            print(f"✓ PC1 aligns strongly with d_conf (cos={cos_pc1_dconf:.3f})")
        elif abs(cos_pc1_dconf) > 0.7:
            print(f"~ PC1 moderately aligns with d_conf (cos={cos_pc1_dconf:.3f})")
        else:
            print(f"⚠ PC1 does NOT align with d_conf (cos={cos_pc1_dconf:.3f})")

    print("=" * 60)


if __name__ == "__main__":
    main()
