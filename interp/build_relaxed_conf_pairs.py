#!/usr/bin/env python3
"""
build_relaxed_conf_pairs.py

Build relaxed contrastive pairs for confidence direction extraction.
Supports two modes:
  --count-only: Print pair counts for threshold grid (no model calls)
  Default: Build pairs CSV and optionally extract directions

Usage:
  # Count mode (quick, no GPU)
  python interp/build_relaxed_conf_pairs.py --count-only

  # Build + extract
  python interp/build_relaxed_conf_pairs.py \
    --self-hi 0.75 --ent-q 0.75 --max-pairs 100 \
    --extract-direction --layers 35 50 79
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add interp directory to path
sys.path.insert(0, str(Path(__file__).parent))
from prompt_utils import build_self_prompt

# --------- CONFIG ---------
MODEL_ID_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"

UNIFIED_CSV_DEFAULT = f"contrastive_pairs/{MODEL_NAME}/{MODEL_NAME}_unified.csv"
EXCLUDE_CSV_DEFAULT = f"contrastive_pairs/{MODEL_NAME}/{MODEL_NAME}_introspective_extremes_AB_train.csv"
COMPILED_JSON = Path(f"compiled_results_smc/{MODEL_NAME}_phase1_compiled.json")


# --------- UTILITIES ---------
def detect_qid_column(df: pd.DataFrame) -> str:
    """Auto-detect QID column from DataFrame."""
    candidates = ["qid", "QID", "question_id", "id", "Qid"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot detect QID column. Candidates: {candidates}, Found: {list(df.columns)}")


def load_excluded_qids(exclude_csv: Path) -> set:
    """Load QIDs to exclude from pairs CSV."""
    if not exclude_csv.exists():
        return set()

    df = pd.read_csv(exclude_csv)
    candidates = ["A_qid", "B_qid", "high_qid", "low_qid", "qid_a", "qid_b"]
    excluded = set()
    for col in candidates:
        if col in df.columns:
            excluded.update(df[col].dropna().astype(str).tolist())
    return excluded


def get_question_text(qid: str, compiled_data: dict) -> str:
    """Get formatted MCQ text from compiled JSON."""
    res = compiled_data.get("results", {}).get(qid, {})
    q_data = res.get("question", {})
    if not isinstance(q_data, dict):
        return ""

    q_stem = (q_data.get("question") or "").strip()
    opts = q_data.get("options")

    if not q_stem:
        return ""

    if isinstance(opts, dict) and opts:
        lines = [q_stem, ""]
        for k in ["A", "B", "C", "D"]:
            if k in opts:
                lines.append(f"{k}) {opts[k]}")
        return "\n".join(lines).strip()

    if isinstance(opts, list) and len(opts) >= 4:
        lines = [q_stem, ""]
        for k, opt in zip(["A", "B", "C", "D"], opts[:4]):
            lines.append(f"{k}) {opt}")
        return "\n".join(lines).strip()

    return q_stem


# --------- COUNT MODE ---------
def run_count_mode(df: pd.DataFrame, self_col: str, entropy_col: str, qid_col: str, excluded_qids: set):
    """Print counts for threshold grid (accounting for exclusions)."""
    print("\n" + "=" * 60)
    print("COUNT MODE: Threshold Grid")
    print(f"Excluded QIDs: {len(excluded_qids)}")
    print("=" * 60)

    self_thresholds = [0.85, 0.80, 0.75, 0.70]
    ent_quantiles = [0.85, 0.80, 0.75, 0.70]

    print(f"\n{'self_hi':<10} {'ent_q':<10} {'n_high':<10} {'n_low':<10} {'n_pairs':<10}")
    print("-" * 50)

    for self_hi in self_thresholds:
        for ent_q in ent_quantiles:
            q_low = df[entropy_col].quantile(1 - ent_q)
            q_high = df[entropy_col].quantile(ent_q)

            high_mask = (df[self_col] >= self_hi) & (df[entropy_col] <= q_low)
            low_mask = (df[self_col] <= (1 - self_hi)) & (df[entropy_col] >= q_high)

            # Apply exclusions
            high_df = df[high_mask]
            low_df = df[low_mask]
            high_df = high_df[~high_df[qid_col].astype(str).isin(excluded_qids)]
            low_df = low_df[~low_df[qid_col].astype(str).isin(excluded_qids)]

            n_high = len(high_df)
            n_low = len(low_df)
            n_pairs = min(n_high, n_low)

            marker = "✓" if n_pairs >= 30 else ""
            print(f"{self_hi:<10} {ent_q:<10} {n_high:<10} {n_low:<10} {n_pairs:<10} {marker}")

    print("\n✓ = n_pairs >= 30 (recommended, after exclusions)\n")


# --------- BUILD PAIRS ---------
def build_pairs(
    df: pd.DataFrame,
    qid_col: str,
    self_col: str,
    entropy_col: str,
    self_hi: float,
    ent_q: float,
    max_pairs: int,
    excluded_qids: set,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Build balanced contrastive pairs."""

    # Compute quantiles
    q_low = df[entropy_col].quantile(1 - ent_q)
    q_high = df[entropy_col].quantile(ent_q)

    # Select bins
    high_mask = (df[self_col] >= self_hi) & (df[entropy_col] <= q_low)
    low_mask = (df[self_col] <= (1 - self_hi)) & (df[entropy_col] >= q_high)

    high_df = df[high_mask].copy()
    low_df = df[low_mask].copy()

    print(f"Before exclusion: high={len(high_df)}, low={len(low_df)}")

    # Exclude QIDs
    high_df = high_df[~high_df[qid_col].astype(str).isin(excluded_qids)]
    low_df = low_df[~low_df[qid_col].astype(str).isin(excluded_qids)]

    print(f"After exclusion: high={len(high_df)}, low={len(low_df)}")

    # Balance
    K = min(len(high_df), len(low_df), max_pairs)

    if K < 5:
        raise ValueError(f"Too few pairs: {K}. Need at least 5.")

    # Deterministic sampling
    rng = np.random.default_rng(seed)

    high_qids = high_df[qid_col].astype(str).sort_values().tolist()
    low_qids = low_df[qid_col].astype(str).sort_values().tolist()

    high_selected = rng.choice(high_qids, size=K, replace=False)
    low_selected = rng.choice(low_qids, size=K, replace=False)

    # Shuffle for pairing
    rng.shuffle(high_selected)
    rng.shuffle(low_selected)

    # Create pairs DataFrame
    pairs = pd.DataFrame({
        "pair_id": range(K),
        "A_qid": high_selected,
        "B_qid": low_selected,
    })

    meta = {
        "thresholds": {
            "self_hi": self_hi,
            "self_lo": 1 - self_hi,
            "ent_q": ent_q,
            "q_low": float(q_low),
            "q_high": float(q_high),
        },
        "K": K,
        "seed": seed,
        "excluded_count": len(excluded_qids),
        "high_qids": list(high_selected),
        "low_qids": list(low_selected),
    }

    return pairs, meta


# --------- DIRECTION EXTRACTION ---------
def load_model_and_tokenizer(model_id: str):
    """Load 4-bit quantized model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def extract_hidden_state(
    model, tokenizer, prompt: str, layer_idx: int
) -> torch.Tensor:
    """Extract last-token hidden state at specified layer."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt")
    target_device = model.model.embed_tokens.weight.device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    captured = {}

    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        captured["hidden"] = h[:, -1, :].detach().clone()

    layer_module = model.model.layers[layer_idx]
    handle = layer_module.register_forward_hook(hook)

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return captured["hidden"].squeeze(0)


def extract_directions(
    pairs_df: pd.DataFrame,
    compiled_data: dict,
    model,
    tokenizer,
    layers: list[int],
    out_dir: Path,
    self_hi: float,
    ent_q: float,
    seed: int,
):
    """Extract confidence directions for specified layers."""
    K = len(pairs_df)

    for layer in layers:
        print(f"\n{'=' * 40}")
        print(f"Extracting direction for layer {layer}")
        print(f"{'=' * 40}")

        H_A_list = []
        H_B_list = []

        for _, row in tqdm(pairs_df.iterrows(), total=K, desc=f"Layer {layer}"):
            a_qid = row["A_qid"]
            b_qid = row["B_qid"]

            a_text = get_question_text(a_qid, compiled_data)
            b_text = get_question_text(b_qid, compiled_data)

            if not a_text or not b_text:
                print(f"Warning: Missing text for {a_qid} or {b_qid}")
                continue

            prompt_a = build_self_prompt(a_text)
            prompt_b = build_self_prompt(b_text)

            h_a = extract_hidden_state(model, tokenizer, prompt_a, layer)
            h_b = extract_hidden_state(model, tokenizer, prompt_b, layer)

            H_A_list.append(h_a)
            H_B_list.append(h_b)

        # Check we have enough valid pairs
        K_valid = len(H_A_list)
        if K_valid < 5:
            raise ValueError(f"Too few valid pairs with text: {K_valid}. Need at least 5.")

        if K_valid < K:
            print(f"Warning: Only {K_valid}/{K} pairs had valid text")

        H_A = torch.stack(H_A_list).float().cpu()
        H_B = torch.stack(H_B_list).float().cpu()

        # Compute direction
        D = H_A - H_B
        d_vec = D.mean(dim=0)
        d_norm = d_vec.norm().item()

        # Guard against zero norm
        if d_norm < 1e-8:
            raise ValueError(f"Direction norm is ~0 ({d_norm}). Cannot normalize.")

        d_unit = d_vec / d_vec.norm()

        print(f"||d_vec||: {d_norm:.4f}")

        # Split-half stability
        n = len(D)
        half = n // 2
        d_half1 = D[:half].mean(dim=0)
        d_half2 = D[half:].mean(dim=0)

        # Guard split-half norms
        if d_half1.norm() < 1e-8 or d_half2.norm() < 1e-8:
            split_cos = 0.0
            print("Warning: Split-half norm ~0, setting split_cos=0")
        else:
            d_half1_unit = d_half1 / d_half1.norm()
            d_half2_unit = d_half2 / d_half2.norm()
            split_cos = (d_half1_unit @ d_half2_unit).item()
        print(f"Split-half cosine: {split_cos:.4f}")

        # Compare with original d_conf
        original_path = out_dir / f"confidence_direction_layer{layer}.pt"
        cos_with_original = None
        if original_path.exists():
            orig_data = torch.load(original_path, map_location="cpu", weights_only=False)
            if isinstance(orig_data, dict) and "direction" in orig_data:
                d_orig = orig_data["direction"].view(-1).float()
            elif isinstance(orig_data, torch.Tensor):
                d_orig = orig_data.view(-1).float()
            else:
                d_orig = None

            if d_orig is not None:
                cos_with_original = (d_unit.cpu().float() @ d_orig).item() / (d_unit.norm().item() * d_orig.norm().item())
                print(f"cos(d_relaxed, d_original): {cos_with_original:.4f}")

        # Save (use K_valid in filename)
        out_path = out_dir / f"confidence_direction_layer{layer}_relaxed_S{self_hi}_E{ent_q}_N{K_valid}.pt"
        torch.save({
            "direction": d_unit,
            "layer": layer,
            "K": K_valid,
            "K_original": K,
            "self_hi": self_hi,
            "ent_q": ent_q,
            "seed": seed,
            "norm": d_norm,
            "split_half_cos": split_cos,
            "cos_with_original": cos_with_original,
            "timestamp": datetime.now().isoformat(),
        }, out_path)
        print(f"Saved: {out_path}")


# --------- MAIN ---------
def main():
    parser = argparse.ArgumentParser(
        description="Build relaxed contrastive pairs and extract confidence directions"
    )
    parser.add_argument("--unified-csv", type=str, default=UNIFIED_CSV_DEFAULT)
    parser.add_argument("--self-col", type=str, default="SelfProb")
    parser.add_argument("--entropy-col", type=str, default="entropy")
    parser.add_argument("--qid-col", type=str, default="auto")
    parser.add_argument("--self-hi", type=float, default=0.75)
    parser.add_argument("--ent-q", type=float, default=0.75)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-pairs-csv", type=str, default=EXCLUDE_CSV_DEFAULT)
    parser.add_argument("--out-pairs-csv", type=str, default=None)
    parser.add_argument("--out-meta-json", type=str, default=None)
    parser.add_argument("--count-only", action="store_true", help="Print counts and exit")
    parser.add_argument("--extract-direction", action="store_true")
    parser.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--layers", type=int, nargs="+", default=[35])
    parser.add_argument("--direction-out-dir", type=str, default="interp/outputs")
    args = parser.parse_args()

    print("=" * 60)
    print("BUILD RELAXED CONFIDENCE PAIRS")
    print("=" * 60)

    # Load unified CSV
    print(f"\nLoading {args.unified_csv}...")
    df = pd.read_csv(args.unified_csv)
    print(f"Loaded {len(df)} rows")

    # Detect QID column
    if args.qid_col == "auto":
        qid_col = detect_qid_column(df)
    else:
        qid_col = args.qid_col
    print(f"QID column: {qid_col}")

    # Validate columns
    for col in [args.self_col, args.entropy_col, qid_col]:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    # Load excluded QIDs (needed for count-only too)
    excluded_qids = load_excluded_qids(Path(args.exclude_pairs_csv))
    print(f"Excluded QIDs: {len(excluded_qids)}")

    # Count-only mode
    if args.count_only:
        run_count_mode(df, args.self_col, args.entropy_col, qid_col, excluded_qids)
        return

    # excluded_qids already loaded above

    # Build pairs
    pairs_df, meta = build_pairs(
        df=df,
        qid_col=qid_col,
        self_col=args.self_col,
        entropy_col=args.entropy_col,
        self_hi=args.self_hi,
        ent_q=args.ent_q,
        max_pairs=args.max_pairs,
        excluded_qids=excluded_qids,
        seed=args.seed,
    )

    K = len(pairs_df)
    print(f"\nBuilt {K} pairs")

    # Output paths
    if args.out_pairs_csv is None:
        out_pairs_csv = Path(f"contrastive_pairs/{MODEL_NAME}/{MODEL_NAME}_introspective_extremes_AB_train_relaxed_S{args.self_hi}_E{args.ent_q}_N{K}.csv")
    else:
        out_pairs_csv = Path(args.out_pairs_csv)

    if args.out_meta_json is None:
        out_meta_json = Path(f"interp/outputs/relaxed_pairs_meta_S{args.self_hi}_E{args.ent_q}_N{K}.json")
    else:
        out_meta_json = Path(args.out_meta_json)

    # Save pairs CSV
    out_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(out_pairs_csv, index=False)
    print(f"Saved pairs: {out_pairs_csv}")

    # Save meta JSON
    meta["pairs_csv"] = str(out_pairs_csv)
    meta["timestamp"] = datetime.now().isoformat()
    out_meta_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta: {out_meta_json}")

    # Extract directions
    if args.extract_direction:
        # Load compiled JSON for question text
        print(f"\nLoading {COMPILED_JSON}...")
        with open(COMPILED_JSON) as f:
            compiled_data = json.load(f)

        tokenizer, model = load_model_and_tokenizer(args.model_id)

        out_dir = Path(args.direction_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        extract_directions(
            pairs_df=pairs_df,
            compiled_data=compiled_data,
            model=model,
            tokenizer=tokenizer,
            layers=args.layers,
            out_dir=out_dir,
            self_hi=args.self_hi,
            ent_q=args.ent_q,
            seed=args.seed,
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
