#!/usr/bin/env python3
"""
residual_confidence_direction.py

Compute the residual confidence direction after projecting out d_conf.
Tests whether confidence has multiple components beyond the learned d_conf.

Usage:
  python interp/residual_confidence_direction.py --layer 35 --n 5   # smoke test
  python interp/residual_confidence_direction.py --layer 35          # full run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
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

    if len(pairs) < 5:
        raise ValueError(f"Too few pairs: {len(pairs)}. Need at least 5.")

    return pairs


# --------- Hidden State Extraction ---------
def extract_hidden_state(
    model, tokenizer, prompt: str, layer_idx: int, use_chat_template: bool
) -> torch.Tensor:
    """Extract last-token hidden state at specified layer."""

    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

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


# --------- Metrics ---------
def compute_auc_manual(scores_high: torch.Tensor, scores_low: torch.Tensor) -> float:
    """Compute AUC using Mann-Whitney U statistic (rank-based, no sklearn).

    AUC = P(score_high > score_low) for random pair.
    """
    n_high = len(scores_high)
    n_low = len(scores_low)

    # Count how many (high, low) pairs where high > low
    count = 0
    for s_h in scores_high:
        for s_l in scores_low:
            if s_h > s_l:
                count += 1
            elif s_h == s_l:
                count += 0.5

    auc = count / (n_high * n_low)
    return auc


def compute_cohens_d(scores_high: torch.Tensor, scores_low: torch.Tensor) -> float:
    """Compute Cohen's d effect size."""
    mean_high = scores_high.mean()
    mean_low = scores_low.mean()

    var_high = scores_high.var()
    var_low = scores_low.var()
    n_high = len(scores_high)
    n_low = len(scores_low)

    # Pooled standard deviation
    pooled_var = ((n_high - 1) * var_high + (n_low - 1) * var_low) / (n_high + n_low - 2)
    pooled_std = torch.sqrt(pooled_var)

    if pooled_std < 1e-8:
        return 0.0

    return ((mean_high - mean_low) / pooled_std).item()


# --------- Main ---------
def main():
    parser = argparse.ArgumentParser(
        description="Compute residual confidence direction after projecting out d_conf"
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--pairs-csv", type=str, default=str(PAIRS_CSV))
    parser.add_argument("--compiled-json", type=str, default=str(COMPILED_JSON))
    parser.add_argument("--dconf-path", type=str, default=None)
    parser.add_argument("--n", type=int, default=None, help="Limit pairs (smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--out-dir", type=str, default="interp/outputs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default d_conf path
    if args.dconf_path is None:
        dconf_path = DIRECTION_DIR / f"confidence_direction_layer{args.layer}.pt"
    else:
        dconf_path = Path(args.dconf_path)

    print("=" * 60)
    print("RESIDUAL CONFIDENCE DIRECTION EXPERIMENT")
    print("=" * 60)
    print(f"Layer: {args.layer}")
    print(f"d_conf path: {dconf_path}")
    print(f"Chat template: {not args.no_chat_template}")
    print(f"N limit: {args.n or 'all'}")
    print("=" * 60)

    # 1. Load pairs
    pairs = load_pairs(Path(args.pairs_csv), Path(args.compiled_json), args.n)

    # 2. Load model
    tokenizer, model = load_model_and_tokenizer(args.model_id)

    # 3. Extract hidden states
    print(f"\nExtracting hidden states at layer {args.layer}...")
    H_high_list = []
    H_low_list = []

    for pair in tqdm(pairs, desc="Pairs"):
        prompt_high = build_self_prompt(pair["high_text"])
        prompt_low = build_self_prompt(pair["low_text"])

        h_high = extract_hidden_state(
            model, tokenizer, prompt_high, args.layer, not args.no_chat_template
        )
        h_low = extract_hidden_state(
            model, tokenizer, prompt_low, args.layer, not args.no_chat_template
        )

        H_high_list.append(h_high)
        H_low_list.append(h_low)

    H_high = torch.stack(H_high_list)  # (N, d)
    H_low = torch.stack(H_low_list)    # (N, d)
    print(f"H_high shape: {H_high.shape}")

    # 4. Load d_conf
    print(f"\nLoading d_conf from {dconf_path}...")
    if not dconf_path.exists():
        raise FileNotFoundError(f"d_conf not found: {dconf_path}")

    data = torch.load(dconf_path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        d_conf = data
    elif isinstance(data, dict) and "direction" in data:
        d_conf = data["direction"]
    else:
        raise ValueError(f"Unknown d_conf format: {type(data)}")

    # Normalize and move to same device/dtype
    d_conf = d_conf.view(-1).float()
    d_conf_norm = d_conf.norm().item()
    d_conf_unit = (d_conf / d_conf.norm()).to(H_high.device).to(H_high.dtype)
    print(f"||d_conf||: {d_conf_norm:.4f}")

    # 5. Compute projections and residuals
    # proj(h) = (h · v) * v
    # Move to CPU float32 for stability
    H_high_f = H_high.float().cpu()
    H_low_f = H_low.float().cpu()
    v = d_conf_unit.float().cpu()

    # Project out d_conf
    proj_high = (H_high_f @ v).unsqueeze(1) * v.unsqueeze(0)  # (N, d)
    proj_low = (H_low_f @ v).unsqueeze(1) * v.unsqueeze(0)    # (N, d)

    H_high_perp = H_high_f - proj_high
    H_low_perp = H_low_f - proj_low

    # 6. Compute residual direction
    d_res = H_high_perp.mean(dim=0) - H_low_perp.mean(dim=0)
    d_res_norm = d_res.norm().item()

    # Guard against tiny residual (would cause NaN)
    if d_res_norm < 1e-8:
        print("\n⚠ Residual norm ~0 — confidence is essentially 1D along d_conf")
        d_res_unit = torch.zeros_like(d_res)
    else:
        d_res_unit = d_res / d_res.norm()

    print(f"\n||d_res|| (before normalization): {d_res_norm:.4f}")

    # 7. Compute separability metrics

    # Baseline d_conf separability (on original H)
    s_high_conf = (H_high_f @ v).squeeze()
    s_low_conf = (H_low_f @ v).squeeze()

    gap_conf = (s_high_conf.mean() - s_low_conf.mean()).item()
    cohens_d_conf = compute_cohens_d(s_high_conf, s_low_conf)
    auc_conf = compute_auc_manual(s_high_conf, s_low_conf)

    # Residual d_res separability (on residualized H)
    s_high_res = (H_high_perp @ d_res_unit).squeeze()
    s_low_res = (H_low_perp @ d_res_unit).squeeze()

    gap_res = (s_high_res.mean() - s_low_res.mean()).item()
    cohens_d_res = compute_cohens_d(s_high_res, s_low_res)
    auc_res = compute_auc_manual(s_high_res, s_low_res)

    # 8. Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"N pairs: {len(pairs)}")
    print(f"\n||d_conf||: {d_conf_norm:.4f}")
    print(f"||d_res||:  {d_res_norm:.4f}")

    print(f"\n--- Baseline d_conf separability ---")
    print(f"  Gap (mean_high - mean_low): {gap_conf:.4f}")
    print(f"  Cohen's d: {cohens_d_conf:.4f}")
    print(f"  AUC: {auc_conf:.4f}")

    print(f"\n--- Residual d_res separability ---")
    print(f"  Gap (mean_high - mean_low): {gap_res:.4f}")
    print(f"  Cohen's d: {cohens_d_res:.4f}")
    print(f"  AUC: {auc_res:.4f}")

    # Interpretation
    print("\n--- INTERPRETATION ---")
    if d_res_norm < 0.1:
        print("✓ ||d_res|| tiny — confidence is essentially 1D along d_conf")
    elif auc_res < 0.6:
        print("✓ Residual AUC ~0.5 — d_conf captures most of the separability")
    else:
        print(f"⚠ Residual AUC = {auc_res:.3f} > 0.5 — there may be a second confidence component")

    if auc_res > 0.7:
        print("⚠️ Strong residual signal! Confidence is multi-dimensional.")

    # Cosine between d_conf and d_res (should be ~0 by construction)
    cos_conf_res = (v @ d_res_unit).item()
    print(f"\ncos(d_conf, d_res): {cos_conf_res:.4f} (should be ~0)")

    print("=" * 60)

    # 9. Save outputs
    # Save direction vector
    direction_out = out_dir / f"confidence_residual_direction_layer{args.layer}.pt"
    torch.save({
        "direction": d_res_unit,
        "norm": d_res_norm,
        "layer": args.layer,
        "n_pairs": len(pairs),
    }, direction_out)
    print(f"\nSaved direction to: {direction_out}")

    # Save metrics JSON
    metrics = {
        "config": {
            "model_id": args.model_id,
            "layer": args.layer,
            "n_pairs": len(pairs),
            "use_chat_template": not args.no_chat_template,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "norms": {
            "d_conf": d_conf_norm,
            "d_res": d_res_norm,
        },
        "baseline_d_conf": {
            "gap": gap_conf,
            "cohens_d": cohens_d_conf,
            "auc": auc_conf,
        },
        "residual_d_res": {
            "gap": gap_res,
            "cohens_d": cohens_d_res,
            "auc": auc_res,
        },
        "cos_dconf_dres": cos_conf_res,
        "qids_used": [p["high_qid"] for p in pairs] + [p["low_qid"] for p in pairs],
    }

    metrics_out = out_dir / f"confidence_residual_metrics_layer{args.layer}.json"
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_out}")


if __name__ == "__main__":
    main()
