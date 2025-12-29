#!/usr/bin/env python3
"""
build_d_conf_bad.py

Construct d_conf_bad from miscalibrated samples to test whether d_conf
captures internal certainty signal vs output style.

Miscalibrated samples:
- Overconfident: high self-confidence + high entropy (model says confident but is uncertain)
- Underconfident: low self-confidence + low entropy (model says uncertain but is certain)

If cosine(d_conf_bad, d_conf) is HIGH: d_conf might just be "output style"
If cosine(d_conf_bad, d_conf) is LOW: d_conf captures internal signal, not style

IMPORTANT: Uses train/eval split to avoid contamination:
- Direction is computed from TRAIN split only
- Steering evaluation should use EVAL split only
- QID lists are saved for reproducibility

Usage:
    python interp/build_d_conf_bad.py --layer 35
    python interp/build_d_conf_bad.py --layer 35 --dry-run  # Just count samples, no GPU

For steering, use:
    python interp/steer_activations.py --direction-path interp/outputs/d_conf_bad_layer35.pt ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add interp directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from prompt_utils import build_self_prompt

# --------- CONFIG ---------
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"

DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
UNIFIED_CSV = DATA_DIR / f"{MODEL_NAME}_unified.csv"
TRAIN_CSV = DATA_DIR / f"{MODEL_NAME}_introspective_extremes_AB_train.csv"
COMPILED_JSON = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"

OUTPUT_DIR = Path("interp/outputs")

# Relaxed thresholds (from sample size analysis)
OVERCONF_SELF_MIN = 0.75  # High confidence
UNDERCONF_SELF_MAX = 0.25  # Low confidence
ENTROPY_QUANTILE_HIGH = 0.67  # P67 for overconfident
ENTROPY_QUANTILE_LOW = 0.33   # P33 for underconfident

# Train/eval split ratio
TRAIN_RATIO = 0.5  # 50% train, 50% eval

# Minimum samples per bin AFTER split
MIN_SAMPLES_PER_BIN = 8


def load_contaminated_qids() -> set[str]:
    """Load QIDs used to train d_conf (to exclude from d_conf_bad)."""
    if not TRAIN_CSV.exists():
        print(f"Warning: Train CSV not found: {TRAIN_CSV}")
        return set()

    contaminated = set()
    df = pd.read_csv(TRAIN_CSV)
    for col in ["A_qid", "B_qid"]:
        if col in df.columns:
            contaminated.update(df[col].dropna().astype(str).tolist())
    return contaminated


def load_question_text() -> dict[str, dict]:
    """Load question text and options from compiled JSON."""
    if not COMPILED_JSON.exists():
        raise FileNotFoundError(f"Compiled JSON not found: {COMPILED_JSON}")

    with open(COMPILED_JSON) as f:
        data = json.load(f)

    q_map = {}
    for qid, res in data.get("results", {}).items():
        q_data = res.get("question", {})
        if isinstance(q_data, dict):
            q_stem = (q_data.get("question") or q_data.get("text") or "").strip()
            opts = q_data.get("options")

            # Format MCQ with options
            if isinstance(opts, dict) and opts:
                lines = [q_stem, ""]
                for k in ["A", "B", "C", "D"]:
                    if k in opts:
                        lines.append(f"{k}) {opts[k]}")
                q_text = "\n".join(lines).strip()
            elif isinstance(opts, list) and len(opts) >= 2:
                lines = [q_stem, ""]
                for k, opt in zip(["A", "B", "C", "D"], opts[:4]):
                    lines.append(f"{k}) {opt}")
                q_text = "\n".join(lines).strip()
            else:
                q_text = q_stem

            if q_text:
                q_map[qid] = {"text": q_text}

    return q_map


def filter_miscalibrated(
    df: pd.DataFrame,
    contaminated_qids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter for miscalibrated samples using relaxed thresholds.

    IMPORTANT: Computes entropy quantiles AFTER excluding contaminated QIDs
    to ensure thresholds are based on the clean dataset.

    Returns:
        (overconfident_df, underconfident_df)
    """
    # First, exclude contaminated QIDs
    df_clean = df[~df["question_id"].isin(contaminated_qids)].copy()
    print(f"After excluding contaminated: {len(df_clean)} rows (was {len(df)})")

    # Compute entropy quantiles on CLEAN data
    p_high = df_clean["entropy"].quantile(ENTROPY_QUANTILE_HIGH)
    p_low = df_clean["entropy"].quantile(ENTROPY_QUANTILE_LOW)

    print(f"Entropy thresholds (on clean data): P{int(ENTROPY_QUANTILE_LOW*100)}={p_low:.4f}, P{int(ENTROPY_QUANTILE_HIGH*100)}={p_high:.4f}")
    print(f"Confidence thresholds: overconf >= {OVERCONF_SELF_MIN}, underconf <= {UNDERCONF_SELF_MAX}")

    # Overconfident: high confidence + high entropy
    overconf = df_clean[
        (df_clean["SelfProb"] >= OVERCONF_SELF_MIN) &
        (df_clean["entropy"] >= p_high)
    ]

    # Underconfident: low confidence + low entropy
    underconf = df_clean[
        (df_clean["SelfProb"] <= UNDERCONF_SELF_MAX) &
        (df_clean["entropy"] <= p_low)
    ]

    return overconf, underconf


def train_eval_split(
    overconf: pd.DataFrame,
    underconf: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split overconf and underconf into train and eval sets.

    Returns:
        (overconf_train, overconf_eval, underconf_train, underconf_eval)
    """
    # Shuffle and split overconf
    overconf_shuffled = overconf.sample(frac=1, random_state=seed)
    n_train_over = int(len(overconf_shuffled) * train_ratio)
    overconf_train = overconf_shuffled.iloc[:n_train_over]
    overconf_eval = overconf_shuffled.iloc[n_train_over:]

    # Shuffle and split underconf
    underconf_shuffled = underconf.sample(frac=1, random_state=seed)
    n_train_under = int(len(underconf_shuffled) * train_ratio)
    underconf_train = underconf_shuffled.iloc[:n_train_under]
    underconf_eval = underconf_shuffled.iloc[n_train_under:]

    return overconf_train, overconf_eval, underconf_train, underconf_eval


def load_model():
    """Load 4-bit quantized model."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def extract_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
) -> torch.Tensor:
    """
    Extract hidden states at specified layer for all prompts.

    Returns: tensor of shape (n_prompts, hidden_dim)
    """
    activations = []

    # Handle device_map="auto" safely
    target_device = model.model.embed_tokens.weight.device

    for prompt in tqdm(prompts, desc=f"Extracting L{layer_idx}"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)

        # hidden_states[layer_idx + 1] is output of layer layer_idx
        # (hidden_states[0] is embeddings)
        h = outputs.hidden_states[layer_idx + 1][:, -1, :].cpu().squeeze(0)
        activations.append(h)

    return torch.stack(activations)


def load_direction(name: str, layer_idx: int) -> torch.Tensor | None:
    """Load a saved direction vector."""
    name_map = {
        "conf": f"confidence_direction_layer{layer_idx}.pt",
        "pass": f"pass_game_direction_layer{layer_idx}.pt",
        "so": f"self_other_direction_layer{layer_idx}.pt",
    }

    if name not in name_map:
        return None

    path = OUTPUT_DIR / name_map[name]
    if not path.exists():
        print(f"Warning: Direction not found: {path}")
        return None

    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        return data.view(-1)
    elif isinstance(data, dict) and "direction" in data:
        return data["direction"].view(-1)
    else:
        print(f"Warning: Unknown format in {path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Build d_conf_bad from miscalibrated samples")
    parser.add_argument("--layer", type=int, default=35, help="Layer to extract activations from")
    parser.add_argument("--dry-run", action="store_true", help="Just count samples, don't extract activations")
    parser.add_argument("--balance", action="store_true", help="Balance to equal N per bin")
    args = parser.parse_args()

    print("=" * 70)
    print("BUILD d_conf_bad: Miscalibrated Direction Vector")
    print("=" * 70)

    # Load data
    print(f"\nLoading unified CSV from {UNIFIED_CSV}...")
    if not UNIFIED_CSV.exists():
        raise FileNotFoundError(f"Unified CSV not found: {UNIFIED_CSV}")
    df = pd.read_csv(UNIFIED_CSV)
    print(f"  Loaded {len(df)} rows")

    # Load contaminated QIDs
    contaminated = load_contaminated_qids()
    print(f"  Excluding {len(contaminated)} contaminated QIDs")

    # Filter miscalibrated
    overconf, underconf = filter_miscalibrated(df, contaminated)
    print(f"\nMiscalibrated samples:")
    print(f"  Overconfident:  {len(overconf)}")
    print(f"  Underconfident: {len(underconf)}")
    print(f"  Total:          {len(overconf) + len(underconf)}")

    if len(overconf) < 5 or len(underconf) < 5:
        print("\nERROR: Not enough samples. Adjust thresholds or skip this experiment.")
        return

    # Balance if requested
    if args.balance:
        min_n = min(len(overconf), len(underconf))
        overconf = overconf.sample(n=min_n, random_state=42)
        underconf = underconf.sample(n=min_n, random_state=42)
        print(f"\nBalanced to {min_n} per bin ({min_n * 2} total)")

    # Train/eval split (CRITICAL: avoid contamination)
    overconf_train, overconf_eval, underconf_train, underconf_eval = train_eval_split(
        overconf, underconf
    )
    print(f"\nTrain/eval split:")
    print(f"  Overconf:  {len(overconf_train)} train / {len(overconf_eval)} eval")
    print(f"  Underconf: {len(underconf_train)} train / {len(underconf_eval)} eval")
    print(f"  Total:     {len(overconf_train) + len(underconf_train)} train / {len(overconf_eval) + len(underconf_eval)} eval")

    # Check minimum sample size AFTER split
    min_train = min(len(overconf_train), len(underconf_train))
    min_eval = min(len(overconf_eval), len(underconf_eval))

    if min_train < MIN_SAMPLES_PER_BIN:
        print(f"\nERROR: Not enough TRAIN samples per bin after split.")
        print(f"  Minimum required: {MIN_SAMPLES_PER_BIN}")
        print(f"  Got: overconf={len(overconf_train)}, underconf={len(underconf_train)}")
        print(f"  Adjust thresholds or skip this experiment.")
        return

    if min_eval < MIN_SAMPLES_PER_BIN:
        print(f"\nERROR: Not enough EVAL samples per bin after split.")
        print(f"  Minimum required: {MIN_SAMPLES_PER_BIN}")
        print(f"  Got: overconf={len(overconf_eval)}, underconf={len(underconf_eval)}")
        print(f"  Adjust thresholds or skip this experiment.")
        return

    # Collect QID lists for reproducibility
    train_qids = {
        "overconf": overconf_train["question_id"].tolist(),
        "underconf": underconf_train["question_id"].tolist(),
    }
    eval_qids = {
        "overconf": overconf_eval["question_id"].tolist(),
        "underconf": underconf_eval["question_id"].tolist(),
    }

    # Flatten eval QIDs for steering script
    eval_qids_flat = eval_qids["overconf"] + eval_qids["underconf"]

    if args.dry_run:
        print("\n[DRY RUN] Stopping before GPU work.")
        print(f"  Would use {len(train_qids['overconf']) + len(train_qids['underconf'])} samples for direction")
        print(f"  Would save {len(eval_qids_flat)} samples for steering eval")
        return

    # Load question text
    print("\nLoading question text...")
    q_text_map = load_question_text()

    # Build prompts (TRAIN only for direction computation)
    overconf_prompts = []
    overconf_qids_used = []
    underconf_prompts = []
    underconf_qids_used = []

    for _, row in overconf_train.iterrows():
        qid = row["question_id"]
        if qid in q_text_map:
            overconf_prompts.append(build_self_prompt(q_text_map[qid]["text"]))
            overconf_qids_used.append(qid)

    for _, row in underconf_train.iterrows():
        qid = row["question_id"]
        if qid in q_text_map:
            underconf_prompts.append(build_self_prompt(q_text_map[qid]["text"]))
            underconf_qids_used.append(qid)

    print(f"\nBuilt TRAIN prompts: {len(overconf_prompts)} overconf, {len(underconf_prompts)} underconf")

    if len(overconf_prompts) < 3 or len(underconf_prompts) < 3:
        print("\nERROR: Not enough TRAIN prompts after filtering. Check question_id mapping.")
        return

    # Load model
    tokenizer, model = load_model()

    # Extract activations (TRAIN only)
    print(f"\nExtracting activations at layer {args.layer} (TRAIN set)...")
    h_overconf = extract_activations(model, tokenizer, overconf_prompts, args.layer)
    h_underconf = extract_activations(model, tokenizer, underconf_prompts, args.layer)

    print(f"  Overconf activations: {h_overconf.shape}")
    print(f"  Underconf activations: {h_underconf.shape}")

    # Compute d_conf_bad = mean(overconf) - mean(underconf)
    # Note: overconf has HIGH confidence, underconf has LOW confidence
    # So d_conf_bad points from low-conf to high-conf (same convention as d_conf)
    d_conf_bad = h_overconf.mean(dim=0) - h_underconf.mean(dim=0)

    # Normalize
    d_conf_bad_norm = d_conf_bad / d_conf_bad.norm()

    print(f"\nd_conf_bad computed:")
    print(f"  Raw norm:        {d_conf_bad.norm().item():.4f}")
    print(f"  Normalized norm: {d_conf_bad_norm.norm().item():.4f}")

    # Compare with other directions
    print("\n" + "-" * 50)
    print("COSINE SIMILARITY WITH OTHER DIRECTIONS")
    print("-" * 50)

    comparisons = {}
    for name in ["conf", "pass", "so"]:
        d_other = load_direction(name, args.layer)
        if d_other is not None:
            d_other_norm = d_other / d_other.norm()
            cos = F.cosine_similarity(d_conf_bad_norm.unsqueeze(0), d_other_norm.unsqueeze(0)).item()
            comparisons[f"d_{name}"] = cos
            print(f"  cosine(d_conf_bad, d_{name}): {cos:.4f}")

    # Save direction
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"d_conf_bad_layer{args.layer}.pt"

    save_data = {
        "direction": d_conf_bad_norm,
        "raw_direction": d_conf_bad,
        "layer": args.layer,
        "n_overconf_train": len(overconf_prompts),
        "n_underconf_train": len(underconf_prompts),
        "train_qids": {
            "overconf": overconf_qids_used,
            "underconf": underconf_qids_used,
        },
        "eval_qids": eval_qids,  # For steering evaluation
        "thresholds": {
            "overconf_self_min": OVERCONF_SELF_MIN,
            "underconf_self_max": UNDERCONF_SELF_MAX,
            "entropy_quantile_high": ENTROPY_QUANTILE_HIGH,
            "entropy_quantile_low": ENTROPY_QUANTILE_LOW,
        },
        "cosine_similarities": comparisons,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(save_data, out_path)
    print(f"\nSaved to: {out_path}")

    # Save JSON summary
    json_path = OUTPUT_DIR / f"d_conf_bad_layer{args.layer}_summary.json"
    json_data = {
        "layer": args.layer,
        "n_overconf_train": len(overconf_prompts),
        "n_underconf_train": len(underconf_prompts),
        "n_overconf_eval": len(eval_qids["overconf"]),
        "n_underconf_eval": len(eval_qids["underconf"]),
        "train_qids": {
            "overconf": overconf_qids_used,
            "underconf": underconf_qids_used,
        },
        "eval_qids": eval_qids,
        "thresholds": {
            "overconf_self_min": OVERCONF_SELF_MIN,
            "underconf_self_max": UNDERCONF_SELF_MAX,
            "entropy_quantile_high": ENTROPY_QUANTILE_HIGH,
            "entropy_quantile_low": ENTROPY_QUANTILE_LOW,
        },
        "cosine_similarities": comparisons,
        "interpretation": (
            "HIGH cosine with d_conf suggests d_conf captures output style. "
            "LOW cosine suggests d_conf captures internal signal."
        ),
        "note": "Direction computed from TRAIN qids only. Use EVAL qids for steering evaluation.",
        "timestamp": datetime.now().isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved summary to: {json_path}")

    # Save eval QIDs to separate file for steering script
    eval_qids_path = OUTPUT_DIR / f"d_conf_bad_layer{args.layer}_eval_qids.json"
    with open(eval_qids_path, "w") as f:
        json.dump(eval_qids_flat, f, indent=2)
    print(f"Saved eval QIDs to: {eval_qids_path}")

    # Print QID counts for clarity
    print(f"\nQID lists saved:")
    print(f"  Train: {len(overconf_qids_used)} overconf + {len(underconf_qids_used)} underconf")
    print(f"  Eval:  {len(eval_qids['overconf'])} overconf + {len(eval_qids['underconf'])} underconf = {len(eval_qids_flat)} total")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if "d_conf" in comparisons:
        cos_conf = comparisons["d_conf"]
        if abs(cos_conf) > 0.7:
            print(f"\nHIGH alignment with d_conf (cos={cos_conf:.3f})")
            print("This suggests d_conf may capture 'output style' rather than internal signal.")
        elif abs(cos_conf) < 0.3:
            print(f"\nLOW alignment with d_conf (cos={cos_conf:.3f})")
            print("This suggests d_conf captures internal certainty signal, not just style.")
        else:
            print(f"\nMODERATE alignment with d_conf (cos={cos_conf:.3f})")
            print("Mixed evidence — d_conf may partially capture both style and signal.")


if __name__ == "__main__":
    main()
