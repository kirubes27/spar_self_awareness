#!/usr/bin/env python3
"""
ablate_entropy_conf_correlation.py

Critical ablation experiment (per Chris):
Project out d_conf at a layer and test whether the correlation between
explicit self-confidence (A–H) and baseline entropy decreases.

This tests INTROSPECTIVE ACCESS: does removing the confidence direction
break the model's ability to report uncertainty that matches its internal entropy?

Usage:
  python interp/ablate_entropy_conf_correlation.py --layer 35 --direction conf --n 20  # smoke test
  python interp/ablate_entropy_conf_correlation.py --layer 35 --direction conf          # full run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add interp directory to path for imports when running from repo root
sys.path.insert(0, str(Path(__file__).parent))
from prompt_utils import build_self_prompt

# --------- CONFIG ---------
MODEL_ID_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"

DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
UNIFIED_CSV = DATA_DIR / f"{MODEL_NAME}_unified.csv"
TRAIN_CSV = DATA_DIR / f"{MODEL_NAME}_introspective_extremes_AB_train.csv"
COMPILED_JSON = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"

DIRECTION_DIR = Path("interp/outputs")
OUTPUT_DIR = Path("interp/outputs")

# Confidence midpoints (A-H scale)
CONF_MIDPOINTS = {
    "A": 0.025,
    "B": 0.075,
    "C": 0.15,
    "D": 0.30,
    "E": 0.50,
    "F": 0.70,
    "G": 0.85,
    "H": 0.95,
}


# --------- Contamination Hygiene ---------
def load_contaminated_qids() -> Set[str]:
    """
    Load the question IDs used to train d_conf (from introspective_extremes_AB_train.csv).
    These must be excluded from evaluation to avoid train-test leakage.
    """
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Train CSV not found: {TRAIN_CSV}.\n"
            "Cannot proceed without contamination filter. "
            "Run from repo root directory."
        )

    contaminated = set()
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a_qid = (row.get("A_qid") or "").strip()
            b_qid = (row.get("B_qid") or "").strip()
            if a_qid:
                contaminated.add(a_qid)
            if b_qid:
                contaminated.add(b_qid)

    return contaminated


# --------- Data Loading ---------
def load_questions(
    n: Optional[int], seed: int, contaminated_qids: Set[str]
) -> List[Dict]:
    """
    Load questions from unified CSV (for entropy) and compiled JSON (for question_text).
    Excludes contaminated QIDs.
    Returns list of dicts with question_id, question_text, entropy.
    """
    import pandas as pd

    print(f"Loading entropy from {UNIFIED_CSV}...")
    df = pd.read_csv(UNIFIED_CSV)

    # Ensure required columns exist
    if "question_id" not in df.columns or "entropy" not in df.columns:
        raise ValueError("Missing required columns: question_id, entropy")

    # Load question text from compiled JSON (CSV has all NaN for question_text)
    print(f"Loading question text from {COMPILED_JSON}...")
    if not COMPILED_JSON.exists():
        raise FileNotFoundError(f"Compiled JSON not found: {COMPILED_JSON}")

    with open(COMPILED_JSON) as f:
        compiled_data = json.load(f)

    # Build question text map from compiled results
    q_text_map = {}
    for qid, res in compiled_data.get("results", {}).items():
        q_data = res.get("question", {})
        if isinstance(q_data, dict):
            q_text = q_data.get("question", "")
        else:
            q_text = ""
        if q_text:
            q_text_map[qid] = q_text

    print(f"Loaded {len(q_text_map)} question texts from compiled JSON.")

    # Filter out contaminated QIDs
    initial_count = len(df)
    df = df[~df["question_id"].isin(contaminated_qids)]
    filtered_count = len(df)
    print(f"Excluded {initial_count - filtered_count} contaminated QIDs. Remaining: {filtered_count}")

    # Build question list, joining entropy from CSV with text from JSON
    questions = []
    missing_text_count = 0
    for _, row in df.iterrows():
        qid = row["question_id"]
        entropy = row["entropy"]
        q_text = q_text_map.get(qid, "")
        if not q_text:
            missing_text_count += 1
            continue  # Skip questions without text
        questions.append({
            "question_id": qid,
            "question_text": q_text,
            "entropy": entropy,
        })

    if missing_text_count > 0:
        print(f"WARNING: Skipped {missing_text_count} questions with missing text.")

    print(f"Final question count: {len(questions)}")

    # Deterministic sampling if n is specified
    if n is not None and len(questions) > n:
        import random
        random.seed(seed)
        questions = random.sample(questions, n)
        print(f"Sampled {n} questions (seed={seed})")

    return questions


# --------- Model Loading ---------
def load_model_and_tokenizer(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
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


# --------- Direction Loading ---------
def get_direction_path(direction_name: str, layer_idx: int) -> Path:
    """Map CLI direction name to file path."""
    if direction_name == "conf":
        return DIRECTION_DIR / f"confidence_direction_layer{layer_idx}.pt"
    elif direction_name == "pass":
        return DIRECTION_DIR / f"pass_game_direction_layer{layer_idx}.pt"
    elif direction_name == "so":
        return DIRECTION_DIR / f"self_other_direction_layer{layer_idx}.pt"
    elif direction_name.startswith("random_"):
        idx = direction_name.split("_")[1]
        # Use the correct layer for random directions
        return DIRECTION_DIR / f"random_direction_{idx}_layer{layer_idx}.pt"
    else:
        raise ValueError(f"Unknown direction: {direction_name}")


def load_direction(path: Path, device: torch.device) -> torch.Tensor:
    """Load direction vector from .pt file and normalize it."""
    if not path.exists():
        raise FileNotFoundError(f"Direction file not found: {path}")

    # CRITICAL: weights_only=False for PyTorch 2.6 compatibility
    data = torch.load(path, map_location=device, weights_only=False)

    if isinstance(data, torch.Tensor):
        v = data
    elif isinstance(data, dict) and "direction" in data:
        v = data["direction"]
    else:
        raise ValueError(f"Unexpected format in {path}")

    v = v.view(-1).to(device)

    # Normalize to unit vector (critical for projection-out math)
    norm = v.norm()
    if norm < 1e-8:
        raise ValueError(f"Direction vector has near-zero norm: {norm}")
    v = v / norm

    print(f"Loaded direction from {path} (original norm: {norm:.4f}, now normalized)")
    return v


# --------- Token IDs ---------
def get_letter_token_ids(tokenizer) -> Dict[str, int]:
    """
    Get token IDs for confidence letters A-H.
    Asserts each letter maps to exactly one token.
    """
    letter_ids = {}
    for letter in "ABCDEFGH":
        ids = tokenizer(letter, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(
                f"Letter '{letter}' tokenizes to {len(ids)} tokens: {ids}. "
                "Expected exactly 1 token per letter."
            )
        letter_ids[letter] = ids[0]
    return letter_ids


# --------- Ablation Hook ---------
def make_ablation_hook(direction_vec: torch.Tensor):
    """
    Create a forward hook that projects out the component of the hidden state
    along direction_vec for the LAST TOKEN only.

    Math: h' = h - (h · v) * v  (where v is unit-normalized)
    """
    # direction_vec is already normalized in load_direction
    v = direction_vec

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # hidden_states: (batch, seq, dim)
        v_unit = v.to(hidden_states.device, hidden_states.dtype).view(1, 1, -1)
        last_token = hidden_states[:, -1:, :]  # (batch, 1, dim)

        # Project out: h' = h - (h · v) * v
        coeff = (last_token * v_unit).sum(dim=-1, keepdim=True)  # (batch, 1, 1)
        projection = coeff * v_unit  # (batch, 1, dim)

        # Apply ablation only to last token
        new_hidden_states = hidden_states.clone()
        new_hidden_states[:, -1:, :] = last_token - projection

        if isinstance(output, tuple):
            return (new_hidden_states,) + output[1:]
        else:
            return new_hidden_states

    return hook


# --------- Forward Pass ---------
def run_inference(
    model,
    tokenizer,
    prompt: str,
    use_chat_template: bool,
    ablation_hook=None,
    layer_idx: int = None,
) -> torch.Tensor:
    """
    Run model forward pass and return logits for the last token.
    If ablation_hook is provided, applies it at layer_idx during inference.
    """
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    # Handle device_map="auto" safely: use embed_tokens device
    target_device = model.model.embed_tokens.weight.device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    if ablation_hook is not None and layer_idx is not None:
        layer_module = model.model.layers[layer_idx]
        handle = layer_module.register_forward_hook(ablation_hook)
    else:
        handle = None

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
    finally:
        if handle is not None:
            handle.remove()

    return logits.squeeze(0)  # (vocab_size,)


def decode_confidence_letter(logits: torch.Tensor, letter_ids: Dict[str, int]) -> str:
    """Decode confidence letter A-H from logits by argmax over letter token IDs."""
    letter_logits = torch.tensor(
        [logits[letter_ids[letter]].item() for letter in "ABCDEFGH"]
    )
    pred_idx = letter_logits.argmax().item()
    return "ABCDEFGH"[pred_idx]


# --------- Main Experiment ---------
def run_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    layer_idx: int,
    direction_vec: torch.Tensor,
    use_chat_template: bool,
) -> Dict:
    """
    Run the ablation experiment:
    - For each question, get baseline confidence and ablated confidence.
    - Compute correlations between confidence and entropy.
    """
    letter_ids = get_letter_token_ids(tokenizer)
    ablation_hook = make_ablation_hook(direction_vec)

    per_question = []
    n_invalid = 0

    for q in tqdm(questions, desc=f"Ablation L{layer_idx}"):
        qid = q["question_id"]
        entropy = q["entropy"]
        question_text = q["question_text"]

        prompt = build_self_prompt(question_text)

        # Baseline (no ablation)
        logits_base = run_inference(
            model, tokenizer, prompt, use_chat_template,
            ablation_hook=None, layer_idx=None
        )
        letter_base = decode_confidence_letter(logits_base, letter_ids)

        # Ablated (project out direction)
        logits_abl = run_inference(
            model, tokenizer, prompt, use_chat_template,
            ablation_hook=ablation_hook, layer_idx=layer_idx
        )
        letter_abl = decode_confidence_letter(logits_abl, letter_ids)

        per_question.append({
            "qid": qid,
            "entropy": entropy,
            "conf_letter_base": letter_base,
            "conf_letter_abl": letter_abl,
            "conf_val_base": CONF_MIDPOINTS[letter_base],
            "conf_val_abl": CONF_MIDPOINTS[letter_abl],
        })

    # Compute correlations
    entropies = [q["entropy"] for q in per_question]
    conf_base = [q["conf_val_base"] for q in per_question]
    conf_abl = [q["conf_val_abl"] for q in per_question]

    # Filter out any NaN entropy values
    valid_indices = [i for i, e in enumerate(entropies) if e == e]  # NaN check
    n_valid = len(valid_indices)
    n_invalid = len(per_question) - n_valid

    if n_valid < 3:
        raise ValueError(f"Too few valid samples for correlation: {n_valid}")

    entropies_valid = [entropies[i] for i in valid_indices]
    conf_base_valid = [conf_base[i] for i in valid_indices]
    conf_abl_valid = [conf_abl[i] for i in valid_indices]

    pearson_base, _ = pearsonr(conf_base_valid, entropies_valid)
    pearson_abl, _ = pearsonr(conf_abl_valid, entropies_valid)
    spearman_base, _ = spearmanr(conf_base_valid, entropies_valid)
    spearman_abl, _ = spearmanr(conf_abl_valid, entropies_valid)

    summary = {
        "pearson_baseline": round(pearson_base, 4),
        "pearson_ablated": round(pearson_abl, 4),
        "pearson_delta": round(pearson_abl - pearson_base, 4),
        "spearman_baseline": round(spearman_base, 4),
        "spearman_ablated": round(spearman_abl, 4),
        "spearman_delta": round(spearman_abl - spearman_base, 4),
        "n_valid": n_valid,
        "n_invalid": n_invalid,
    }

    return {"summary": summary, "per_question": per_question}


# --------- CLI ---------
def main():
    parser = argparse.ArgumentParser(
        description="Ablation: project out d_conf and test entropy-confidence correlation"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID_DEFAULT,
        help=f"HuggingFace model ID (default: {MODEL_ID_DEFAULT})",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=35,
        help="Layer index to apply ablation (default: 35)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="conf",
        help="Direction to ablate: conf, pass, so, random_0 (default: conf)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of questions to use (default: all clean 470)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template (use raw prompt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    use_chat_template = not args.no_chat_template
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ABLATION: ENTROPY-CONFIDENCE CORRELATION TEST")
    print("=" * 70)
    print(f"Model:          {args.model_id}")
    print(f"Layer:          {args.layer}")
    print(f"Direction:      {args.direction}")
    print(f"N questions:    {args.n or 'all (clean 470)'}")
    print(f"Seed:           {args.seed}")
    print(f"Chat template:  {use_chat_template}")
    print("=" * 70)

    # Load contaminated QIDs (for exclusion)
    contaminated_qids = load_contaminated_qids()
    print(f"Loaded {len(contaminated_qids)} contaminated QIDs to exclude.")

    # Load questions (clean set)
    questions = load_questions(args.n, args.seed, contaminated_qids)
    qids_used = [q["question_id"] for q in questions]

    # Load model
    tokenizer, model = load_model_and_tokenizer(args.model_id)

    # Load direction
    direction_path = get_direction_path(args.direction, args.layer)
    target_device = model.model.embed_tokens.weight.device
    direction_vec = load_direction(direction_path, target_device)

    # Run experiment
    results = run_experiment(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        layer_idx=args.layer,
        direction_vec=direction_vec,
        use_chat_template=use_chat_template,
    )

    # Build output
    output = {
        "config": {
            "model_id": args.model_id,
            "layer": args.layer,
            "direction": args.direction,
            "direction_path": str(direction_path),
            "use_chat_template": use_chat_template,
            "n_questions": len(questions),
            "n_contaminated_excluded": len(contaminated_qids),
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "qids_used": qids_used,
        "summary": results["summary"],
        "per_question": results["per_question"],
    }

    # Save
    out_path = output_dir / f"ablation_entropy_conf_corr_{args.direction}_layer{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    summary = results["summary"]
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Metric':<20} {'Baseline':>12} {'Ablated':>12} {'Delta':>12}")
    print("-" * 56)
    print(f"{'Pearson(conf,ent)':<20} {summary['pearson_baseline']:>12.4f} {summary['pearson_ablated']:>12.4f} {summary['pearson_delta']:>12.4f}")
    print(f"{'Spearman(conf,ent)':<20} {summary['spearman_baseline']:>12.4f} {summary['spearman_ablated']:>12.4f} {summary['spearman_delta']:>12.4f}")
    print(f"\nN valid: {summary['n_valid']}, N invalid: {summary['n_invalid']}")
    print("=" * 70)
    print(f"\nSaved to: {out_path}")

    # Quick interpretation
    print("\n--- INTERPRETATION ---")
    if summary["pearson_baseline"] < 0:
        print("✓ Baseline correlation is NEGATIVE (higher conf → lower entropy), as expected.")
    else:
        print("⚠ Baseline correlation is positive (unexpected).")

    if abs(summary["pearson_ablated"]) < abs(summary["pearson_baseline"]):
        print("✓ Ablation REDUCED the magnitude of correlation — d_conf is necessary for introspection.")
    else:
        print("⚠ Ablation did NOT reduce correlation — d_conf may not be necessary.")


if __name__ == "__main__":
    main()
