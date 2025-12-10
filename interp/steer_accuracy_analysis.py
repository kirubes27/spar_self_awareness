#!/usr/bin/env python3
"""
steer_accuracy_analysis.py

Run pass_game steering and compute accuracy among answered questions.
Key question: Does steering toward +α make the model overconfident (answer more but get more wrong)?

Usage:
  python interp/steer_accuracy_analysis.py --layer 35 --n 500
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt_utils import build_pass_game_prompt

# --------- CONFIG ---------
MODEL_ID_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"

DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
INPUT_CSV = DATA_DIR / f"{MODEL_NAME}_unified.csv"
COMPILED_JSON = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"

DIRECTION_DIR = Path("interp/outputs")
OUTPUT_DIR = Path("interp/outputs")

DEFAULT_ALPHAS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]


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
def load_questions_with_correctness(n: Optional[int] = None) -> List[Dict]:
    """Load questions with baseline correctness labels."""
    print(f"Loading questions from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    print(f"Loading compiled JSON from {COMPILED_JSON}...")
    with open(COMPILED_JSON) as f:
        compiled = json.load(f)

    questions = []
    for _, row in df.iterrows():
        qid = row["question_id"]
        if qid in compiled["results"]:
            res = compiled["results"][qid]
            q_data = res.get("question", {})

            # Use is_correct field directly from compiled results
            is_correct = res.get("is_correct", False)

            questions.append({
                "question_id": qid,
                "question_text": q_data.get("question", ""),
                "options": q_data.get("options", {}),
                "baseline_correct": is_correct,
            })

    if n is not None and len(questions) > n:
        questions = questions[:n]

    n_correct = sum(q["baseline_correct"] for q in questions)
    print(f"Loaded {len(questions)} questions.")
    print(f"Baseline accuracy: {n_correct}/{len(questions)} = {n_correct/len(questions):.1%}")
    return questions


# --------- Direction Loading ---------
def load_direction(layer_idx: int, device: torch.device) -> torch.Tensor:
    """Load d_conf direction."""
    path = DIRECTION_DIR / f"confidence_direction_layer{layer_idx}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Direction file not found: {path}")

    data = torch.load(path, map_location=device, weights_only=False)

    if isinstance(data, torch.Tensor):
        v = data
    elif isinstance(data, dict) and "direction" in data:
        v = data["direction"]
    else:
        raise ValueError(f"Unexpected format in {path}")

    return v.view(-1).to(device)


# --------- Steering Hook ---------
def make_steering_hook(direction_vec: torch.Tensor, alpha: float):
    """Create steering hook: h' = h + α * d at last token."""

    def hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output

        v = direction_vec.to(hs.device, hs.dtype).view(1, 1, -1)
        hs_new = hs.clone()
        hs_new[:, -1:, :] = hs[:, -1:, :] + alpha * v

        if isinstance(output, tuple):
            return (hs_new,) + output[1:]
        return hs_new

    return hook


# --------- Token IDs (copied from steer_activations.py) ---------
def get_token_ids(tokenizer) -> Dict:
    """Precompute token IDs for decoding."""
    return {
        "1": tokenizer("1", add_special_tokens=False).input_ids[0],
        "2": tokenizer("2", add_special_tokens=False).input_ids[0],
    }


# --------- Forward Pass with Steering ---------
def run_with_steering(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    direction_vec: torch.Tensor,
    alpha: float,
    token_ids: Dict,
) -> str:
    """Run pass_game with steering and return decision '1' or '2'."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    layer_module = model.model.layers[layer_idx]
    handle = layer_module.register_forward_hook(make_steering_hook(direction_vec, alpha))

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
    finally:
        handle.remove()

    # Decode decision using precomputed token IDs
    p1 = logits[0, token_ids["1"]].item()
    p2 = logits[0, token_ids["2"]].item()

    return "1" if p1 >= p2 else "2"


# --------- Main Experiment ---------
def run_accuracy_analysis(
    model,
    tokenizer,
    questions: List[Dict],
    layer_idx: int,
    direction_vec: torch.Tensor,
    alphas: List[float],
) -> List[Dict]:
    """Run steering at each α and compute accuracy among answered questions."""

    token_ids = get_token_ids(tokenizer)
    results = []

    for alpha in alphas:
        print(f"\n--- Alpha = {alpha} ---")

        decisions = []
        for q in tqdm(questions, desc=f"α={alpha}"):
            prompt = build_pass_game_prompt(q["question_text"], q["options"])
            decision = run_with_steering(
                model, tokenizer, prompt, layer_idx, direction_vec, alpha, token_ids
            )
            decisions.append({
                "question_id": q["question_id"],
                "decision": decision,
                "baseline_correct": q["baseline_correct"],
            })

        # Compute metrics for answered questions
        answered = [d for d in decisions if d["decision"] == "1"]
        n_answer = len(answered)
        n_pass = len(decisions) - n_answer
        n_correct_answered = sum(1 for d in answered if d["baseline_correct"])

        accuracy = n_correct_answered / n_answer if n_answer > 0 else 0.0
        coverage = n_answer / len(decisions)

        # Compute metrics for passed questions (are they actually harder?)
        passed = [d for d in decisions if d["decision"] == "2"]
        n_correct_passed = sum(1 for d in passed if d["baseline_correct"])
        pass_accuracy = n_correct_passed / len(passed) if passed else 0.0

        results.append({
            "alpha": alpha,
            "n_answer": n_answer,
            "n_pass": n_pass,
            "p_answer": round(coverage, 4),
            "n_correct_answered": n_correct_answered,
            "accuracy_given_answer": round(accuracy, 4),
            "n_correct_passed": n_correct_passed,
            "accuracy_given_pass": round(pass_accuracy, 4),
            "decisions": decisions,  # Full per-question data
        })

        print(f"  Coverage: {n_answer}/{len(decisions)} ({coverage:.1%})")
        print(f"  Accuracy (answered): {n_correct_answered}/{n_answer} = {accuracy:.1%}")
        print(f"  Accuracy (passed): {n_correct_passed}/{n_pass} = {pass_accuracy:.1%}")

    return results


# --------- CLI ---------
def main():
    parser = argparse.ArgumentParser(description="Accuracy vs steering analysis")
    parser.add_argument(
        "--layer",
        type=int,
        default=35,
        help="Layer index to apply steering (default: 35)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of questions to use (default: all)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help=f"Alpha values (default: {DEFAULT_ALPHAS})",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID_DEFAULT,
        help=f"HuggingFace model ID (default: {MODEL_ID_DEFAULT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACCURACY VS STEERING ANALYSIS")
    print("=" * 60)
    print(f"Layer: {args.layer}")
    print(f"N questions: {args.n or 'all'}")
    print(f"Alphas: {args.alphas}")
    print(f"Model: {args.model_id}")
    print("=" * 60)

    tokenizer, model = load_model_and_tokenizer(args.model_id)
    questions = load_questions_with_correctness(args.n)
    direction_vec = load_direction(args.layer, model.device)
    print(f"Loaded d_conf direction (norm: {direction_vec.norm():.4f})")

    results = run_accuracy_analysis(
        model, tokenizer, questions, args.layer, direction_vec, args.alphas
    )

    # Build output (without per-question data for compact file)
    results_compact = []
    for r in results:
        results_compact.append({
            "alpha": r["alpha"],
            "n_answer": r["n_answer"],
            "n_pass": r["n_pass"],
            "p_answer": r["p_answer"],
            "n_correct_answered": r["n_correct_answered"],
            "accuracy_given_answer": r["accuracy_given_answer"],
            "n_correct_passed": r["n_correct_passed"],
            "accuracy_given_pass": r["accuracy_given_pass"],
        })

    output = {
        "config": {
            "layer": args.layer,
            "alphas": args.alphas,
            "n_questions": len(questions),
            "model_id": args.model_id,
            "timestamp": datetime.now().isoformat(),
        },
        "baseline_accuracy": sum(q["baseline_correct"] for q in questions) / len(questions),
        "results": results_compact,
    }

    # Save compact results
    out_path = output_dir / f"accuracy_vs_alpha_layer{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Save full per-question results separately
    full_path = output_dir / f"accuracy_vs_alpha_layer{args.layer}_full.json"
    full_output = {"config": output["config"], "results": results}
    with open(full_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved full per-question data to {full_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Accuracy vs Coverage")
    print("=" * 80)
    print(f"{'α':>6} | {'Coverage':>10} | {'Acc(Answer)':>12} | {'Acc(Pass)':>12} | {'Answered':>8}")
    print("-" * 70)
    for r in results_compact:
        print(
            f"{r['alpha']:>6.1f} | {r['p_answer']:>10.1%} | "
            f"{r['accuracy_given_answer']:>12.1%} | "
            f"{r['accuracy_given_pass']:>12.1%} | {r['n_answer']:>8}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
