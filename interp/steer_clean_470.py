#!/usr/bin/env python3
"""
steer_clean_470.py

Sanity check: Re-run steering on 470 non-contaminated questions.
Also saves per-question decisions to check flip patterns.

This is a standalone script that doesn't modify steer_activations.py.

Usage:
  python interp/steer_clean_470.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from prompt_utils import build_pass_game_prompt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# --------- CONFIG ---------
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"

DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
INPUT_CSV = DATA_DIR / f"{MODEL_NAME}_unified.csv"
COMPILED_JSON = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"
INTRO_TRAIN = DATA_DIR / f"{MODEL_NAME}_introspective_extremes_AB_train.csv"

DIRECTION_DIR = Path("interp/outputs")
OUTPUT_DIR = Path("interp/outputs")

ALPHAS = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def get_contaminated_qids() -> set:
    """Get the 30 question IDs used to train d_conf."""
    train = pd.read_csv(INTRO_TRAIN)
    qids = set(train["A_qid"].tolist() + train["B_qid"].tolist())
    print(f"Found {len(qids)} contaminated question IDs")
    return qids


def load_direction(layer_idx: int) -> torch.Tensor:
    """Load confidence direction."""
    path = DIRECTION_DIR / f"confidence_direction_layer{layer_idx}.pt"
    data = torch.load(path, map_location="cuda", weights_only=False)
    if isinstance(data, dict):
        return data["direction"]
    return data


def load_model_and_tokenizer() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
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


def load_clean_questions(exclude_qids: set) -> list[dict]:
    """Load questions, excluding contaminated ones."""
    print(f"Loading questions from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    with open(COMPILED_JSON) as f:
        data = json.load(f)

    q_map = {}
    for qid, res in data["results"].items():
        q_data = res["question"] if isinstance(res["question"], dict) else res
        q_text = q_data.get("question")
        options = q_data.get("options")
        if q_text and options:
            q_map[qid] = {"question_text": q_text, "options": options}

    questions = []
    n_excluded = 0
    for _, row in df.iterrows():
        qid = row["question_id"]
        if qid in q_map:
            if qid in exclude_qids:
                n_excluded += 1
                continue
            questions.append(
                {
                    "question_id": qid,
                    "question_text": q_map[qid]["question_text"],
                    "options": q_map[qid]["options"],
                }
            )

    print(f"Loaded {len(questions)} clean questions (excluded {n_excluded})")
    return questions


def make_steering_hook(direction_vec: torch.Tensor, alpha: float):
    """Create forward hook for steering."""

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            v = direction_vec.to(hidden_states.device, hidden_states.dtype).view(1, 1, -1)
            hidden_states = hidden_states.clone()
            hidden_states[:, -1:, :] += alpha * v
            return (hidden_states,) + output[1:]
        else:
            out = output.clone()
            v = direction_vec.to(out.device, out.dtype).view(1, 1, -1)
            out[:, -1:, :] += alpha * v
            return out

    return hook


def run_with_steering(
    model, tokenizer, prompt: str, layer_idx: int, direction_vec: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Run model with steering and return logits."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    layer_module = model.model.layers[layer_idx]
    handle = layer_module.register_forward_hook(make_steering_hook(direction_vec, alpha))

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
    finally:
        handle.remove()

    return logits.squeeze(0)


def decode_pass_game(logits: torch.Tensor, tok_1: int, tok_2: int) -> str:
    """Decode pass game: '1' (Answer) or '2' (Pass)."""
    return "1" if logits[tok_1].item() >= logits[tok_2].item() else "2"


def main():
    print("=" * 60)
    print("CLEAN 470-QUESTION STEERING EXPERIMENT")
    print("=" * 60)

    # Get contaminated questions
    contaminated = get_contaminated_qids()

    # Load model
    tokenizer, model = load_model_and_tokenizer()

    # Load clean questions
    questions = load_clean_questions(contaminated)

    # Load direction
    layer_idx = 35
    direction = load_direction(layer_idx)
    print(f"Loaded d_conf for layer {layer_idx}, norm = {direction.norm().item():.4f}")

    # Token IDs
    tok_1 = tokenizer("1", add_special_tokens=False).input_ids[0]
    tok_2 = tokenizer("2", add_special_tokens=False).input_ids[0]

    # Run experiment
    results_by_alpha = {}
    per_question_all = {}  # {qid: {alpha: decision}}

    for alpha in ALPHAS:
        print(f"\n--- Alpha = {alpha} ---")
        decisions = []
        per_question = []

        for q in tqdm(questions, desc=f"α={alpha}"):
            prompt = build_pass_game_prompt(q["question_text"], q["options"])
            logits = run_with_steering(model, tokenizer, prompt, layer_idx, direction, alpha)
            decision = decode_pass_game(logits, tok_1, tok_2)
            decisions.append(decision)
            per_question.append({"qid": q["question_id"], "decision": decision})

            # Track per-question across alphas
            if q["question_id"] not in per_question_all:
                per_question_all[q["question_id"]] = {}
            per_question_all[q["question_id"]][alpha] = decision

        n_answer = sum(1 for d in decisions if d == "1")
        n_pass = len(decisions) - n_answer
        p_answer = n_answer / len(decisions)

        results_by_alpha[alpha] = {
            "alpha": alpha,
            "n_answer": n_answer,
            "n_pass": n_pass,
            "p_answer": round(p_answer, 4),
            "per_question": per_question,
        }
        print(f"  p_answer = {p_answer:.3f} ({n_answer}/{len(decisions)})")

    # Compute summary stats
    baseline_p = results_by_alpha[0.0]["p_answer"]
    for alpha, res in results_by_alpha.items():
        res["delta"] = round(res["p_answer"] - baseline_p, 4)

    # Analyze flip patterns
    print("\n" + "=" * 60)
    print("FLIP PATTERN ANALYSIS")
    print("=" * 60)

    monotonic_up = 0  # pass→answer as α increases (expected)
    monotonic_down = 0  # answer→pass as α increases (opposite)
    non_monotonic = 0  # mixed
    always_answer = 0
    always_pass = 0

    for qid, alpha_decisions in per_question_all.items():
        decisions_ordered = [alpha_decisions[a] for a in ALPHAS]

        if all(d == "1" for d in decisions_ordered):
            always_answer += 1
        elif all(d == "2" for d in decisions_ordered):
            always_pass += 1
        else:
            # Check monotonicity
            first_answer_idx = next((i for i, d in enumerate(decisions_ordered) if d == "1"), None)
            last_pass_idx = next(
                (i for i, d in enumerate(reversed(decisions_ordered)) if d == "2"), None
            )
            if last_pass_idx is not None:
                last_pass_idx = len(decisions_ordered) - 1 - last_pass_idx

            # Monotonic up: all passes before all answers
            is_monotonic_up = True
            seen_answer = False
            for d in decisions_ordered:
                if d == "1":
                    seen_answer = True
                elif seen_answer:  # pass after answer
                    is_monotonic_up = False
                    break

            # Monotonic down: all answers before all passes
            is_monotonic_down = True
            seen_pass = False
            for d in decisions_ordered:
                if d == "2":
                    seen_pass = True
                elif seen_pass:  # answer after pass
                    is_monotonic_down = False
                    break

            if is_monotonic_up and not is_monotonic_down:
                monotonic_up += 1
            elif is_monotonic_down and not is_monotonic_up:
                monotonic_down += 1
            else:
                non_monotonic += 1

    total = len(per_question_all)
    print(f"Always answer (all α):     {always_answer:4d} ({100*always_answer/total:.1f}%)")
    print(f"Always pass (all α):       {always_pass:4d} ({100*always_pass/total:.1f}%)")
    print(
        f"Monotonic pass→answer:     {monotonic_up:4d} ({100*monotonic_up/total:.1f}%) ← EXPECTED"
    )
    print(
        f"Monotonic answer→pass:     {monotonic_down:4d} ({100*monotonic_down/total:.1f}%) ← OPPOSITE"
    )
    print(f"Non-monotonic:             {non_monotonic:4d} ({100*non_monotonic/total:.1f}%)")

    # Save results
    output = {
        "config": {
            "task": "pass_game",
            "direction": "conf",
            "layer": layer_idx,
            "n_questions": len(questions),
            "n_excluded": len(contaminated),
            "alphas": ALPHAS,
            "timestamp": datetime.now().isoformat(),
        },
        "results": [results_by_alpha[a] for a in ALPHAS],
        "flip_analysis": {
            "always_answer": always_answer,
            "always_pass": always_pass,
            "monotonic_up": monotonic_up,
            "monotonic_down": monotonic_down,
            "non_monotonic": non_monotonic,
        },
        "per_question_decisions": per_question_all,
    }

    output_file = OUTPUT_DIR / "steering_pass_game_CLEAN_470.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON: CLEAN 470 vs ORIGINAL 500")
    print("=" * 60)
    print(f"{'α':>6} | {'Clean 470':>10} | (Original 500 for reference)")
    print("-" * 40)
    for alpha in ALPHAS:
        p = results_by_alpha[alpha]["p_answer"]
        print(f"{alpha:>6.1f} | {p:>10.1%} |")

    swing = results_by_alpha[3.0]["p_answer"] - results_by_alpha[-3.0]["p_answer"]
    print("-" * 40)
    print(f"Swing (α=-3 to α=+3): {swing:.1%} ({swing*100:.1f}pp)")


if __name__ == "__main__":
    main()
