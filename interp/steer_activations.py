#!/usr/bin/env python3
"""
steer_activations.py

Causal activation steering experiment: inject ±α * direction_vec into the
residual stream and measure behavioral changes.

Supports:
  - Directions: d_conf, d_pass, d_so
  - Tasks: pass_game, simplemc_self, simplemc_other

Usage:
  python interp/steer_activations.py \
    --task pass_game \
    --directions conf \
    --layers 35 50 \
    --alphas -2 -1 -0.5 0 0.5 1 2 \
    --use-chat-template
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt_utils import build_pass_game_prompt, build_self_prompt, build_other_prompt

# --------- CONFIG ---------
MODEL_ID_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"

DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
INPUT_CSV = DATA_DIR / f"{MODEL_NAME}_unified.csv"
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

DEFAULT_ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


# --------- Direction Loading ---------
def get_direction_path(direction_name: str, layer_idx: int) -> Path:
    """Map CLI direction name to file path."""
    # Handle random directions (e.g., random_0, random_5)
    if direction_name.startswith("random_"):
        idx = direction_name.split("_")[1]
        return DIRECTION_DIR / f"random_direction_{idx}_layer35.pt"
    elif direction_name == "conf":
        return DIRECTION_DIR / f"confidence_direction_layer{layer_idx}.pt"
    elif direction_name == "conf_bad":
        return DIRECTION_DIR / f"d_conf_bad_layer{layer_idx}.pt"
    elif direction_name == "pass":
        return DIRECTION_DIR / f"pass_game_direction_layer{layer_idx}.pt"
    elif direction_name == "so":
        return DIRECTION_DIR / f"self_other_direction_layer{layer_idx}.pt"
    else:
        raise ValueError(f"Unknown direction: {direction_name}")


def load_direction(path: Path, device: torch.device) -> torch.Tensor:
    """Load direction vector from .pt file."""
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

    return v.to(device)


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


# --------- Data Loading ---------
def load_questions(
    n: Optional[int] = None,
    only_qids: Optional[List[str]] = None,
    exclude_qids: Optional[List[str]] = None,
) -> List[Dict]:
    """Load questions from unified CSV and compiled JSON.

    Args:
        n: Maximum number of questions to load.
        only_qids: If provided, only include questions with these QIDs.
        exclude_qids: If provided, exclude questions with these QIDs.
    """
    print(f"Loading questions from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    print(f"Loading compiled JSON from {COMPILED_JSON}...")
    with open(COMPILED_JSON) as f:
        data = json.load(f)

    # Build question map from compiled results
    q_map = {}
    for qid, res in data["results"].items():
        q_data = res["question"] if isinstance(res["question"], dict) else res
        q_text = q_data.get("question")
        options = q_data.get("options")
        if q_text and options:
            q_map[qid] = {"question_text": q_text, "options": options}

    # Filter to questions in both CSV and JSON
    questions = []
    for _, row in df.iterrows():
        qid = row["question_id"]
        if qid in q_map:
            questions.append(
                {
                    "question_id": qid,
                    "question_text": q_map[qid]["question_text"],
                    "options": q_map[qid]["options"],
                }
            )

    initial_count = len(questions)

    # Apply QID filters
    if only_qids is not None:
        only_qids_set = set(only_qids)
        questions = [q for q in questions if q["question_id"] in only_qids_set]
        print(f"  After --only-qids filter: {len(questions)} (was {initial_count})")

    if exclude_qids is not None:
        exclude_qids_set = set(exclude_qids)
        before_exclude = len(questions)
        questions = [q for q in questions if q["question_id"] not in exclude_qids_set]
        print(f"  After --exclude-qids filter: {len(questions)} (was {before_exclude})")

    if n is not None and len(questions) > n:
        questions = questions[:n]

    print(f"Loaded {len(questions)} questions.")
    return questions


# --------- Steering Hook ---------
def make_steering_hook(direction_vec: torch.Tensor, alpha: float):
    """
    Create a forward hook that injects alpha * direction_vec at the last token.
    Handles both tuple and tensor outputs from HuggingFace models.
    """

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Cast to same device/dtype as hidden states
            v = direction_vec.to(hidden_states.device, hidden_states.dtype).view(1, 1, -1)
            hidden_states = hidden_states.clone()
            hidden_states[:, -1:, :] += alpha * v
            return (hidden_states,) + output[1:]
        else:
            out = output
            v = direction_vec.to(out.device, out.dtype).view(1, 1, -1)
            out = out.clone()
            out[:, -1:, :] += alpha * v
            return out

    return hook


# --------- Forward Pass with Steering ---------
def run_with_steering(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    direction_vec: torch.Tensor,
    alpha: float,
    use_chat_template: bool,
) -> torch.Tensor:
    """
    Run model with steering hook and return logits for last token.
    """
    # Format prompt
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Attach hook
    layer_module = model.model.layers[layer_idx]
    handle = layer_module.register_forward_hook(make_steering_hook(direction_vec, alpha))

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
    finally:
        handle.remove()

    return logits.squeeze(0)  # (vocab_size,)


# --------- Token IDs ---------
def get_token_ids(tokenizer) -> Dict:
    """Precompute token IDs for decoding."""
    return {
        "1": tokenizer("1", add_special_tokens=False).input_ids[0],
        "2": tokenizer("2", add_special_tokens=False).input_ids[0],
        "letters": {
            letter: tokenizer(letter, add_special_tokens=False).input_ids[0]
            for letter in "ABCDEFGH"
        },
    }


# --------- Decoding ---------
def decode_pass_game(logits: torch.Tensor, token_ids: Dict) -> str:
    """Decode pass game decision: '1' (Answer) or '2' (Pass)."""
    p1 = logits[token_ids["1"]].item()
    p2 = logits[token_ids["2"]].item()
    return "1" if p1 >= p2 else "2"


def decode_confidence(logits: torch.Tensor, token_ids: Dict) -> str:
    """Decode confidence letter A-H."""
    letter_logits = torch.tensor(
        [logits[token_ids["letters"][letter]].item() for letter in "ABCDEFGH"]
    )
    pred_idx = letter_logits.argmax().item()
    return "ABCDEFGH"[pred_idx]


# --------- Metrics Computation ---------
def compute_pass_game_metrics(decisions: List[str]) -> Dict:
    """Compute pass game metrics from list of decisions."""
    n_answer = sum(1 for d in decisions if d == "1")
    n_pass = len(decisions) - n_answer
    total = len(decisions)
    return {
        "n_answer": n_answer,
        "n_pass": n_pass,
        "p_answer": n_answer / total if total > 0 else 0.0,
    }


def compute_confidence_metrics(letters: List[str]) -> Dict:
    """Compute confidence metrics from list of letter predictions."""
    counts = {letter: 0 for letter in "ABCDEFGH"}
    for letter in letters:
        counts[letter] += 1

    total = len(letters)
    mean_conf = (
        sum(CONF_MIDPOINTS[letter] * counts[letter] for letter in counts) / total
        if total > 0
        else 0.0
    )
    p_high = (counts["G"] + counts["H"]) / total if total > 0 else 0.0

    return {
        "counts": counts,
        "mean_conf": round(mean_conf, 4),
        "p_high": round(p_high, 4),
    }


# --------- Main Experiment Loop ---------
def run_steering_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    task: str,
    direction_name: str,
    layer_idx: int,
    alphas: List[float],
    use_chat_template: bool,
    output_dir: Path,
    model_id: str,
    direction_path_override: Optional[Path] = None,
) -> Dict:
    """Run steering experiment for one (task, direction, layer) combination."""

    # Load direction
    if direction_path_override is not None:
        direction_path = direction_path_override
    else:
        direction_path = get_direction_path(direction_name, layer_idx)
    direction_vec = load_direction(direction_path, model.device)
    print(f"Loaded direction from {direction_path} (shape: {direction_vec.shape})")

    # Get token IDs
    token_ids = get_token_ids(tokenizer)

    # Results storage
    results_by_alpha = {}

    for alpha in alphas:
        print(f"\n--- Alpha = {alpha} ---")

        if task == "pass_game":
            decisions = []
            for q in tqdm(questions, desc=f"pass_game α={alpha}"):
                prompt = build_pass_game_prompt(q["question_text"], q["options"])
                logits = run_with_steering(
                    model, tokenizer, prompt, layer_idx, direction_vec, alpha, use_chat_template
                )
                decision = decode_pass_game(logits, token_ids)
                decisions.append(decision)

            metrics = compute_pass_game_metrics(decisions)

        elif task in ["simplemc_self", "simplemc_other"]:
            letters = []
            prompt_fn = build_self_prompt if task == "simplemc_self" else build_other_prompt

            for q in tqdm(questions, desc=f"{task} α={alpha}"):
                prompt = prompt_fn(q["question_text"])
                logits = run_with_steering(
                    model, tokenizer, prompt, layer_idx, direction_vec, alpha, use_chat_template
                )
                letter = decode_confidence(logits, token_ids)
                letters.append(letter)

            metrics = compute_confidence_metrics(letters)

        else:
            raise ValueError(f"Unknown task: {task}")

        results_by_alpha[alpha] = {"alpha": alpha, **metrics}

        # Print summary
        if task == "pass_game":
            print(
                f"  n_answer={metrics['n_answer']}, n_pass={metrics['n_pass']}, p_answer={metrics['p_answer']:.3f}"
            )
        else:
            print(f"  mean_conf={metrics['mean_conf']:.3f}, p_high={metrics['p_high']:.3f}")

    # Build output structure
    baseline = results_by_alpha.get(0.0, results_by_alpha[alphas[0]])

    # Add delta_p_answer for pass_game
    if task == "pass_game":
        baseline_p = baseline["p_answer"]
        for alpha, res in results_by_alpha.items():
            res["delta_p_answer"] = round(res["p_answer"] - baseline_p, 4)

    output = {
        "config": {
            "task": task,
            "direction": direction_name,
            "layer": layer_idx,
            "alphas": alphas,
            "model_id": model_id,
            "use_chat_template": use_chat_template,
            "n_questions": len(questions),
            "timestamp": datetime.now().isoformat(),
        },
        "baseline": baseline,
        "results": [results_by_alpha[alpha] for alpha in alphas],
    }

    # Save JSON
    output_file = output_dir / f"steering_{task}_d_{direction_name}_layer{layer_idx}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_file}")

    return output


# --------- CLI ---------
def main():
    parser = argparse.ArgumentParser(description="Causal activation steering experiment")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["pass_game", "simplemc_self", "simplemc_other"],
        help="Task to run: pass_game, simplemc_self, or simplemc_other",
    )
    parser.add_argument(
        "--directions",
        type=str,
        nargs="+",
        default=["conf"],
        help="Directions to steer with (conf, pass, so, or random_0, random_1, etc.)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[35, 50],
        help="Layer indices to steer (default: 35 50)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help=f"Alpha values (default: {DEFAULT_ALPHAS})",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of questions to use (default: all)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID_DEFAULT,
        help=f"HuggingFace model ID (default: {MODEL_ID_DEFAULT})",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        default=True,
        help="Use chat template (default: True)",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--direction-path",
        type=str,
        default=None,
        help="Path to direction .pt file. If provided, overrides --directions.",
    )
    parser.add_argument(
        "--only-qids",
        type=str,
        default=None,
        help="Path to JSON file with list of QIDs to include (filter to only these).",
    )
    parser.add_argument(
        "--exclude-qids",
        type=str,
        default=None,
        help="Path to JSON file with list of QIDs to exclude.",
    )
    args = parser.parse_args()

    # Handle chat template flag
    use_chat_template = not args.no_chat_template

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load QID filter lists if provided
    only_qids = None
    if args.only_qids:
        with open(args.only_qids) as f:
            only_qids = json.load(f)
        print(f"Loaded {len(only_qids)} QIDs from --only-qids: {args.only_qids}")

    exclude_qids = None
    if args.exclude_qids:
        with open(args.exclude_qids) as f:
            exclude_qids = json.load(f)
        print(f"Loaded {len(exclude_qids)} QIDs from --exclude-qids: {args.exclude_qids}")

    # Determine directions to run
    if args.direction_path:
        # Use custom path, create a placeholder direction name
        directions_to_run = ["custom"]
        direction_path_override = Path(args.direction_path)
        print(f"Using custom direction path: {direction_path_override}")
    else:
        directions_to_run = args.directions
        direction_path_override = None

    print("=" * 60)
    print("ACTIVATION STEERING EXPERIMENT")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Directions: {directions_to_run}")
    print(f"Layers: {args.layers}")
    print(f"Alphas: {args.alphas}")
    print(f"N questions: {args.n or 'all'}")
    print(f"Only QIDs: {args.only_qids or 'none'}")
    print(f"Exclude QIDs: {args.exclude_qids or 'none'}")
    print(f"Model: {args.model_id}")
    print(f"Chat template: {use_chat_template}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    # Load model and data
    tokenizer, model = load_model_and_tokenizer(args.model_id)
    questions = load_questions(args.n, only_qids=only_qids, exclude_qids=exclude_qids)

    # Run experiments
    for direction in directions_to_run:
        for layer in args.layers:
            print(f"\n{'='*60}")
            print(f"Running: task={args.task}, direction={direction}, layer={layer}")
            print(f"{'='*60}")

            try:
                run_steering_experiment(
                    model=model,
                    tokenizer=tokenizer,
                    questions=questions,
                    task=args.task,
                    direction_name=direction,
                    layer_idx=layer,
                    alphas=args.alphas,
                    use_chat_template=use_chat_template,
                    output_dir=output_dir,
                    model_id=args.model_id,
                    direction_path_override=direction_path_override,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
