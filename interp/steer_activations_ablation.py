#!/usr/bin/env python3
"""
steer_activations_ablation.py

Ablation experiment: project out the d_conf direction at layer 35 and measure
how that changes behavior on pass_game, simplemc_self, simplemc_other.

This tests whether d_conf is NECESSARY for the behavior, not just sufficient.

Usage:
  python interp/steer_activations_ablation.py --task pass_game --layer 35 --n 500
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

from prompt_utils import build_pass_game_prompt, build_other_prompt, build_self_prompt

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
def load_questions(n: Optional[int] = None) -> List[Dict]:
    """Load questions from unified CSV and compiled JSON."""
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

    if n is not None and len(questions) > n:
        questions = questions[:n]

    print(f"Loaded {len(questions)} questions.")
    return questions


# --------- Direction Loading ---------
def load_d_conf(layer_idx: int, device: torch.device) -> torch.Tensor:
    """Load the d_conf direction for a given layer."""
    path = DIRECTION_DIR / f"confidence_direction_layer{layer_idx}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Direction file not found: {path}")

    data = torch.load(path, map_location=device, weights_only=False)

    if isinstance(data, torch.Tensor):
        v = data
    elif isinstance(data, dict) and "direction" in data:
        v = data["direction"]
    else:
        raise ValueError(f"Unexpected format in {path}: {type(data)}")

    v = v.view(-1)
    return v.to(device)


# --------- Ablation Hook ---------
def make_ablation_hook(direction_vec: torch.Tensor):
    """
    Create a forward hook that projects out the component of the hidden state
    along direction_vec for the last token at a given layer.

    h_last' = h_last - proj_{d_conf}(h_last)
    """
    v = direction_vec / direction_vec.norm()

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            hs = hidden_states
        else:
            hs = output

        # hs: (batch, seq, dim)
        v_unit = v.to(hs.device, hs.dtype).view(1, 1, -1)
        last = hs[:, -1:, :]  # last token only

        coeff = (last * v_unit).sum(dim=-1, keepdim=True)  # (batch, 1, 1)
        proj = coeff * v_unit  # (batch, 1, dim)

        hs_new = hs.clone()
        hs_new[:, -1:, :] = last - proj

        if isinstance(output, tuple):
            return (hs_new,) + output[1:]
        else:
            return hs_new

    return hook


# --------- Forward Pass with Ablation ---------
def run_with_ablation(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    direction_vec: torch.Tensor,
    use_chat_template: bool,
) -> torch.Tensor:
    """Run model with ablation hook at layer_idx and return logits for last token."""
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    layer_module = model.model.layers[layer_idx]
    handle = layer_module.register_forward_hook(make_ablation_hook(direction_vec))

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
        "p_answer": round(n_answer / total, 4) if total > 0 else 0.0,
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


# --------- Main Experiment ---------
def run_ablation_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    task: str,
    layer_idx: int,
    direction_vec: torch.Tensor,
    use_chat_template: bool,
) -> Dict:
    """Run ablation experiment for one task."""
    token_ids = get_token_ids(tokenizer)

    if task == "pass_game":
        decisions = []
        for q in tqdm(questions, desc=f"pass_game ablation L{layer_idx}"):
            prompt = build_pass_game_prompt(q["question_text"], q["options"])
            logits = run_with_ablation(
                model, tokenizer, prompt, layer_idx, direction_vec, use_chat_template
            )
            decision = decode_pass_game(logits, token_ids)
            decisions.append(decision)

        metrics = compute_pass_game_metrics(decisions)

    elif task in ["simplemc_self", "simplemc_other"]:
        letters = []
        prompt_fn = build_self_prompt if task == "simplemc_self" else build_other_prompt

        for q in tqdm(questions, desc=f"{task} ablation L{layer_idx}"):
            prompt = prompt_fn(q["question_text"])
            logits = run_with_ablation(
                model, tokenizer, prompt, layer_idx, direction_vec, use_chat_template
            )
            letter = decode_confidence(logits, token_ids)
            letters.append(letter)

        metrics = compute_confidence_metrics(letters)

    else:
        raise ValueError(f"Unknown task: {task}")

    return metrics


# --------- CLI ---------
def main():
    parser = argparse.ArgumentParser(description="d_conf ablation experiment")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["pass_game", "simplemc_self", "simplemc_other"],
        help="Task to run",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=35,
        help="Layer index to apply ablation (default: 35)",
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
    args = parser.parse_args()

    use_chat_template = not args.no_chat_template
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("D_CONF ABLATION EXPERIMENT")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Layer: {args.layer}")
    print(f"N questions: {args.n or 'all'}")
    print(f"Model: {args.model_id}")
    print(f"Chat template: {use_chat_template}")
    print("=" * 60)

    # Load model and data
    tokenizer, model = load_model_and_tokenizer(args.model_id)
    questions = load_questions(args.n)

    # Load d_conf direction
    d_conf_vec = load_d_conf(args.layer, model.device)
    print(f"Loaded d_conf direction (norm: {d_conf_vec.norm():.4f})")

    # Run ablation
    metrics = run_ablation_experiment(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        task=args.task,
        layer_idx=args.layer,
        direction_vec=d_conf_vec,
        use_chat_template=use_chat_template,
    )

    # Build output
    result = {
        "config": {
            "task": args.task,
            "layer": args.layer,
            "direction": "d_conf",
            "mode": "ablation",
            "model_id": args.model_id,
            "n_questions": len(questions),
            "use_chat_template": use_chat_template,
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": metrics,
    }

    # Save
    out_path = output_dir / f"ablation_{args.task}_d_conf_layer{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.task}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if k != "counts":
            print(f"  {k}: {v}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
