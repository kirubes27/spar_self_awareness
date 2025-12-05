#!/usr/bin/env python3
"""
Compare Pass Game Direction vs Self-Other AND Confidence Directions.

Features:
  - Multi-layer analysis in one pass (default: 35, 50, 79).
  - Uses all matched questions by default (robustness).
  - Evaluates both d_so (Self-Other) and d_conf (Confidence).
  - Computes AUC and Cosine Similarity for clear signal detection.

Run e.g.:
  python interp/compare_pass_game_direction.py --layers 35 50 79 --n-splits 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --------- CONFIG (adjust paths if needed) ---------
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"

DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
INPUT_CSV = DATA_DIR / f"{MODEL_NAME}_unified.csv"

COMPILED_JSON = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"

OUTPUT_DIR = Path("interp/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------- Utilities ---------
def build_pass_game_prompt(question_text: str, options_dict: dict) -> str:
    """Builds the simplified pass-game prompt used for the decision-only test."""
    options_text = ""
    for k, v in options_dict.items():
        options_text += f"{k}: {v}\n"

    prompt = f"""Consider the answer to the following question:
{question_text}
{options_text}
Consider your level of certainty in your answer. If you are at least 10% certain that you know the correct answer, output '1'; if not, output '2'.
Respond ONLY with '1' or '2'.
Your choice ('1' or '2'): """
    return prompt


def load_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load 4-bit quantized Llama-3.3-70B-Instruct."""
    print(f"Loading tokenizer for {MODEL_ID}...")
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


def get_all_layers_hidden_states_and_choice(
    model, tokenizer, prompt: str
) -> Tuple[torch.Tensor, str]:
    """
    Run the model on a single pass-game prompt and return:
      - all_hs: hidden states for all layers at last token shape (num_layers, d)
      - choice: "1" or "2" based on next-token argmax prob
    """
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (batch, seq, dim). 0=embeds, 1..N=layers.
        # We want layers 0..79 (which are indices 1..80 in the tuple).
        # Stack them: (num_layers, batch, seq, dim)
        # We take the last token directly.

        # Taking outputs.hidden_states[1:] gives layers 0..79
        layer_hs = [hs[:, -1, :] for hs in outputs.hidden_states[1:]]
        # Stack -> (num_layers, 1, d)
        all_hs = torch.stack(layer_hs, dim=0).squeeze(1) # (num_layers, d)

        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        id_1 = tokenizer.encode("1", add_special_tokens=False)[0]
        id_2 = tokenizer.encode("2", add_special_tokens=False)[0]

        prob_1 = probs[0, id_1].item()
        prob_2 = probs[0, id_2].item()

        choice = "1" if prob_1 > prob_2 else "2"

    return all_hs, choice


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC AUC for binary labels (1=Answer, 0=Pass)."""
    assert scores.shape[0] == labels.shape[0]
    import warnings
    if len(np.unique(labels)) < 2:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores)) + 1

    pos_mask = labels == 1
    n_pos = pos_mask.sum()
    n_neg = (labels == 0).sum()

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_pos_ranks = ranks[pos_mask].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def analyze_layer(
    layer_idx: int,
    H_L: torch.Tensor,     # (N, d)
    labels: np.ndarray,    # (N,) 1=Answer, 0=Pass
    n_splits: int,
    seed: int,
    device: torch.device
) -> Dict:
    """Evaluate d_pass, d_so, and d_conf for a specific layer."""

    # Masks
    ans_mask = labels == 1
    pass_mask = labels == 0
    n_ans = ans_mask.sum()
    n_pass = pass_mask.sum()

    stats = {
        "layer": layer_idx,
        "n_total": int(len(labels)),
        "n_answer": int(n_ans),
        "n_pass": int(n_pass)
    }

    # 1. Learn d_pass (Full Data)
    if n_ans == 0 or n_pass == 0:
        d_pass_full = None
    else:
        mean_ans = H_L[ans_mask].mean(dim=0)
        mean_pass = H_L[pass_mask].mean(dim=0)
        d_pass_full = mean_ans - mean_pass

    # 2. Split-Half CV for d_pass
    rng = np.random.default_rng(seed)
    n = len(labels)
    indices = np.arange(n)

    split_metrics = []
    if n_ans > 1 and n_pass > 1:
        for _ in range(n_splits):
            rng.shuffle(indices)
            mid = n // 2
            train_idx = indices[:mid]
            test_idx = indices[mid:]

            # Check balance
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]
            if len(np.unique(train_labels)) < 2 or len(np.unique(test_labels)) < 2:
                continue

            H_train = H_L[train_idx]
            H_test = H_L[test_idx]

            # Train d_pass
            m_ans = H_train[train_labels==1].mean(dim=0)
            m_pass = H_train[train_labels==0].mean(dim=0)
            d_p = m_ans - m_pass

            # Evaluate
            with torch.no_grad():
                scores_train = (H_train @ d_p).cpu().numpy()
                scores_test = (H_test @ d_p).cpu().numpy()

            split_metrics.append({
                "train_auc": compute_auc(scores_train, train_labels),
                "test_auc": compute_auc(scores_test, test_labels)
            })

    # Aggregates
    def get_stats(key):
        vals = [m[key] for m in split_metrics]
        if not vals: return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    stats["d_pass"] = {
        "split_train_auc_mean": get_stats("train_auc")[0],
        "split_train_auc_std": get_stats("train_auc")[1],
        "split_test_auc_mean": get_stats("test_auc")[0],
        "split_test_auc_std": get_stats("test_auc")[1],
    }

    # Helper for external directions
    def eval_external_direction(name: str, filename: str):
        path = OUTPUT_DIR / filename
        if not path.exists():
            stats[name] = {"available": False}
            return

        try:
            data = torch.load(path, map_location=device)
            d_ext = data["direction"].to(device)

            # Project
            with torch.no_grad():
                scores = (H_L @ d_ext).cpu().numpy()

            # Orientation check: Mean score of Answer should be > Mean score of Pass
            if n_ans > 0 and n_pass > 0:
                m_a = scores[ans_mask].mean()
                m_p = scores[pass_mask].mean()
                flipped = False
                if m_a < m_p:
                    scores = -scores
                    d_ext = -d_ext
                    flipped = True

                auc = compute_auc(scores, labels)

                # Cosine with d_pass_full
                if d_pass_full is not None:
                    cos = F.cosine_similarity(d_pass_full.unsqueeze(0), d_ext.unsqueeze(0)).item()
                else:
                    cos = float("nan")
            else:
                auc = float("nan")
                cos = float("nan")
                flipped = False

            stats[name] = {
                "available": True,
                "auc_ans_vs_pass": float(auc),
                "cos_d_pass_full": float(cos),
                "flipped": flipped
            }
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            stats[name] = {"available": False, "error": str(e)}

    # 3. Evaluate d_so
    eval_external_direction("d_so", f"self_other_direction_layer{layer_idx}.pt")

    # 4. Evaluate d_conf
    eval_external_direction("d_conf", f"confidence_direction_layer{layer_idx}.pt")

    return stats


# --------- Main ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="If set, number of questions to sample; if None, use all matched questions."
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=[35, 50, 79],
        help="Layer indices (0-based) to analyze. Default: 35, 50, 79.",
    )
    parser.add_argument("--n-splits", type=int, default=10, help="Split-half repetitions")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    print(f"Analyzing layers: {args.layers}")

    # 1. Load Data
    print(f"Loading CSV from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    print(f"Loading compiled JSON from {COMPILED_JSON}...")
    with open(COMPILED_JSON) as f:
        data = json.load(f)

    # Join by qid
    q_map = {}
    for qid, res in data["results"].items():
        q_data = res["question"] if isinstance(res["question"], dict) else res
        q_text = q_data.get("question")
        options = q_data.get("options")
        if q_text and options:
            q_map[qid] = (q_text, options)

    valid_indices = [i for i, row in df.iterrows() if row["question_id"] in q_map]
    df = df.loc[valid_indices]

    # Sample if requested
    if args.n is not None and len(df) > args.n:
        print(f"Sampling {args.n} from {len(df)} matched questions...")
        df = df.sample(n=args.n, random_state=args.seed)
    else:
        print(f"Using all {len(df)} matched questions.")

    df = df.reset_index(drop=True)
    if len(df) == 0:
        print("Error: No questions found after matching!")
        return

    # 2. Load Model
    tokenizer, model = load_model()

    # 3. Collect Data
    print("Collecting hidden states for all layers...")
    all_hidden_list = [] # List of tensors (num_layers, d)
    labels_list = []     # List of ints (1 or 0)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        qid = row["question_id"]
        q_text, options = q_map[qid]
        prompt = build_pass_game_prompt(q_text, options)

        # Run model
        hs_layers, choice = get_all_layers_hidden_states_and_choice(model, tokenizer, prompt)

        all_hidden_list.append(hs_layers.cpu())
        labels_list.append(1 if choice == "1" else 0)

    # Convert to tensors
    N = len(df)
    # Stack -> (N, num_layers, d)
    all_hidden = torch.stack(all_hidden_list)
    labels = np.array(labels_list, dtype=int)

    n_ans = (labels == 1).sum()
    n_pass = (labels == 0).sum()
    print(f"\nTotal Data: N={N}, Answer={n_ans}, Pass={n_pass}")

    if n_ans < 5 or n_pass < 5:
        print("Warning: Extreme class imbalance. Stats might be unstable.")

    # 4. Analyze each requested layer
    all_stats = []

    for layer in args.layers:
        print(f"\nProcessing Layer {layer}...")
        try:
            # all_hidden is (N, 80, d) assuming 80 layers
            # We need to map the canonical layer index to the tensor index.
            # get_all_layers.. returned outputs.hidden_states[1:] which is layers 0..L-1
            # So index `layer` corresponds to `layer`.
            if layer >= all_hidden.shape[1]:
                print(f"Error: Layer {layer} out of bounds (max {all_hidden.shape[1]-1}). skipping.")
                continue

            H_L = all_hidden[:, layer, :].to(model.device)

            stats = analyze_layer(
                layer, H_L, labels,
                n_splits=args.n_splits,
                seed=args.seed,
                device=model.device
            )
            all_stats.append(stats)

            # Print Summary
            print(f"===== Pass Game Direction Analysis (Layer {layer}) =====")
            print(f"N total: {stats['n_total']} (Answer={stats['n_answer']}, Pass={stats['n_pass']})")

            if stats["d_so"]["available"]:
                auc = stats["d_so"]["auc_ans_vs_pass"]
                cos = stats["d_so"]["cos_d_pass_full"]
                print(f"d_so AUC (Ans vs Pass):   {auc:.4f}")
                print(f"cos(d_pass_full, d_so):   {cos:.4f}")
            else:
                print("d_so: Not found")

            if stats["d_conf"]["available"]:
                auc = stats["d_conf"]["auc_ans_vs_pass"]
                cos = stats["d_conf"]["cos_d_pass_full"]
                print(f"d_conf AUC (Ans vs Pass): {auc:.4f}")
                print(f"cos(d_pass_full, d_conf): {cos:.4f}")
            else:
                print("d_conf: Not found")

            test_auc = stats["d_pass"]["split_test_auc_mean"]
            test_std = stats["d_pass"]["split_test_auc_std"]
            print(f"d_pass split-half test AUC: {test_auc:.4f} \u00b1 {test_std:.4f}")

        except Exception as e:
            print(f"Detailed error for layer {layer}: {e}")
            import traceback
            traceback.print_exc()

    # 5. Save Results
    out_file = OUTPUT_DIR / "pass_game_stats_all_layers.json"
    with open(out_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved all stats to {out_file}")

if __name__ == "__main__":
    main()
