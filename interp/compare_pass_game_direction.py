#!/usr/bin/env python3
"""
Compare Pass Game Direction vs Self-Other Direction (upgraded).

For a single layer L:
  1. Run the simplified Pass Game prompt on a batch of questions.
  2. Collect hidden states at the last token.
  3. Label each state as Answer (1) or Pass (0).
  4. Evaluate:
      - How well the existing d_so separates Answer vs Pass (AUC).
      - A new d_pass direction (Answer - Pass) with split-half AUC.
      - Cosine alignment between d_pass and d_so.
  5. Save d_pass and a JSON stats file.

Run e.g.:
  python interp/compare_pass_game_direction.py --layer 35 --n 200 --n-splits 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

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
INPUT_CSV = DATA_DIR / f"{MODEL_NAME}_different_perspective_train.csv"

COMPILED_JSON = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"

OUTPUT_DIR = Path("interp/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------- Utilities ---------
def build_pass_game_prompt(question_text: str, options_dict: dict) -> str:
    """Builds the simplified pass-game prompt used for the decision-only test."""
    # Format options as:
    # A: ...
    # B: ...
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


def get_hidden_state_and_choice(
    model, tokenizer, prompt: str, layer_idx: int
) -> Tuple[torch.Tensor, str]:
    """
    Run the model on a single pass-game prompt and return:
      - h_L: hidden state at layer L (last token)
      - choice: "1" or "2" based on next-token argmax prob
    """
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

        # hidden_states index: 0 = embeddings, 1 = layer0, ..., L+1 = layer L
        h = outputs.hidden_states[layer_idx + 1][:, -1, :].squeeze(0)

        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        # ids for "1" and "2"
        id_1 = tokenizer.encode("1", add_special_tokens=False)[0]
        id_2 = tokenizer.encode("2", add_special_tokens=False)[0]

        prob_1 = probs[0, id_1].item()
        prob_2 = probs[0, id_2].item()

        choice = "1" if prob_1 > prob_2 else "2"

    return h, choice


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute ROC AUC for binary labels without sklearn.
    labels: 1 for Answer, 0 for Pass.
    """
    assert scores.shape[0] == labels.shape[0]
    # sort scores ascending
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores)) + 1  # 1-based ranks

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_pos_ranks = ranks[pos_mask].sum()
    # Mann–Whitney U -> AUC
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# --------- Main ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150, help="Number of questions to sample")
    parser.add_argument("--layer", type=int, default=35, help="Layer index (0-based)")
    parser.add_argument(
        "--direction-file",
        type=str,
        default=None,
        help="Path to self_other_direction_layer{L}.pt (defaults to interp/outputs/self_other_direction_layer{L}.pt)",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Split-half repetitions")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for splits")
    args = parser.parse_args()

    layer = args.layer

    # 1. Load CSV with question_text
    print(f"Loading CSV from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # 2. Load compiled JSON to recover options
    print(f"Loading full question data from {COMPILED_JSON}...")
    with open(COMPILED_JSON) as f:
        data = json.load(f)

    # Map: question_text -> options dict
    q_map = {}
    for qid, res in data["results"].items():
        q_data = res["question"] if isinstance(res["question"], dict) else res
        q_text = q_data.get("question")
        options = q_data.get("options")
        if q_text and options:
            q_map[q_text] = options

    # Filter df to rows where we have options
    valid_indices = [i for i, row in df.iterrows() if row["question_text"] in q_map]
    df = df.loc[valid_indices]

    if len(df) > args.n:
        df = df.sample(n=args.n, random_state=42)

    df = df.reset_index(drop=True)
    print(f"Selected {len(df)} questions with options.")

    # 3. Load model
    tokenizer, model = load_model()

    # 4. Collect activations
    answer_states: List[torch.Tensor] = []
    pass_states: List[torch.Tensor] = []

    print(f"Running simplified Pass Game (layer {layer})...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        q_text = row["question_text"]
        options = q_map[q_text]
        prompt = build_pass_game_prompt(q_text, options)

        h, choice = get_hidden_state_and_choice(model, tokenizer, prompt, layer)
        if choice == "1":
            answer_states.append(h)
        else:
            pass_states.append(h)

    n_answer = len(answer_states)
    n_pass = len(pass_states)
    print(f"\nStats: Answer='1': {n_answer}, Pass='2': {n_pass}")

    if n_answer < 5 or n_pass < 5:
        print("Warning: very unbalanced groups; results may be noisy.")

    if n_answer == 0 or n_pass == 0:
        print("Error: one group is empty; aborting.")
        return

    # Stack all states
    all_states = answer_states + pass_states
    H = torch.stack(all_states)  # (N, d)
    labels = np.array([1] * n_answer + [0] * n_pass, dtype=int)

    # 5. Load d_so for this layer
    if args.direction_file is not None:
        direction_path = Path(args.direction_file)
    else:
        direction_path = OUTPUT_DIR / f"self_other_direction_layer{layer}.pt"

    print(f"Loading d_so from {direction_path}...")
    d_so_data = torch.load(direction_path, map_location=model.device)
    d_so = d_so_data["direction"].to(model.device)  # (d,)

    # 6. Evaluate d_so on Answer vs Pass
    with torch.no_grad():
        scores_dso = (H @ d_so).cpu().numpy()
    mean_ans = float(scores_dso[:n_answer].mean())
    mean_pass = float(scores_dso[n_answer:].mean())

    # Orientation: flip so that Answer has higher mean score
    flipped = False
    if mean_ans < mean_pass:
        scores_dso *= -1.0
        d_so = -d_so
        mean_ans, mean_pass = -mean_ans, -mean_pass
        flipped = True

    auc_dso = compute_auc(scores_dso, labels)

    # 7. Learn d_pass on full data
    mean_answer = torch.stack(answer_states).mean(dim=0)
    mean_pass_vec = torch.stack(pass_states).mean(dim=0)
    d_pass_full = mean_answer - mean_pass_vec  # oriented toward Answer

    cos_full = float(
        F.cosine_similarity(d_pass_full.unsqueeze(0), d_so.unsqueeze(0)).item()
    )

    # 8. Split-half evaluation for d_pass
    rng = np.random.default_rng(args.seed)
    n = H.shape[0]
    indices = np.arange(n)

    split_metrics = []
    for split in range(args.n_splits):
        rng.shuffle(indices)
        mid = n // 2
        train_idx = indices[:mid]
        test_idx = indices[mid:]

        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Need both classes in train
        if train_labels.sum() == 0 or train_labels.sum() == len(train_labels):
            continue

        H_train = H[train_idx]
        H_test = H[test_idx]

        # Build d_pass on train
        ans_mask = train_labels == 1
        pass_mask = train_labels == 0
        mean_ans_train = H_train[ans_mask].mean(dim=0)
        mean_pass_train = H_train[pass_mask].mean(dim=0)
        d_pass_train = mean_ans_train - mean_pass_train

        with torch.no_grad():
            train_scores = (H_train @ d_pass_train).cpu().numpy()
            test_scores = (H_test @ d_pass_train).cpu().numpy()

        train_auc = compute_auc(train_scores, train_labels)
        test_auc = compute_auc(test_scores, test_labels)
        cos_split = float(
            F.cosine_similarity(d_pass_train.unsqueeze(0), d_so.unsqueeze(0)).item()
        )

        split_metrics.append(
            {
                "train_auc": float(train_auc),
                "test_auc": float(test_auc),
                "cos": cos_split,
            }
        )

    if not split_metrics:
        print("Warning: no valid split-half runs (class imbalance).")

    # Aggregate split stats
    def mean_std(key: str) -> Tuple[float, float]:
        if not split_metrics:
            return float("nan"), float("nan")
        vals = np.array([m[key] for m in split_metrics], dtype=float)
        return float(vals.mean()), float(vals.std())

    train_auc_mean, train_auc_std = mean_std("train_auc")
    test_auc_mean, test_auc_std = mean_std("test_auc")
    cos_mean, cos_std = mean_std("cos")

    # 9. Print summary
    print("\n" + "=" * 50)
    print(f"Pass Game Direction Analysis (Layer {layer})")
    print("=" * 50)
    print(f"N total: {n} (Answer={n_answer}, Pass={n_pass})")
    print(f"d_so orientation flipped for this task: {flipped}")
    print(f"d_so Answer mean proj:   {mean_ans:.4f}")
    print(f"d_so Pass mean proj:     {mean_pass:.4f}")
    print(f"d_so AUC (Ans vs Pass):  {auc_dso:.4f}")
    print(f"d_pass_full · d_so (cos): {cos_full:.4f}")
    print(f"d_pass split-half train AUC: {train_auc_mean:.4f} ± {train_auc_std:.4f}")
    print(f"d_pass split-half test  AUC: {test_auc_mean:.4f} ± {test_auc_std:.4f}")
    print(f"d_pass split-half cos(d_pass, d_so): {cos_mean:.4f} ± {cos_std:.4f}")
    print("=" * 50)

    # 10. Save artifacts
    pass_vec_path = OUTPUT_DIR / f"pass_game_direction_layer{layer}.pt"
    torch.save(
        {
            "direction": d_pass_full.cpu(),
            "layer": int(layer),
            "cos_sim_full": cos_full,
            "d_so_flipped_for_task": flipped,
        },
        pass_vec_path,
    )
    print(f"Saved d_pass_full to {pass_vec_path}")

    stats = {
        "layer": int(layer),
        "n_total": int(n),
        "n_answer": int(n_answer),
        "n_pass": int(n_pass),
        "d_so_path": str(direction_path),
        "d_so_flipped_for_task": flipped,
        "d_so_mean_answer_proj": mean_ans,
        "d_so_mean_pass_proj": mean_pass,
        "d_so_auc_ans_vs_pass": auc_dso,
        "d_pass_full_cos_with_d_so": cos_full,
        "split_metrics": split_metrics,
        "split_train_auc_mean": train_auc_mean,
        "split_train_auc_std": train_auc_std,
        "split_test_auc_mean": test_auc_mean,
        "split_test_auc_std": test_auc_std,
        "split_cos_mean": cos_mean,
        "split_cos_std": cos_std,
    }
    stats_path = OUTPUT_DIR / f"pass_game_stats_layer{layer}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
