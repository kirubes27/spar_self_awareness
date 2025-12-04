#!/usr/bin/env python3
"""
Logit Lens Sanity Check for Self vs Other Prompts.

This script:
1. Loads a batch of 50 questions from the contrastive pairs dataset.
2. Runs the model on "Self" prompts ("How confident are YOU?") and "Other" prompts ("How confident are OTHERS?").
3. Caches the hidden states at the last token for Layers 35, 50, and 79.
4. Averages the hidden states across the 50 examples (separately for Self and Other).
5. Projects the averaged states onto the vocabulary (Logit Lens).
6. Saves the top-k tokens to text files for inspection.

Goal: To see if the model is "thinking" about Self-concepts ("I", "my") vs Other-concepts ("they", "people")
at these specific layers, even before the final output.
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt_utils import build_self_prompt, build_other_prompt

# --------- CONFIG ---------
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"
DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
INPUT_CSV = DATA_DIR / f"{MODEL_NAME}_different_perspective_train.csv"
OUTPUT_DIR = Path("interp/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS_TO_CHECK = [35, 50, 79]
SAMPLE_SIZE = 50
TOP_K = 50

def load_model():
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

def get_unembedding_matrix(model):
    return model.lm_head.weight

def analyze_hidden_states(model, tokenizer, prompts, layer_indices, label):
    """
    Run model on prompts, collect hidden states at specified layers, average them, and project to vocab.
    """
    print(f"\nAnalyzing {len(prompts)} '{label}' prompts...")

    # Storage for hidden states: layer -> list of tensors
    hidden_states_by_layer = {layer: [] for layer in layer_indices}

    for prompt in tqdm(prompts):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Collect hidden states at last token
        for layer in layer_indices:
            # hidden_states[i+1] is output of layer i
            h = outputs.hidden_states[layer + 1][:, -1, :].squeeze(0)
            hidden_states_by_layer[layer].append(h)

    # Process each layer
    W_U = get_unembedding_matrix(model)

    for layer in layer_indices:
        # Average hidden states
        stacked_h = torch.stack(hidden_states_by_layer[layer])
        avg_h = stacked_h.mean(dim=0) # (hidden_dim,)

        # Project to vocab
        avg_h = avg_h.to(W_U.device, dtype=W_U.dtype)
        logits = torch.matmul(W_U, avg_h)

        # Get top-k
        scores = logits.float().cpu().numpy()
        top_indices = scores.argsort()[-TOP_K:][::-1]

        # Save to file
        filename = OUTPUT_DIR / f"sanity_check_{label}_layer{layer}.txt"
        print(f"  Saving top tokens for Layer {layer} to {filename}")

        with open(filename, "w") as f:
            header = f"Logit Lens Sanity Check: {label.upper()} Prompts (Avg of {len(prompts)})\n"
            header += f"Layer: {layer}\n"
            header += "=" * 60 + "\n"
            f.write(header)

            for rank, idx in enumerate(top_indices, start=1):
                token = tokenizer.convert_ids_to_tokens(int(idx))
                score = float(scores[int(idx)])
                printable = repr(token)
                line = f"{rank:2d}. id={idx:6d}  token={printable:20s}  score={score:.4f}\n"
                f.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE, help="Number of samples to average")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Take random sample
    if len(df) > args.n:
        df = df.sample(n=args.n, random_state=42)

    print(f"Selected {len(df)} questions for analysis.")

    # Build prompts
    # We need the question text. Assuming 'question_text' column exists (it should in our CSVs)
    # If not, we might need to load the JSON map like in other scripts.
    # Let's check if 'question_text' is in columns, otherwise load map.

    if "question_text" not in df.columns:
        # Fallback: load from compiled results
        print("  'question_text' column missing, loading from JSON map...")
        import json
        json_path = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"
        with open(json_path) as f:
            data = json.load(f)
        q_map = {k: v["question"] for k, v in data["results"].items()}
        questions = [q_map[qid] for qid in df["question_id"] if qid in q_map]
    else:
        questions = df["question_text"].tolist()

    self_prompts = [build_self_prompt(q) for q in questions]
    other_prompts = [build_other_prompt(q) for q in questions]

    # 2. Load Model
    tokenizer, model = load_model()

    # 3. Analyze Self Prompts
    analyze_hidden_states(model, tokenizer, self_prompts, LAYERS_TO_CHECK, "SELF")

    # 4. Analyze Other Prompts
    analyze_hidden_states(model, tokenizer, other_prompts, LAYERS_TO_CHECK, "OTHER")

    print("\nDone! Check interp/outputs/ for results.")

if __name__ == "__main__":
    main()
