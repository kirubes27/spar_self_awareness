#!/usr/bin/env python3
"""
Compare Pass Game Direction vs Self-Other Direction.

Goal:
1. Run the "Simplified Pass Game" prompt on a batch of questions.
2. Collect hidden states at the last token (decision point).
3. Group by model's choice: Answer ('1') vs Pass ('2').
4. Compute d_pass = mean(Answer) - mean(Pass).
5. Compare d_pass with our existing d_so (Self-Other) vector.

Prompt used (from pass_game_from_capabilities.py):
"Consider the answer to the following question:
[Question]
[Options]
Consider your level of certainty in your answer. If you are at least 10% certain that you know the correct answer, output '1'; if not, output '2'.
Respond ONLY with '1' or '2'.
Your choice ('1' or '2'): "
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.nn.functional as F

# --------- CONFIG ---------
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "llama-3.3-70b-instruct"
DATA_DIR = Path("contrastive_pairs") / MODEL_NAME
INPUT_CSV = DATA_DIR / f"{MODEL_NAME}_different_perspective_train.csv"
OUTPUT_DIR = Path("interp/outputs")
DIRECTION_FILE = OUTPUT_DIR / "self_other_direction_layer35.pt"

# Pass Game Prompt Template (Simplified Frame)
def build_pass_game_prompt(question_text, options_dict):
    # Format options
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

def get_hidden_states_and_choice(model, tokenizer, prompt, layer_idx):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

        # Get hidden state at last token
        h = outputs.hidden_states[layer_idx + 1][:, -1, :].squeeze(0)

        # Get model's choice (next token)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        # Check prob of '1' vs '2'
        id_1 = tokenizer.encode("1", add_special_tokens=False)[0]
        id_2 = tokenizer.encode("2", add_special_tokens=False)[0]

        prob_1 = probs[0, id_1].item()
        prob_2 = probs[0, id_2].item()

        choice = "1" if prob_1 > prob_2 else "2"

    return h, choice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--layer", type=int, default=35, help="Layer to analyze")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # We need full question data (options etc).
    # The CSV might not have options. Let's try to load from compiled JSON if needed.
    # For now, assuming we can get text. If options missing, we might need a fallback.
    # Actually, let's just use the 'question_text' from CSV and assume it contains options or is enough.
    # If not, we'll need to load the JSON map.

    import json
    json_path = Path("compiled_results_smc") / f"{MODEL_NAME}_phase1_compiled.json"
    print(f"Loading full question data from {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    # Create map: question_text -> options
    q_map = {}
    for qid, res in data["results"].items():
        q_data = res["question"] if isinstance(res["question"], dict) else res
        q_text = q_data.get("question")
        options = q_data.get("options")
        if q_text and options:
            q_map[q_text] = options

    # Filter DF to those we have options for
    valid_indices = [i for i, row in df.iterrows() if row["question_text"] in q_map]
    df = df.loc[valid_indices]

    if len(df) > args.n:
        df = df.sample(n=args.n, random_state=42)

    print(f"Selected {len(df)} questions.")

    # 2. Load Model
    tokenizer, model = load_model()

    # 3. Collect Data
    answer_states = []
    pass_states = []

    print(f"Running Pass Game prompts (Layer {args.layer})...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        q_text = row["question_text"]
        options = q_map[q_text]
        prompt = build_pass_game_prompt(q_text, options)

        h, choice = get_hidden_states_and_choice(model, tokenizer, prompt, args.layer)

        if choice == "1":
            answer_states.append(h)
        else:
            pass_states.append(h)

    print(f"\nStats: Answer='1': {len(answer_states)}, Pass='2': {len(pass_states)}")

    if len(answer_states) < 5 or len(pass_states) < 5:
        print("Warning: Not enough samples in one group to compute reliable direction.")
        # Proceed anyway but warn

    if not answer_states or not pass_states:
        print("Error: One group is empty! Cannot compute difference.")
        return

    # 4. Compute d_pass
    mean_answer = torch.stack(answer_states).mean(dim=0)
    mean_pass = torch.stack(pass_states).mean(dim=0)
    d_pass = mean_answer - mean_pass

    # 5. Load d_so and Compare
    print(f"Loading d_so from {DIRECTION_FILE}...")
    d_so_data = torch.load(DIRECTION_FILE, map_location=model.device)
    d_so = d_so_data["direction"].to(model.device)

    # Cosine Similarity
    cos_sim = F.cosine_similarity(d_pass.unsqueeze(0), d_so.unsqueeze(0)).item()

    print("\n" + "="*40)
    print(f"RESULTS (Layer {args.layer})")
    print("="*40)
    print(f"Cosine Similarity (d_pass vs d_so): {cos_sim:.4f}")
    print("="*40)

    # Save d_pass for future use
    save_path = OUTPUT_DIR / f"pass_game_direction_layer{args.layer}.pt"
    torch.save({"direction": d_pass.cpu(), "layer": args.layer, "cos_sim": cos_sim}, save_path)
    print(f"Saved d_pass to {save_path}")

if __name__ == "__main__":
    main()
