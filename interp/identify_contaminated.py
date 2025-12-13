#!/usr/bin/env python3
"""
steer_clean_470.py

Re-run steering on 470 non-contaminated questions (excluding 30 training questions).
Also saves per-question decisions to check flip patterns.
"""

import json

import pandas as pd


# Get the 30 contaminated question IDs
train = pd.read_csv(
    "contrastive_pairs/llama-3.3-70b-instruct/llama-3.3-70b-instruct_introspective_extremes_AB_train.csv"
)
CONTAMINATED_QIDS = set(train["A_qid"].tolist() + train["B_qid"].tolist())

print(f"Excluding {len(CONTAMINATED_QIDS)} contaminated questions")

# Save to file for the steering script
with open("interp/outputs/contaminated_qids.json", "w") as f:
    json.dump(list(CONTAMINATED_QIDS), f)

print("Saved contaminated question IDs to interp/outputs/contaminated_qids.json")
print("\nRun steering with:")
print(
    "  python interp/steer_activations.py --task pass_game --direction conf --layer 35 --exclude-qids interp/outputs/contaminated_qids.json --save-per-question"
)
