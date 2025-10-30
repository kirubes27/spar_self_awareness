#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_contrastive_pairs.py

Zero-config miner for contrastive pairs using Phase-1 (SimpleMC-500) artifacts.

Assumed repo layout (from repo root):
  - compiled_results_smc/<model>_phase1_compiled.json        (baseline A/B/C/D with probs dict)
  - capabilities_1p_test_logs/<model>_SimpleMC_500_*_test_data.json    (1P/self confidence)
  - capabilities_3p_test_logs/<model>_SimpleMC_500_*_test_data.json    (3P/other confidence)

Outputs (in contrastive_pairs/<model>/):
  - <model>_unified.csv                (all aligned data: qid, H, pmax, correct, SelfProb, OtherProb, etc.)
  - <model>_self_other_gap.csv         (top ~100 by |SelfProb-OtherProb| with pmax>=0.55)
  - <model>_calibration_extremes.csv   (≤50 pairs: overconf-wrong Self>=0.8 ↔ underconf-right Self<=0.2)
  - <model>_easy_vs_hard.csv           (≤100 pairs, matched by correct_answer letter; easy pmax≥0.8 vs hard pmax≤0.4)

Run:
  python generate_contrastive_pairs.py

Configuration via environment variables (optional):
  - P_MAX_MIN_FOR_GAP=0.55         # Minimum pmax to consider for self-other gap pairs
  - EASY_PMAX_TH=0.80              # Threshold for "easy" questions
  - HARD_PMAX_TH=0.40              # Threshold for "hard" questions  
  - OC_WRONG_SELF_TH=0.80          # Overconfident-wrong threshold
  - UC_RIGHT_SELF_TH=0.20          # Underconfident-right threshold
  - TOP_GAP_N=100                  # Number of self-other gap pairs to extract
  - CALIBRATION_PAIRS=50           # Number of calibration extreme pairs
  - EASY_HARD_PER_LETTER=25        # Pairs per answer letter for easy-vs-hard
  - MIN_ALIGNED=480                # Minimum aligned questions required
"""

import os
import glob
import json
import math
import time
import re
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


# ---------- Config ----------
P_MAX_MIN_FOR_GAP = float(os.environ.get("P_MAX_MIN_FOR_GAP", "0.55"))
EASY_PMAX_TH = float(os.environ.get("EASY_PMAX_TH", "0.80"))
HARD_PMAX_TH = float(os.environ.get("HARD_PMAX_TH", "0.40"))
OC_WRONG_SELF_TH = float(os.environ.get("OC_WRONG_SELF_TH", "0.80"))
UC_RIGHT_SELF_TH = float(os.environ.get("UC_RIGHT_SELF_TH", "0.20"))
TOP_GAP_N = int(os.environ.get("TOP_GAP_N", "100"))
CALIBRATION_PAIRS = int(os.environ.get("CALIBRATION_PAIRS", "50"))
EASY_HARD_PER_LETTER = int(os.environ.get("EASY_HARD_PER_LETTER", "25"))
MIN_ALIGNED = int(os.environ.get("MIN_ALIGNED", "480"))

# List of models to process (same as phase1_self_other_analysis.py)
MODEL_NAMES = [
    "claude-3-5-sonnet-20241022",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5-20250929_think",
    "deepseek-chat",
    "deepseek-chat-v3.1",
    "deepseek-r1",
    "deepseek-v3.1-base",
    "gemini-2.5-flash_nothink",
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-08-06",
    "grok-3-latest",
    "hermes-4-70b",
    "kimi-k2",
    "kimi-k2-0905",
    "llama-3.1-405b-instruct",
    "llama-3.1-8b-instruct",
    "llama-3.3-70b-instruct",
    "llama-4-maverick",
    "mistral-small-3.2-24b-instruct",
    "openai-gpt-5-chat",
    "qwen3-235b-a22b-2507",
]


# ---------- Helpers ----------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _compute_entropy(prob_dict):
    """
    Compute natural-log entropy from a probability dict.
    Handles normalization and NaN cases.
    """
    if not isinstance(prob_dict, dict):
        return float("nan")
    
    vals = []
    for k, v in prob_dict.items():
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    
    s = sum(vals)
    if s <= 0:
        return float("nan")
    
    probs = [v/s for v in vals]
    ent = 0.0
    for p in probs:
        if p > 0:
            ent += -p * math.log(p)
    return ent


def _parse_baseline_phase1(json_obj):
    """
    Returns dict[qid] = {
        "entropy": <float>,
        "pmax": <float>,
        "actual_correct": <0.0/1.0 float>,
        "correct_answer": <str>,
        "question_text": <str>,
    }
    Expects compiled_results_smc/<model>_phase1_compiled.json shape.
    """
    out = {}
    results = json_obj.get("results", {})
    if not isinstance(results, dict):
        return out

    for qid, rec in results.items():
        # per-Q entropy of model's *answering* logits
        ent = _compute_entropy(rec.get("probs", {}))
        
        # pmax from probs
        prob_dict = rec.get("probs", {})
        if isinstance(prob_dict, dict) and prob_dict:
            pmax = max(prob_dict.values()) if prob_dict else 0.0
        else:
            pmax = float("nan")

        # actual correctness of model's chosen answer
        is_corr = rec.get("is_correct")
        if isinstance(is_corr, bool):
            acc = 1.0 if is_corr else 0.0
        elif isinstance(is_corr, (int, float)):
            acc = float(is_corr)
        else:
            acc = float("nan")

        # correct answer label
        correct_answer = ""
        if "correct_answer_label" in rec:
            correct_answer = rec["correct_answer_label"]
        elif "correct_answer" in rec:
            correct_answer = rec["correct_answer"]
        elif "question" in rec and isinstance(rec["question"], dict):
            correct_answer = rec["question"].get("correct_answer_label", 
                                                 rec["question"].get("correct_answer", ""))

        question_text = ""
        q_block = rec.get("question", {})
        if isinstance(q_block, dict):
            question_text = q_block.get("text", "").strip()
        elif "question_text" in rec:
            question_text = rec["question_text"]

        out[qid] = {
            "entropy": ent,
            "pmax": pmax,
            "actual_correct": acc,
            "correct_answer": correct_answer,
            "question_text": question_text,
        }
    return out


def _parse_capabilities_file(json_obj):
    """
    Returns dict[qid] = {
        "expected_prob": <float in [0,1]>,
        "question_text": <str>,
    }

    Expects capabilities_[13]p_test_logs/..._test_data.json shape, where
    'is_correct' is ALREADY the model's *expected correctness probability*
    (midpoint-weighted A..H bin, not a binary 0/1).
    """
    out = {}
    results = json_obj.get("results", {})
    if not isinstance(results, dict):
        return out

    for qid, rec in results.items():
        exp_prob = _safe_float(rec.get("is_correct"))

        question_text = ""
        q_block = rec.get("question", {})
        if isinstance(q_block, dict):
            question_text = q_block.get("text", "").strip()

        out[qid] = {
            "expected_prob": exp_prob,
            "question_text": question_text,
        }
    return out


def _find_best_cap_file(dir_path, model_name):
    """
    Find the "best" 1P / 3P capability file for a model.

    We look for files like:
        <model>_SimpleMC_500_<timestamp>_test_data.json
    in dir_path. We prefer qcount>=400 (so 500-Q runs over 50-Q pilots),
    then highest timestamp.

    Returns (chosen_path, records_dict) OR (None, {}).
    """
    if not os.path.isdir(dir_path):
        return None, {}

    pattern = os.path.join(dir_path, f"{model_name}_SimpleMC_*_test_data.json")
    candidates = glob.glob(pattern)

    parsed = []
    for p in candidates:
        base = os.path.basename(p)
        m = re.match(
            rf"^{re.escape(model_name)}_SimpleMC_(\d+)_([0-9]+)_test_data\.json$",
            base
        )
        if not m:
            continue
        qcount = int(m.group(1))   # "500" vs "50"
        ts     = int(m.group(2))   # timestamp-ish integer
        parsed.append((qcount, ts, p))

    if not parsed:
        return None, {}

    # Sort so best choice is first:
    #   1. prefer qcount >= 400
    #   2. then larger qcount
    #   3. then newer timestamp
    parsed.sort(key=lambda x: (x[0] >= 400, x[0], x[1]), reverse=True)
    chosen_path = parsed[0][2]

    try:
        with open(chosen_path, "r") as f:
            data = json.load(f)
        recs = _parse_capabilities_file(data)
    except Exception:
        return None, {}

    return chosen_path, recs


def build_unified_df(model_name: str) -> Optional[pd.DataFrame]:
    """
    Build unified dataframe for a single model by aligning baseline, 1P, and 3P data.
    Returns None if insufficient data.
    """
    baseline_path = os.path.join("compiled_results_smc", f"{model_name}_phase1_compiled.json")
    
    if not os.path.exists(baseline_path):
        log(f"  SKIP: Missing baseline file {baseline_path}")
        return None

    # Load baseline
    try:
        with open(baseline_path, "r") as f:
            baseline_json = json.load(f)
        base_records = _parse_baseline_phase1(baseline_json)
    except Exception as e:
        log(f"  SKIP: Failed to read baseline ({e})")
        return None

    # Load best 1P (self) file
    onep_path, onep_records = _find_best_cap_file("capabilities_1p_test_logs", model_name)
    # Load best 3P (other) file
    threep_path, threep_records = _find_best_cap_file("capabilities_3p_test_logs", model_name)

    if not onep_path or not threep_path:
        log(f"  SKIP: Missing 1P or 3P files (1P={onep_path}, 3P={threep_path})")
        return None

    log(f"  Baseline: {baseline_path}")
    log(f"  1P/self:  {onep_path}")
    log(f"  3P/other: {threep_path}")

    # Align questions across baseline, 1P, 3P
    aligned_ids = sorted(
        set(base_records.keys())
        & set(onep_records.keys())
        & set(threep_records.keys())
    )
    aligned_n = len(aligned_ids)

    if aligned_n < MIN_ALIGNED:
        log(f"  SKIP: Insufficient aligned questions ({aligned_n} < {MIN_ALIGNED})")
        return None

    log(f"  Aligned questions: {aligned_n}")

    # Build DataFrame with all variables
    data_rows = []
    for qid in aligned_ids:
        ent        = base_records[qid]["entropy"]
        pmax       = base_records[qid]["pmax"]
        acc        = base_records[qid]["actual_correct"]
        corr_ans   = base_records[qid]["correct_answer"]
        q_text     = base_records[qid]["question_text"]
        self_prob  = onep_records[qid]["expected_prob"]
        other_prob = threep_records[qid]["expected_prob"]
        
        data_rows.append({
            'question_id': qid,
            'entropy': ent,
            'pmax': pmax,
            'correct': acc,
            'correct_answer': corr_ans,
            'SelfProb': self_prob,
            'OtherProb': other_prob,
            'question_text': q_text,
        })

    df = pd.DataFrame(data_rows)

    # Replace inf with NaN for proper handling
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaNs in critical fields
    df = df.dropna(subset=['entropy', 'pmax', 'correct', 'SelfProb', 'OtherProb'])
    
    log(f"  Final rows after NaN removal: {len(df)}")

    # Quick sanity check correlations
    try:
        r_ent_self = df['entropy'].corr(-df['SelfProb'])
        r_ent_other = df['entropy'].corr(-df['OtherProb'])
        log(f"  Quick check: corr(H, -Self)={r_ent_self:.3f}, corr(H, -Other)={r_ent_other:.3f}")
    except Exception:
        pass

    return df


def mine_pairs(df: pd.DataFrame, outdir: str, model_name: str) -> None:
    """
    Generate three types of contrastive pairs and save to CSV files.
    """
    os.makedirs(outdir, exist_ok=True)

    # Save unified data
    unified_path = os.path.join(outdir, f"{model_name}_unified.csv")
    df.to_csv(unified_path, index=False)
    log(f"  Saved unified data: {unified_path}")

    # 1) Self vs Other big gap pairs
    df_gap = df[df["pmax"] >= P_MAX_MIN_FOR_GAP].copy()
    df_gap["gap"] = (df_gap["SelfProb"] - df_gap["OtherProb"]).abs()
    df_gap = df_gap.sort_values("gap", ascending=False).head(TOP_GAP_N)
    
    gap_out = df_gap[["question_id", "SelfProb", "OtherProb", "gap", "pmax", "correct", 
                      "entropy", "correct_answer", "question_text"]].copy()
    gap_path = os.path.join(outdir, f"{model_name}_self_other_gap.csv")
    gap_out.to_csv(gap_path, index=False)
    log(f"  Saved self-other gap pairs: {gap_path} ({len(gap_out)} pairs)")

    # 2) Calibration extremes pairs
    overconf = df[(df["correct"] == 0) & (df["SelfProb"] >= OC_WRONG_SELF_TH)].copy()
    underconf = df[(df["correct"] == 1) & (df["SelfProb"] <= UC_RIGHT_SELF_TH)].copy()
    
    overconf = overconf.sort_values("SelfProb", ascending=False)
    underconf = underconf.sort_values("SelfProb", ascending=True)
    
    n_pairs = min(len(overconf), len(underconf), CALIBRATION_PAIRS)
    
    pairs = []
    for i in range(n_pairs):
        oc = overconf.iloc[i]
        uc = underconf.iloc[i]
        pairs.append({
            "pair_id": f"calib_{i:03d}",
            "overconf_qid": oc["question_id"],
            "overconf_self": oc["SelfProb"],
            "overconf_pmax": oc["pmax"],
            "overconf_entropy": oc["entropy"],
            "overconf_question": oc["question_text"],
            "underconf_qid": uc["question_id"],
            "underconf_self": uc["SelfProb"],
            "underconf_pmax": uc["pmax"],
            "underconf_entropy": uc["entropy"],
            "underconf_question": uc["question_text"],
        })
    
    if pairs:
        calib_path = os.path.join(outdir, f"{model_name}_calibration_extremes.csv")
        pd.DataFrame(pairs).to_csv(calib_path, index=False)
        log(f"  Saved calibration extreme pairs: {calib_path} ({len(pairs)} pairs)")
    else:
        log(f"  WARNING: No calibration extreme pairs found")

    # 3) Easy vs Hard pairs (matched by correct_answer)
    if "correct_answer" in df.columns and df["correct_answer"].notna().any():
        easy = df[df["pmax"] >= EASY_PMAX_TH].copy().sort_values("pmax", ascending=False)
        hard = df[df["pmax"] <= HARD_PMAX_TH].copy().sort_values("pmax", ascending=True)
        
        pairs_eh = []
        for letter in sorted(df["correct_answer"].dropna().unique()):
            easy_l = easy[easy["correct_answer"] == letter].reset_index(drop=True)
            hard_l = hard[hard["correct_answer"] == letter].reset_index(drop=True)
            
            c = min(len(easy_l), len(hard_l), EASY_HARD_PER_LETTER)
            for i in range(c):
                e = easy_l.iloc[i]
                h = hard_l.iloc[i]
                pairs_eh.append({
                    "pair_id": f"eh_{letter}_{i:03d}",
                    "answer_letter": letter,
                    "easy_qid": e["question_id"],
                    "easy_pmax": e["pmax"],
                    "easy_entropy": e["entropy"],
                    "easy_correct": int(e["correct"]),
                    "easy_question": e["question_text"],
                    "hard_qid": h["question_id"],
                    "hard_pmax": h["pmax"],
                    "hard_entropy": h["entropy"],
                    "hard_correct": int(h["correct"]),
                    "hard_question": h["question_text"],
                })
        
        if pairs_eh:
            eh_path = os.path.join(outdir, f"{model_name}_easy_vs_hard.csv")
            pd.DataFrame(pairs_eh).to_csv(eh_path, index=False)
            log(f"  Saved easy-vs-hard pairs: {eh_path} ({len(pairs_eh)} pairs)")
        else:
            log(f"  WARNING: No easy-vs-hard pairs found")
    else:
        log(f"  SKIP easy-vs-hard: No correct_answer labels available")


def main():
    log("=== Contrastive Pair Generation ===")
    log(f"Configuration:")
    log(f"  P_MAX_MIN_FOR_GAP = {P_MAX_MIN_FOR_GAP}")
    log(f"  EASY_PMAX_TH = {EASY_PMAX_TH}")
    log(f"  HARD_PMAX_TH = {HARD_PMAX_TH}")
    log(f"  OC_WRONG_SELF_TH = {OC_WRONG_SELF_TH}")
    log(f"  UC_RIGHT_SELF_TH = {UC_RIGHT_SELF_TH}")
    log(f"  TOP_GAP_N = {TOP_GAP_N}")
    log(f"  CALIBRATION_PAIRS = {CALIBRATION_PAIRS}")
    log(f"  EASY_HARD_PER_LETTER = {EASY_HARD_PER_LETTER}")
    log(f"  MIN_ALIGNED = {MIN_ALIGNED}")
    log("")

    outroot = "contrastive_pairs"
    os.makedirs(outroot, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    for model_name in MODEL_NAMES:
        log(f"Processing: {model_name}")
        
        df = build_unified_df(model_name)
        
        if df is None or len(df) < MIN_ALIGNED:
            skipped_count += 1
            continue
        
        outdir = os.path.join(outroot, model_name)
        mine_pairs(df, outdir, model_name)
        processed_count += 1
        log("")

    log("=== Summary ===")
    log(f"Processed: {processed_count} models")
    log(f"Skipped: {skipped_count} models")
    log(f"Output directory: {outroot}/")
    log("Done!")


if __name__ == "__main__":
    main()

