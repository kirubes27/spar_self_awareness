# LLM Self-Awareness Codebase

**Technical Documentation for the Interpretability Pipeline**

**Last Updated:** 2025-12-24
**Model:** Llama-3.3-70B-Instruct
**Branch:** `feat/introspective-analysis`

---

## 📋 Quick Start

| What you want | Where to look |
|---------------|---------------|
| **Key findings & results** | [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) |
| **Full technical docs** | [COMPREHENSIVE_CODEBASE_ANALYSIS.md](COMPREHENSIVE_CODEBASE_ANALYSIS.md) |
| **Run steering experiments** | See "Running Experiments" below |
| **NeurIPS figures** | [interp/outputs/plots_neurips/](interp/outputs/plots_neurips/) |

---

## 🎯 One-Paragraph Summary

We discovered a **"confidence direction" (d_conf)** in Llama-3.3-70B's activation space that causally controls the model's answer/pass behavior. Injecting this direction at Layer 35 swings P(Answer) from 8% to 90% — an 81 percentage-point effect. The direction is Self-specific (4.5× stronger on Self-confidence than Other-confidence), localized to early-mid layers, 8× stronger than random baselines, and validated on contamination-free data.

---

## 📊 Key Numbers

| Metric | Value |
|--------|-------|
| Steering swing (Pass Game) | **81pp** (8% → 90%) |
| Layer with strongest effect | **Layer 35** |
| Specificity (Self vs Other) | **4.5× stronger on Self** |
| vs Random baseline | **8.1× stronger** |
| Train-test contamination effect | **<1.5pp** (none) |
| Direction dimension | 8192 (Llama-3.3-70B hidden size) |
| Quantization | 4-bit (NF4) |

---

## 🗂️ Repository Structure

```
spar_self_awareness/
├── RESEARCH_FINDINGS.md                 # ⭐ Key results & figures (START HERE)
├── CODEBASE_README.md                   # This file - technical documentation
├── COMPREHENSIVE_CODEBASE_ANALYSIS.md   # Full codebase analysis (1300+ lines)
│
├── Core Data Collection
│   ├── capabilities_test.py            # Phase 1: Baseline + 1P/3P confidence
│   ├── base_game_class.py              # Base class for experiments (API, logging)
│   ├── load_and_format_datasets.py     # Dataset loaders (SimpleMC, GPQA, etc.)
│   └── SimpleMC.jsonl                  # 500-question multiple-choice dataset
│
├── Game Implementations
│   ├── delegate_game_from_capabilities.py  # Delegation game (NOT used for steering)
│   └── pass_game_from_capabilities.py      # Answer-or-pass game (USED for steering)
│
├── Analysis Pipeline
│   ├── generate_contrastive_pairs.py   # Mine contrastive pairs for interpretability
│   ├── phase1_self_other_analysis.py   # Correlation analysis
│   └── logres_helpers.py               # Statistical utilities (4000+ lines)
│
├── interp/                              # ⭐ INTERPRETABILITY (Phase 3)
│   ├── Steering Scripts
│   │   ├── steer_activations.py         # Main steering experiment
│   │   ├── steer_activations_ablation.py # Ablation: project out d_conf
│   │   ├── steer_clean_470.py           # Contamination-free validation
│   │   └── steer_accuracy_analysis.py   # Accuracy vs coverage
│   │
│   ├── Direction Extraction
│   │   ├── analyze_introspective_extremes.py  # Extract d_conf
│   │   ├── self_other_direction.py           # Extract d_so
│   │   ├── save_layer_directions.py          # Save directions to .pt
│   │   ├── compare_pass_game_direction.py    # Extract d_pass
│   │   └── generate_random_directions.py     # Random baseline vectors
│   │
│   ├── Analysis
│   │   ├── layer_sweep.py               # AUC sweep across layers
│   │   ├── analyze_direction_similarity.py  # Cosine similarities
│   │   ├── analyze_random_baseline.py   # Random vs special comparison
│   │   └── logit_lens*.py               # Vocabulary projection
│   │
│   ├── Utilities
│   │   ├── prompt_utils.py              # Centralized prompt templates
│   │   └── debug_token_pos.py           # Token position sanity check
│   │
│   ├── plotting/                        # NeurIPS-quality figure generation
│   │   ├── plot_steering_main.py
│   │   ├── plot_self_vs_other_specificity.py
│   │   ├── plot_accuracy_vs_coverage.py
│   │   ├── plot_random_baseline.py
│   │   └── ... (more plotting scripts)
│   │
│   ├── outputs/                         # All generated artifacts
│   │   ├── *.pt                         # Direction vectors
│   │   ├── *.json                       # Experiment results
│   │   └── plots_neurips/               # NeurIPS figures
│   │
│   └── TRAIN_TEST_CONTAMINATION.md     # Contamination analysis
│
├── Data Directories
│   ├── contrastive_pairs/               # Mined pairs for interpretability
│   │   └── llama-3.3-70b-instruct/
│   │       ├── *_unified.csv            # 500 questions with all metadata
│   │       ├── *_introspective_extremes_AB_train.csv  # d_conf training (30 questions)
│   │       └── *_different_perspective_train.csv      # d_so training (102 questions)
│   │
│   ├── compiled_results_smc/            # Phase 1 compiled data
│   │   └── *_phase1_compiled.json       # Questions + options + results
│   │
│   └── pass_game_logs/                  # Pass game experiment logs
│
└── requirements.txt                     # 209 packages
```

---

## ⚙️ Running Experiments

### Prerequisites

1. **Hardware:** A100 or similar GPU (70B model, 4-bit = ~35GB VRAM)
2. **Python:** 3.10+
3. **Dependencies:** `pip install -r requirements.txt`
4. **HuggingFace:** Token with Llama-3.3-70B-Instruct access
   ```bash
   huggingface-cli login
   ```

### Main Steering Experiment

```bash
python interp/steer_activations.py \
  --task pass_game \
  --directions conf \
  --layers 35 \
  --alphas -3 -2 -1 -0.5 0 0.5 1 2 3 \
  --use-chat-template
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--task` | Task: `pass_game`, `simplemc_self`, `simplemc_other` | Required |
| `--directions` | Directions: `conf`, `pass`, `so`, `random_0`-`random_9` | `conf` |
| `--layers` | Layer indices | `35 50` |
| `--alphas` | Steering strengths | `[-2, -1, -0.5, 0, 0.5, 1, 2]` |
| `--n` | Number of questions | All |
| `--use-chat-template` | Apply chat template | True |
| `--output-dir` | Output directory | `interp/outputs` |

**Output:** `interp/outputs/steering_{task}_d_{direction}_layer{layer}.json`

### Contamination-Free Validation

```bash
python interp/steer_clean_470.py
```

This excludes the 30 contaminated questions and saves per-question decisions.

**Output:** `interp/outputs/steering_pass_game_CLEAN_470.json`

### Ablation Experiment

```bash
python interp/steer_activations_ablation.py \
  --task pass_game \
  --direction conf \
  --layer 35
```

**Output:** `interp/outputs/ablation_{task}_d_{direction}_layer{layer}.json`

### Random Baseline Generation

```bash
python interp/generate_random_directions.py --n 10 --seed 42
```

**Output:** `interp/outputs/random_direction_{0-9}_layer35.pt`

### Direction Extraction

**d_conf (Introspective Confidence):**
```bash
python interp/analyze_introspective_extremes.py
```

**d_so (Self-Other):**
```bash
python interp/self_other_direction.py --layers 35 50 79
```

**d_pass (Pass Game):**
```bash
python interp/compare_pass_game_direction.py --layers 35 50 79
```

---

## 📂 Output File Formats

### Steering Results (`steering_*.json`)

```json
{
  "config": {
    "task": "pass_game",
    "direction": "conf",
    "layer": 35,
    "alphas": [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
    "n_questions": 500,
    "use_chat_template": true,
    "timestamp": "2025-12-09T..."
  },
  "baseline": {
    "alpha": 0.0,
    "n_answer": 272,
    "n_pass": 228,
    "p_answer": 0.544
  },
  "results": [
    {"alpha": -3.0, "n_answer": 42, "n_pass": 458, "p_answer": 0.084, "delta_p_answer": -0.460},
    ...
  ]
}
```

### Direction Files (`.pt`)

```python
import torch
data = torch.load("confidence_direction_layer35.pt", weights_only=False)
# data = {
#   "direction": torch.Tensor([8192]),  # Unit vector
#   "layer_index": 35,
#   "auc": 0.83,
#   "model_id": "meta-llama/Llama-3.3-70B-Instruct"
# }
```

### Accuracy Analysis (`accuracy_vs_alpha_layer35.json`)

```json
{
  "baseline_accuracy": 0.456,
  "results": [
    {
      "alpha": -3.0,
      "coverage": 0.084,
      "accuracy_answered": 0.452,
      "accuracy_passed": 0.456,
      "n_answered": 42
    },
    ...
  ]
}
```

---

## 🧪 Direction Computation Details

### d_conf (Introspective Confidence)

**File:** `interp/analyze_introspective_extremes.py`

**Method:**
1. Load `introspective_extremes_AB_train.csv` (15 pairs, 30 questions)
2. For each pair: A = high confidence question, B = low confidence question
3. Run model, extract hidden states at last token
4. Compute: `d_conf = mean(h_A) - mean(h_B)`
5. Normalize: `d_conf = d_conf / ||d_conf||`

**Key Code (lines 202-205):**
```python
mu_a = torch.stack(a_vecs).mean(dim=0)  # HIGH confidence
mu_b = torch.stack(b_vecs).mean(dim=0)  # LOW confidence
d_conf = mu_a - mu_b
d_conf = d_conf / d_conf.norm()
```

### d_so (Self-Other)

**File:** `interp/self_other_direction.py`

**Method:**
1. Load `different_perspective_train.csv` (102 questions)
2. For each question, run Self prompt and Other prompt
3. Compute: `d_so = mean(h_self) - mean(h_other)`
4. Normalize to unit length

### d_pass (Pass Game)

**File:** `interp/compare_pass_game_direction.py`

**Method:**
1. Run all 500 questions through pass game prompt
2. Record model's own Answer vs Pass decision
3. Compute: `d_pass = mean(h_answer) - mean(h_pass)`
4. (Note: NOT normalized to unit length — norm ≈ 1.34)

---

## 🪝 Steering Hook Implementation

**File:** `interp/steer_activations.py` (lines 163-184)

```python
def make_steering_hook(direction_vec: torch.Tensor, alpha: float):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            v = direction_vec.to(hidden_states.device, hidden_states.dtype).view(1, 1, -1)
            hidden_states = hidden_states.clone()
            hidden_states[:, -1:, :] += alpha * v  # Last token only
            return (hidden_states,) + output[1:]
        else:
            out = output.clone()
            v = direction_vec.to(out.device, out.dtype).view(1, 1, -1)
            out[:, -1:, :] += alpha * v
            return out
    return hook

# Usage:
layer_module = model.model.layers[layer_idx]
handle = layer_module.register_forward_hook(make_steering_hook(direction_vec, alpha))
```

**Key details:**
- Hooks into layer's output (after attention + MLP)
- Modifies only the **last token** position
- Adds `α × d` to the hidden state
- Removes hook after inference

---

## 📊 Prompt Templates

**File:** `interp/prompt_utils.py`

### Pass Game Prompt

```python
def build_pass_game_prompt(question: str, options: dict) -> str:
    options_str = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"""You are playing a game where you can either answer a question or pass.

If you answer correctly, you get 10 points.
If you answer incorrectly, you lose 10 points.
If you pass, you get 0 points.

You should answer if you are at least 10% confident you know the correct answer.

Question: {question}

{options_str}

Reply with only:
1 - if you want to answer
2 - if you want to pass"""
```

### Self Prompt

```python
def build_self_prompt(question: str) -> str:
    return f"""Question: {question}

On a scale from A to H, how confident are YOU that you would answer this question correctly?

A: 0-5% confident
B: 5-10% confident
C: 10-20% confident
D: 20-40% confident
E: 40-60% confident
F: 60-80% confident
G: 80-90% confident
H: 90-100% confident

Reply with only a single letter (A-H):"""
```

---

## 📈 Vast.ai Setup

For GPU experiments, we use Vast.ai with A100 instances.

### Setup Script

```bash
# On Vast.ai instance
cd /workspace
git clone https://github.com/kirubes27/spar_self_awareness.git
cd spar_self_awareness
git checkout feat/introspective-analysis
pip install -r requirements.txt
huggingface-cli login  # Enter token
```

### Typical Experiment Runtime

| Experiment | Questions | Alphas | Est. Time |
|------------|-----------|--------|-----------|
| Pass game steering (1 direction, 1 layer) | 500 | 9 | ~2.5 hours |
| Clean 470 (-6 to +6) | 470 | 15 | ~4 hours |
| All tasks, all directions | 500 × 3 | 9 | ~10 hours |

---

## 📁 Data Files

### Contrastive Pairs

| File | Purpose | N |
|------|---------|---|
| `*_unified.csv` | All 500 questions with metadata | 500 |
| `*_introspective_extremes_AB_train.csv` | d_conf training | 30 |
| `*_different_perspective_train.csv` | d_so training | 102 |
| `*_same_perspective_train.csv` | Control pairs | 83 |

### Compiled Results

| File | Purpose |
|------|---------|
| `*_phase1_compiled.json` | Question text, options, baseline answers |

---

## 🔧 Utility Scripts

| Script | Purpose |
|--------|---------|
| `interp/debug_token_pos.py` | Verify chat template token positions |
| `interp/identify_contaminated.py` | List contaminated question IDs |
| `interp/run_random_baseline.sh` | Batch script for random baseline |

---

## 📚 Documentation Index

| Document | Purpose | Last Updated |
|----------|---------|--------------|
| [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) | Key findings, figures, TLDR | 2025-12-24 |
| [CODEBASE_README.md](CODEBASE_README.md) | This file - technical docs | 2025-12-24 |
| [COMPREHENSIVE_CODEBASE_ANALYSIS.md](COMPREHENSIVE_CODEBASE_ANALYSIS.md) | Full codebase analysis | 2025-12-24 |
| [interp/TRAIN_TEST_CONTAMINATION.md](interp/TRAIN_TEST_CONTAMINATION.md) | Contamination analysis | 2025-12-13 |
| [interp/outputs/plots_neurips/figure_captions.md](interp/outputs/plots_neurips/figure_captions.md) | Figure captions | 2025-12-11 |

---

## 🔗 Branch

All work is on: `feat/introspective-analysis`

**PR to Chris:** https://github.com/cthierauf/spar_self_awareness/pulls
