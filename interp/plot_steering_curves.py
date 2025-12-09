#!/usr/bin/env python3
"""
plot_steering_curves.py

Generate NeurIPS-quality steering curves visualization.
Shows P(Answer) vs α for d_conf, d_pass, d_so across layers.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style settings for NeurIPS
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paths
OUTPUT_DIR = Path("interp/outputs")
LAYERS = [35, 50]
DIRECTIONS = ["conf", "pass", "so"]

# Colors and styles
STYLES = {
    "conf": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "label": r"$d_{\mathrm{conf}}$ (introspective)"},
    "pass": {"color": "#2ca02c", "marker": "^", "linestyle": "-", "label": r"$d_{\mathrm{pass}}$ (positive control)"},
    "so":   {"color": "#7f7f7f", "marker": "s", "linestyle": "--", "label": r"$d_{\mathrm{so}}$ (negative control)"},
}


def load_steering_results(direction: str, layer: int) -> dict:
    """Load steering results from JSON file."""
    path = OUTPUT_DIR / f"steering_pass_game_d_{direction}_layer{layer}.json"
    with open(path) as f:
        return json.load(f)


def extract_curve(data: dict) -> tuple:
    """Extract alphas and p_answer from results."""
    alphas = [r["alpha"] for r in data["results"]]
    p_answers = [r["p_answer"] for r in data["results"]]
    return np.array(alphas), np.array(p_answers)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax_idx, layer in enumerate(LAYERS):
        ax = axes[ax_idx]

        # Get baseline from any file (they're all the same at α=0)
        baseline_data = load_steering_results("conf", layer)
        baseline = baseline_data["baseline"]["p_answer"]

        # Plot each direction
        for direction in DIRECTIONS:
            data = load_steering_results(direction, layer)
            alphas, p_answers = extract_curve(data)

            style = STYLES[direction]
            ax.plot(
                alphas, p_answers,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=7,
                label=style["label"],
            )

        # Reference lines
        ax.axhline(y=baseline, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"Baseline ({baseline:.1%})")
        ax.axhline(y=0.5, color="lightgray", linestyle="--", linewidth=1, alpha=0.5)

        # Formatting
        ax.set_xlabel(r"Steering strength ($\alpha$)")
        ax.set_title(f"Layer {layer}")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

        # Only add y-label to first panel
        if ax_idx == 0:
            ax.set_ylabel("P(Answer)")

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    # Main title
    fig.suptitle(
        "Activation Steering Causally Controls Pass/Answer Behavior",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "steering_curves_pass_game.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")

    # Also save PDF for paper
    pdf_path = OUTPUT_DIR / "steering_curves_pass_game.pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()
