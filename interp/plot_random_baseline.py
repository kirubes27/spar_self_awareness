#!/usr/bin/env python3
"""
plot_random_baseline.py

Visualize the random baseline comparison: d_conf vs random directions.
Creates a bar chart showing how much stronger d_conf is vs random vectors.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style settings for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_results():
    """Load random baseline stats."""
    stats_path = Path("interp/outputs/random_baseline_stats.json")
    with open(stats_path) as f:
        return json.load(f)


def plot_swing_comparison(results: dict, output_path: Path):
    """Create bar chart comparing d_conf swing vs random swings."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    tasks = ["pass_game", "simplemc_self", "simplemc_other"]
    task_labels = ["Pass Game\n(P(Answer))", "SimpleMC Self\n(Confidence)", "SimpleMC Other\n(Confidence)"]

    colors = {
        'd_conf': '#2ecc71',  # Green
        'd_so': '#3498db',    # Blue
        'd_pass': '#9b59b6',  # Purple
        'random': '#95a5a6',  # Gray
    }

    for idx, (task, label) in enumerate(zip(tasks, task_labels)):
        ax = axes[idx]
        data = results[task]

        # Get special direction swings
        specials = data["specials"]
        random_stats = data["random"]

        # Prepare bars
        labels = []
        swings = []
        bar_colors = []

        for name in ["d_conf", "d_so", "d_pass"]:
            if name in specials:
                labels.append(name)
                swings.append(abs(specials[name]["swing"]))
                bar_colors.append(colors[name])

        # Add random max
        labels.append("Random\n(max)")
        swings.append(max(abs(random_stats["swing_min"]), abs(random_stats["swing_max"])))
        bar_colors.append(colors["random"])

        # Create bars
        x = np.arange(len(labels))
        bars = ax.bar(x, swings, color=bar_colors, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, swing in zip(bars, swings):
            height = bar.get_height()
            ax.annotate(f'{swing:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Absolute Swing (α: -3 → +3)" if idx == 0 else "")
        ax.set_title(label, fontweight='bold')
        ax.set_ylim(0, max(swings) * 1.2)

        # Add ratio annotation
        if "d_conf" in specials and random_stats["n"] > 0:
            d_conf_swing = abs(specials["d_conf"]["swing"])
            random_max = max(abs(random_stats["swing_min"]), abs(random_stats["swing_max"]))
            if random_max > 0:
                ratio = d_conf_swing / random_max
                ax.text(0.95, 0.95, f'd_conf is\n{ratio:.1f}× larger',
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.8))

    plt.suptitle("d_conf Steering Effect vs Random Directions (Layer 35)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def plot_random_distribution(results: dict, output_path: Path):
    """Create distribution plot showing all random swings vs d_conf."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Load individual random swings for pass_game
    output_dir = Path("interp/outputs")
    random_swings = []

    for i in range(10):
        fpath = output_dir / f"steering_pass_game_d_random_{i}_layer35.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            alpha_to_p = {r["alpha"]: r["p_answer"] for r in data["results"]}
            swing = alpha_to_p[3.0] - alpha_to_p[-3.0]
            random_swings.append(swing)

    # Get d_conf swing
    d_conf_swing = results["pass_game"]["specials"]["d_conf"]["swing"]

    # Plot random swings as scatter
    x_random = np.arange(len(random_swings))
    ax.scatter(x_random, random_swings, s=100, c='#95a5a6', edgecolors='black',
               linewidths=1, label='Random directions', zorder=3)

    # Add horizontal line for d_conf
    ax.axhline(y=d_conf_swing, color='#2ecc71', linewidth=3, linestyle='--',
               label=f'd_conf = {d_conf_swing:.3f}', zorder=2)

    # Add zero line
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)

    # Fill region between d_conf and random
    ax.fill_between([-0.5, 9.5], 0, d_conf_swing, alpha=0.1, color='#2ecc71')

    ax.set_xlabel("Random Direction Index", fontweight='bold')
    ax.set_ylabel("Swing: P(Answer) at α=+3 minus α=-3", fontweight='bold')
    ax.set_title("Pass Game: d_conf vs 10 Random Directions", fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 9.5)
    ax.set_xticks(range(10))
    ax.legend(loc='upper right')

    # Add annotation
    ax.annotate(f'd_conf is {abs(d_conf_swing)/max(abs(s) for s in random_swings):.1f}× stronger\nthan strongest random',
               xy=(4.5, d_conf_swing * 0.6), fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    results = load_results()

    output_dir = Path("interp/outputs")

    # Main comparison plot
    plot_swing_comparison(results, output_dir / "random_baseline_comparison.png")

    # Distribution plot
    plot_random_distribution(results, output_dir / "random_vs_dconf_distribution.png")

    print("\n✓ All visualizations generated!")
