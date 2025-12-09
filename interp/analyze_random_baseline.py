#!/usr/bin/env python3
"""
analyze_random_baseline.py

Analyze random direction baseline vs special directions (d_conf, d_so, d_pass).
Computes swing and max deviation metrics for comparison.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np


def compute_pass_game_stats(results_json: dict) -> dict:
    """For pass_game: compute P(answer) swing and max deviation."""
    alpha_to_p = {r["alpha"]: r["p_answer"] for r in results_json["results"]}
    p_minus3 = alpha_to_p[-3.0]
    p_plus3 = alpha_to_p[3.0]
    baseline = alpha_to_p[0.0]

    swing = p_plus3 - p_minus3
    max_dev = max(abs(p - baseline) for p in alpha_to_p.values())
    return {"swing": swing, "max_dev": max_dev}


def compute_simplemc_stats(results_json: dict) -> dict:
    """For SimpleMC: compute mean_conf swing and max deviation."""
    alpha_to_conf = {r["alpha"]: r["mean_conf"] for r in results_json["results"]}
    c_minus3 = alpha_to_conf[-3.0]
    c_plus3 = alpha_to_conf[3.0]
    baseline = alpha_to_conf[0.0]

    swing = c_plus3 - c_minus3
    max_dev = max(abs(c - baseline) for c in alpha_to_conf.values())
    return {"swing": swing, "max_dev": max_dev}


def analyze_task(task_name: str, compute_fn) -> dict:
    """Analyze random vs special directions for one task."""
    output_dir = Path("interp/outputs")

    # Find all random direction results
    pattern = output_dir / f"steering_{task_name}_d_random_*_layer35.json"
    random_files = sorted(glob.glob(str(pattern)))

    random_stats = []
    for fpath in random_files:
        with open(fpath) as fp:
            data = json.load(fp)
        stats = compute_fn(data)
        random_stats.append(stats)
        print(f"  {Path(fpath).name}: swing={stats['swing']:.4f}, max_dev={stats['max_dev']:.4f}")

    specials = {}

    # Load d_conf and d_so
    for name in ["d_conf", "d_so"]:
        fname = output_dir / f"steering_{task_name}_{name}_layer35.json"
        if fname.exists():
            with open(fname) as fp:
                data = json.load(fp)
            specials[name] = compute_fn(data)
        else:
            print(f"  Warning: {fname} not found")

    # d_pass only for pass_game
    if task_name == "pass_game":
        fname = output_dir / "steering_pass_game_d_pass_layer35.json"
        if fname.exists():
            with open(fname) as fp:
                data = json.load(fp)
            specials["d_pass"] = compute_fn(data)
        else:
            print(f"  Warning: {fname} not found")

    # Compute statistics
    swings = [s["swing"] for s in random_stats]
    max_devs = [s["max_dev"] for s in random_stats]

    summary = {
        "task": task_name,
        "random": {
            "n": len(random_stats),
            "swing_mean": float(np.mean(swings)) if swings else 0.0,
            "swing_std": float(np.std(swings)) if swings else 0.0,
            "swing_min": float(np.min(swings)) if swings else 0.0,
            "swing_max": float(np.max(swings)) if swings else 0.0,
            "max_dev_mean": float(np.mean(max_devs)) if max_devs else 0.0,
            "max_dev_max": float(np.max(max_devs)) if max_devs else 0.0,
        },
        "specials": specials,
    }
    return summary


def generate_markdown(results: dict) -> str:
    """Generate markdown summary of results."""
    md = "# Random Direction Baseline Results\n\n"
    md += f"**Generated:** {__import__('datetime').datetime.now().isoformat()[:10]}\n\n"

    for task_name, data in results.items():
        md += f"## {task_name}\n\n"

        # Special directions table
        md += "### Special Directions\n\n"
        md += "| Direction | Swing | Max Dev |\n"
        md += "|-----------|-------|---------|\n"
        for name, stats in data["specials"].items():
            md += f"| {name} | {stats['swing']:.3f} | {stats['max_dev']:.3f} |\n"

        r = data["random"]
        md += f"\n### Random Directions (N={r['n']})\n\n"
        md += (
            f"- Swing: mean={r['swing_mean']:.3f}, std={r['swing_std']:.3f}, "
            f"range=[{r['swing_min']:.3f}, {r['swing_max']:.3f}]\n"
        )
        md += (
            f"- Max deviation: mean={r['max_dev_mean']:.3f}, "
            f"max={r['max_dev_max']:.3f}\n\n"
        )

        # Comparison: how many × bigger is d_conf than random?
        if "d_conf" in data["specials"] and r["n"] > 0:
            d_conf_swing = abs(data["specials"]["d_conf"]["swing"])
            random_max = max(abs(r["swing_min"]), abs(r["swing_max"]))
            ratio = d_conf_swing / random_max if random_max > 0 else float("inf")
            md += f"**d_conf is {ratio:.1f}× larger than max random swing**\n\n"

    return md


def main():
    print("=" * 60)
    print("RANDOM BASELINE ANALYSIS")
    print("=" * 60)

    results = {}

    print("\n--- pass_game ---")
    results["pass_game"] = analyze_task("pass_game", compute_pass_game_stats)

    print("\n--- simplemc_self ---")
    results["simplemc_self"] = analyze_task("simplemc_self", compute_simplemc_stats)

    print("\n--- simplemc_other ---")
    results["simplemc_other"] = analyze_task("simplemc_other", compute_simplemc_stats)

    # Save JSON
    output_dir = Path("interp/outputs")
    out_json = output_dir / "random_baseline_stats.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved JSON to {out_json}")

    # Save markdown summary
    md = generate_markdown(results)
    out_md = output_dir / "random_baseline_summary.md"
    with open(out_md, "w") as f:
        f.write(md)
    print(f"✓ Saved markdown to {out_md}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(md)


if __name__ == "__main__":
    main()
