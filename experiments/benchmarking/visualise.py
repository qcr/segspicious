"""Load serialised metrics and produce comparison figures.

Reads JSON results from ``results/`` and generates plots under
``figures/``, subfolderd by scenario name.

Figures per scenario:
    - miou_comparison.png       — mIoU bar chart across candidates
    - accuracy_comparison.png   — pixel accuracy & mean class accuracy
    - per_class_iou.png         — per-class IoU grouped by candidate

Usage:
    python -m experiments.benchmarking.visualise
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _MODULE_DIR / "results"
FIGURES_DIR = _MODULE_DIR / "figures"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_scenario_results(scenario_dir: Path) -> list[dict]:
    """Load all result JSONs under a scenario directory."""
    results = []
    for json_path in sorted(scenario_dir.rglob("*.json")):
        with open(json_path) as f:
            results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_miou_comparison(results: list[dict], scenario_name: str, out_dir: Path) -> None:
    """Bar chart of mIoU per candidate."""
    names = [r["candidate_name"] for r in results]
    values = [r["metrics"]["iou"]["mean_iou"] for r in results]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
    bars = ax.bar(range(len(names)), values, color="#4C72B0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mIoU")
    ax.set_title(f"mIoU Comparison — {scenario_name}")
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "miou_comparison.png", dpi=150)
    plt.close(fig)


def _plot_accuracy_comparison(results: list[dict], scenario_name: str, out_dir: Path) -> None:
    """Grouped bar chart of pixel accuracy and mean class accuracy."""
    names = [r["candidate_name"] for r in results]
    pixel_acc = [r["metrics"]["accuracy"]["pixel_accuracy"] for r in results]
    mean_class_acc = [r["metrics"]["accuracy"]["mean_class_accuracy"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
    bars1 = ax.bar(x - width / 2, pixel_acc, width, label="Pixel Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, mean_class_acc, width, label="Mean Class Accuracy", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy Comparison — {scenario_name}")
    ax.set_ylim(0, 1)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


def _plot_per_class_iou(results: list[dict], scenario_name: str, out_dir: Path) -> None:
    """Grouped bar chart of per-class IoU across candidates."""
    # All results in a scenario share the same class set.
    class_names = results[0]["class_names"]
    n_classes = len(class_names)
    n_candidates = len(results)

    x = np.arange(n_classes)
    width = 0.8 / max(n_candidates, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_candidates, 1)))

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.8), 6))

    for i, r in enumerate(results):
        per_class = r["metrics"]["iou"]["per_class_iou"]
        offset = (i - (n_candidates - 1) / 2) * width
        ax.bar(x + offset, per_class, width, label=r["candidate_name"], color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("IoU")
    ax.set_title(f"Per-Class IoU — {scenario_name}")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "per_class_iou.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not RESULTS_DIR.exists():
        print(f"No results directory found at {RESULTS_DIR}")
        print("Run eval.py first.")
        return

    scenario_dirs = sorted(
        d for d in RESULTS_DIR.iterdir() if d.is_dir()
    )

    if not scenario_dirs:
        print("No scenario results found.")
        return

    print(f"=== Generating figures ({len(scenario_dirs)} scenario(s)) ===\n")

    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        results = _load_scenario_results(scenario_dir)

        if not results:
            print(f"  {scenario_name}: no results, skipping.")
            continue

        print(f"  {scenario_name}: {len(results)} candidate(s)")

        out_dir = FIGURES_DIR / scenario_name
        out_dir.mkdir(parents=True, exist_ok=True)

        _plot_miou_comparison(results, scenario_name, out_dir)
        _plot_accuracy_comparison(results, scenario_name, out_dir)
        _plot_per_class_iou(results, scenario_name, out_dir)

        print(f"    → Figures saved to {out_dir}")
        print()

    print("=== Done ===")


if __name__ == "__main__":
    main()
