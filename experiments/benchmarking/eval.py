"""Evaluate all candidates and serialise metrics to disk.

For each (model × scenario) pair, loads the trained candidate,
runs the evaluation loop over ``scenario.test``, and writes a
JSON file with the full metric output.

Results layout:
    results/{scenario.name}/{candidate.name}.json

Usage:
    python -m experiments.benchmarking.eval
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import segspicious
from experiments.benchmarking.models import get_models
from experiments.benchmarking.scenarios import get_scenarios
from segspicious import load
from segspicious.metrics import IoU, PixelAccuracy

REPO_ID = "alistair-english/segspicious-checkpoints"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_path(scenario_name: str, candidate_name: str) -> Path:
    """Derive the JSON output path for a (scenario, candidate) pair.

    ``candidate.name`` is ``{model.name}/{train_dataset.name}`` which
    naturally gives a two-level directory nesting under the scenario.
    """
    return RESULTS_DIR / scenario_name / f"{candidate_name}.json"


def _evaluate_candidate(candidate, scenario) -> dict:
    """Run the eval loop and return the full results dict."""
    test_ds = scenario.test

    iou = IoU(num_classes=test_ds.num_classes, ignore_index=test_ds.ignore_index)
    acc = PixelAccuracy(
        num_classes=test_ds.num_classes, ignore_index=test_ds.ignore_index
    )

    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    for images, labels in tqdm(loader, desc=f"    eval {test_ds.name}"):
        images = images.to(DEVICE)
        output = candidate.predict(images)
        output.prediction = output.prediction.cpu()
        iou.update(output, labels)
        acc.update(output, labels)

    iou_result = iou.compute()
    acc_result = acc.compute()

    return {
        "candidate_name": candidate.name,
        "scenario_name": scenario.name,
        "model_name": candidate.model.name,
        "train_dataset_name": scenario.train.name,
        "test_dataset_name": scenario.test.name,
        "class_names": list(test_ds.class_names),
        "metrics": {
            "iou": iou_result._asdict(),
            "accuracy": acc_result._asdict(),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    segspicious.configure(repo_id=REPO_ID)

    scenarios = get_scenarios()

    print(f"=== Evaluating candidates ({len(scenarios)} scenario(s)) ===")
    print(f"    device: {DEVICE}")
    print()

    for scenario in scenarios:
        print(f"Scenario: {scenario.name}")

        for model in get_models():
            print(f"  [{model.name}] → candidate: {model.name}/{scenario.train.name}")

            try:
                candidate = load(model, scenario.train)
            except FileNotFoundError as e:
                print(f"    ✗ No checkpoint found — run train.py first.")
                print(f"      {e}")
                print()
                continue

            result = _evaluate_candidate(candidate, scenario)

            # Print summary
            metrics = result["metrics"]
            print(f"    mIoU:               {metrics['iou']['mean_iou']:.4f}")
            print(f"    Pixel accuracy:     {metrics['accuracy']['pixel_accuracy']:.4f}")
            print(f"    Mean class accuracy: {metrics['accuracy']['mean_class_accuracy']:.4f}")

            # Serialise
            out_path = _result_path(scenario.name, candidate.name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"    → Saved: {out_path}")
            print()

    print("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
