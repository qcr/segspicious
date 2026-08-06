"""Evaluate all candidates and serialise metrics to disk.

For each (model × scenario) pair, loads the trained candidate,
runs the evaluation loop over ``scenario.test``, and writes a
JSON file with the full metric output.

Metrics are computed **incrementally**: if a result file already
exists on disk, only metrics whose key is absent from the file are
computed.  This makes reruns idempotent — add new entries to
``METRIC_REGISTRY``, rerun, and only the new ones are evaluated.

Each metric handles field availability internally.  If the model's
output doesn't populate a field the metric needs (e.g. ``ood_score``),
the metric's ``update()`` is a no-op and ``compute()`` returns
``None``.  The ``None`` is written to the results file (serialised
as JSON ``null``) so that subsequent runs know the metric was
already attempted for this candidate.

Results layout::

    results/{scenario.name}/{candidate.name}.json

Result key semantics:

- **absent** — not yet attempted, will be computed on next run.
- **null** — attempted, not applicable (model doesn't provide the
  required output field).
- **{...}** — successfully computed metric results.

Usage::

    python -m experiments.benchmarking.eval
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import segspicious
from experiments.benchmarking.models import get_models
from experiments.benchmarking.scenarios import get_scenarios
from segspicious import load
from segspicious.metrics import IoU, PixelAccuracy

if TYPE_CHECKING:
    from segspicious.datasets import SegmentationDataset
    from segspicious.outputs import SegmentationOutput

REPO_ID = "alistair-english/segspicious-checkpoints"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDef:
    """An entry in the metric registry.

    Attributes:
        key: Unique identifier used as the key under ``"metrics"``
            in the results JSON.
        factory: Creates a fresh metric instance given the test
            dataset.  The metric must expose ``update(output, labels)``
            and ``compute() -> NamedTuple | None``.
    """

    key: str
    factory: Callable[[SegmentationDataset], object]


METRIC_REGISTRY: list[MetricDef] = [
    MetricDef(
        key="iou",
        factory=lambda ds: IoU(num_classes=ds.num_classes, ignore_index=ds.ignore_index),
    ),
    MetricDef(
        key="accuracy",
        factory=lambda ds: PixelAccuracy(num_classes=ds.num_classes, ignore_index=ds.ignore_index),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_path(scenario_name: str, candidate_name: str) -> Path:
    """Derive the JSON output path for a (scenario, candidate) pair."""
    return RESULTS_DIR / scenario_name / f"{candidate_name}.json"


def _load_existing(path: Path) -> dict:
    """Load an existing results file, or return an empty dict."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_results(path: Path, data: dict) -> None:
    """Atomically-ish write *data* as JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _serialise_result(result: object) -> dict | None:
    """Convert a metric's ``compute()`` return value to JSON-friendly form.

    - ``None`` → ``None``  (serialised as JSON ``null``)
    - A ``NamedTuple`` → its ``._asdict()`` dict representation
    - A plain ``dict`` → passed through unchanged
    """
    if result is None:
        return None
    if hasattr(result, "_asdict"):
        return result._asdict()  # type: ignore[union-attr]
    if isinstance(result, dict):
        return result
    raise TypeError(f"Cannot serialise metric result of type {type(result)}")


def _output_to_cpu(output: SegmentationOutput) -> SegmentationOutput:
    """Move every ``Tensor`` field on *output* to CPU (in-place)."""
    for field in dataclasses.fields(output):
        val = getattr(output, field.name)
        if isinstance(val, Tensor):
            setattr(output, field.name, val.cpu())
    return output


def _format_result_summary(result: dict) -> str:
    """One-line summary of scalar values in a result dict."""
    parts = []
    for k, v in result.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
    return ", ".join(parts) if parts else str(result)


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
            candidate_name = f"{model.name}/{scenario.train.name}"
            print(f"  [{model.name}] → candidate: {candidate_name}")

            try:
                candidate = load(model, scenario.train)
            except FileNotFoundError as e:
                print("    ✗ No checkpoint found — run train.py first.")
                print(f"      {e}")
                print()
                continue

            # ----- Determine which metrics still need computing ----------
            out_path = _result_path(scenario.name, candidate.name)
            existing = _load_existing(out_path)
            computed_keys = set(existing.get("metrics", {}))

            pending = [m for m in METRIC_REGISTRY if m.key not in computed_keys]

            if not pending:
                n = len(METRIC_REGISTRY)
                print(f"    ✓ All {n} metric(s) already computed — skipping")
                print()
                continue

            cached = len(computed_keys & {m.key for m in METRIC_REGISTRY})
            print(f"    {len(pending)} metric(s) to compute" + (f" ({cached} cached)" if cached else ""))

            # ----- Instantiate pending metrics ---------------------------
            test_ds = scenario.test
            active: list[tuple[str, object]] = [(m.key, m.factory(test_ds)) for m in pending]

            # ----- Single inference pass ---------------------------------
            loader = DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=(DEVICE.type == "cuda"),
            )

            for images, labels in tqdm(loader, desc=f"    eval {test_ds.name}"):
                images = images.to(DEVICE)
                output = candidate.predict(images)
                _output_to_cpu(output)

                for _key, metric in active:
                    metric.update(output, labels)  # type: ignore[attr-defined]

            # ----- Compute, serialise, merge -----------------------------
            metrics = existing.setdefault("metrics", {})
            for key, metric in active:
                raw = metric.compute()  # type: ignore[attr-defined]
                result = _serialise_result(raw)
                metrics[key] = result

                if result is None:
                    print(f"    {key}: n/a (required field not provided)")
                else:
                    print(f"    {key}: {_format_result_summary(result)}")

            # Update metadata (may have changed if scenarios evolved)
            existing.update(
                {
                    "candidate_name": candidate.name,
                    "scenario_name": scenario.name,
                    "model_name": model.name,
                    "train_dataset_name": scenario.train.name,
                    "test_dataset_name": scenario.test.name,
                    "class_names": list(test_ds.class_names),
                }
            )

            _save_results(out_path, existing)
            print(f"    → {out_path}")
            print()

    print("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
