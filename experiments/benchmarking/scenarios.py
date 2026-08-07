"""Benchmark scenarios: (train, val, test) dataset configurations to evaluate against."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from experiments.datasets.wildscenes2d import Wildscenes2dDataset
from segspicious.datasets import SegmentationDataset, Split, subset

# ---------------------------------------------------------------------------
# Dataset root paths — override via environment variables if needed.
# ---------------------------------------------------------------------------

WILDSCENES_ROOT = Path(
    os.environ.get(
        "WILDSCENES_ROOT",
        "~/datasets/WildScenes/WildScenes2d",
    )
).expanduser()


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A benchmark scenario: a named (train, val, test) dataset triple.

    Attributes:
        name: Short human-readable label used for results directories
            and figure titles.
        train: Training dataset.  Also used to derive checkpoint paths
            (via ``candidate.name``).
        val: Optional validation dataset for monitoring / early stopping.
        test: Evaluation dataset that metrics are computed against.
    """

    name: str
    train: SegmentationDataset
    val: SegmentationDataset | None
    test: SegmentationDataset


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------


def get_scenarios() -> list[Scenario]:
    """Return the list of benchmark scenarios to run."""
    return [
        # Scenario(
        #     name="wildscenes2d_subset100",
        #     train=subset(
        #         Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.TRAIN),
        #         n=100,
        #         seed=42,
        #     ),
        #     val=Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.VAL),
        #     test=Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.TEST),
        # ),
        Scenario(
            name="wildscenes2d_full",
            train=Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.TRAIN),
            val=Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.VAL),
            test=Wildscenes2dDataset(WILDSCENES_ROOT, split=Split.TEST),
        ),
    ]
