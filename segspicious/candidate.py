"""Candidate protocol and output types for segmentation benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from segspicious.dataset import SegmentationDataset


@dataclass
class SegmentationOutput:
    """Output from a segmentation candidate.

    Attributes:
        prediction: (H, W) int array — argmax class map.
    """

    prediction: np.ndarray

    @property
    def height(self) -> int:
        """Spatial height of the prediction map."""
        return int(self.prediction.shape[0])

    @property
    def width(self) -> int:
        """Spatial width of the prediction map."""
        return int(self.prediction.shape[1])


@dataclass
class UncertaintyOutput(SegmentationOutput):
    """Extended output that includes uncertainty estimates.

    Inherits ``prediction`` from :class:`SegmentationOutput`.  Each optional
    field defaults to ``None``; a candidate populates only the fields it can
    meaningfully provide.

    Attributes:
        class_probs: (H, W, C) probability distribution over classes.
        predictive_uncertainty: (H, W) total uncertainty.
        aleatoric_uncertainty: (H, W) data ambiguity.
        epistemic_uncertainty: (H, W) model ignorance.
        ood_score: (H, W) input unfamiliarity.
    """

    class_probs: np.ndarray | None = None
    predictive_uncertainty: np.ndarray | None = None
    aleatoric_uncertainty: np.ndarray | None = None
    epistemic_uncertainty: np.ndarray | None = None
    ood_score: np.ndarray | None = None

    @property
    def num_classes(self) -> int | None:
        """Number of classes, or ``None`` if *class_probs* is not set."""
        if self.class_probs is None:
            return None
        return int(self.class_probs.shape[2])

    def __post_init__(self) -> None:
        """Validate that all populated arrays share the same spatial dims."""
        h, w = self.prediction.shape[:2]
        spatial_fields = (
            "class_probs",
            "predictive_uncertainty",
            "aleatoric_uncertainty",
            "epistemic_uncertainty",
            "ood_score",
        )
        for name in spatial_fields:
            arr = getattr(self, name)
            if arr is not None and arr.shape[:2] != (h, w):
                raise ValueError(
                    f"{name} has spatial shape {arr.shape[:2]}, "
                    f"expected ({h}, {w}) to match prediction"
                )


@runtime_checkable
class Candidate(Protocol):
    """Protocol for a segmentation candidate (model + inference pipeline)."""

    @property
    def name(self) -> str: ...

    def train(self, dataset: SegmentationDataset) -> None: ...

    def predict(self, image: np.ndarray) -> SegmentationOutput: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...
