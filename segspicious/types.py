"""Core data types for segmentation UQ benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationOutput:
    """Output from a segmentation candidate.

    Attributes:
        prediction: (H, W) int array — argmax class map.
    """

    prediction: np.ndarray


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


@dataclass
class SegmentationSample:
    """A single sample from a segmentation dataset.

    Attributes:
        image: (H, W, C) uint8 RGB image.
        labels: (H, W) int array of class indices.
        ood_mask: (H, W) bool array — ``True`` = out-of-distribution.
            ``None`` when no OoD annotation is available.
    """

    image: np.ndarray
    labels: np.ndarray
    ood_mask: np.ndarray | None = None
