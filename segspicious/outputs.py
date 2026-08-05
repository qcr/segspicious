"""Output types for segmentation and uncertainty quantification."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class SegmentationOutput:
    """Output from a segmentation model.

    Attributes:
        prediction: (B, H, W) long tensor with argmax class index per pixel.
            Values in [0, num_classes).
    """

    prediction: Tensor


@dataclass
class UncertaintyOutput(SegmentationOutput):
    """Output from an uncertainty-aware segmentation model.

    Extends SegmentationOutput with optional uncertainty fields.
    A model populates only the fields it can meaningfully provide.

    Attributes:
        class_probs: (B, C, H, W) float tensor. Probability distribution over
            classes per pixel (sums to 1 along C, non-negative).
        predictive_uncertainty: (B, H, W) float tensor. Total uncertainty
            (responds to all sources).
        aleatoric_uncertainty: (B, H, W) float tensor. Irreducible data
            ambiguity.
        epistemic_uncertainty: (B, H, W) float tensor. Reducible model
            ignorance.
        ood_score: (B, H, W) float tensor. How unlike training data each
            pixel is.
    """

    class_probs: Tensor | None = None
    predictive_uncertainty: Tensor | None = None
    aleatoric_uncertainty: Tensor | None = None
    epistemic_uncertainty: Tensor | None = None
    ood_score: Tensor | None = None
