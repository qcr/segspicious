"""Segmentation quality metrics."""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from segspicious.outputs import SegmentationOutput


class IoUResult(NamedTuple):
    """Result from :class:`IoU`."""

    per_class_iou: list[float]
    """Per-class IoU, one value per class."""
    mean_iou: float
    """Mean IoU across classes."""


class AccuracyResult(NamedTuple):
    """Result from :class:`PixelAccuracy`."""

    pixel_accuracy: float
    """Fraction of correctly classified pixels."""
    mean_class_accuracy: float
    """Per-class accuracy averaged across classes."""


class IoU:
    """Intersection over Union.

    Computes per-class IoU and mean IoU. Wraps
    ``torchmetrics.MulticlassJaccardIndex``. Call ``.update()``
    per batch, ``.compute()`` once at the end.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self._per_class = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=ignore_index, average="none"
        )

    def update(self, output: SegmentationOutput, labels: Tensor) -> None:
        """Accumulate a batch.

        Args:
            output: Model output with ``prediction`` field ``(B, H, W)``.
            labels: Ground truth ``(B, H, W)`` long tensor.
        """
        self._per_class.update(output.prediction, labels)

    def compute(self) -> IoUResult:
        """Compute final metrics."""
        per_class = self._per_class.compute()
        return IoUResult(
            per_class_iou=per_class.tolist(),
            mean_iou=per_class.mean().item(),
        )

    def reset(self) -> None:
        self._per_class.reset()


class PixelAccuracy:
    """Pixel-level accuracy.

    Wraps ``torchmetrics.MulticlassAccuracy``. Call ``.update()``
    per batch, ``.compute()`` once at the end.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self._pixel = MulticlassAccuracy(
            num_classes=num_classes, ignore_index=ignore_index, average="micro"
        )
        self._mean_class = MulticlassAccuracy(
            num_classes=num_classes, ignore_index=ignore_index, average="macro"
        )

    def update(self, output: SegmentationOutput, labels: Tensor) -> None:
        """Accumulate a batch.

        Args:
            output: Model output with ``prediction`` field ``(B, H, W)``.
            labels: Ground truth ``(B, H, W)`` long tensor.
        """
        self._pixel.update(output.prediction, labels)
        self._mean_class.update(output.prediction, labels)

    def compute(self) -> AccuracyResult:
        """Compute final metrics."""
        return AccuracyResult(
            pixel_accuracy=self._pixel.compute().item(),
            mean_class_accuracy=self._mean_class.compute().item(),
        )

    def reset(self) -> None:
        self._pixel.reset()
        self._mean_class.reset()
