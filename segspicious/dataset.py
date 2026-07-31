"""Dataset protocol and sample type for segmentation benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class Split(Enum):
    """Dataset split."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


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

    @property
    def height(self) -> int:
        """Spatial height of the image / label maps."""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Spatial width of the image / label maps."""
        return int(self.image.shape[1])

    @property
    def num_channels(self) -> int:
        """Number of image channels (typically 3 for RGB)."""
        return int(self.image.shape[2])


@runtime_checkable
class SegmentationDataset(Protocol):
    """Protocol for a segmentation dataset (a single split).

    Implementations must expose metadata as properties and support
    integer indexing into :class:`SegmentationSample` instances.
    """

    @property
    def num_classes(self) -> int: ...

    @property
    def class_names(self) -> tuple[str, ...]: ...

    @property
    def ignore_index(self) -> int: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> SegmentationSample: ...
