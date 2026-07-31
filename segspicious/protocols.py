"""Protocols defining the dataset and candidate interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from segspicious.types import SegmentationOutput, SegmentationSample


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


@runtime_checkable
class Candidate(Protocol):
    """Protocol for a segmentation candidate (model + inference pipeline)."""

    @property
    def name(self) -> str: ...

    def train(self, dataset: SegmentationDataset) -> None: ...

    def predict(self, image: np.ndarray) -> SegmentationOutput: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...
