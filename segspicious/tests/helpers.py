"""Test helpers — synthetic datasets and utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from segspicious.datasets.base import SegmentationDataset


class SyntheticDataset(SegmentationDataset):
    """Small in-memory dataset for unit tests.

    Generates random images in [0, 1] and random labels in
    [0, num_classes). Fully deterministic given a seed.
    """

    def __init__(
        self,
        num_samples: int = 10,
        num_classes: int = 5,
        height: int = 8,
        width: int = 8,
        ignore_index: int = 255,
        class_names: tuple[str, ...] | None = None,
        seed: int = 0,
    ) -> None:
        self._num_classes = num_classes
        self._all_class_names = class_names or tuple(
            f"class_{i}" for i in range(num_classes)
        )
        self._ignore_index = ignore_index

        gen = torch.Generator().manual_seed(seed)
        self._images = torch.rand(num_samples, 3, height, width, generator=gen)
        self._labels = torch.randint(
            0, num_classes, (num_samples, height, width), generator=gen
        )

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def all_class_names(self) -> tuple[str, ...]:
        return self._all_class_names

    @property
    def ignore_index(self) -> int:
        return self._ignore_index

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._images[index], self._labels[index]
