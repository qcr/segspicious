"""Test helpers — synthetic datasets and utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from segspicious.datasets.base import SegmentationDataset


class SyntheticDataset(SegmentationDataset):
    """Small in-memory dataset for unit tests.

    Generates random images in [0, 1] and random labels in
    [0, num_classes). Fully deterministic given a seed.

    Set ``class_names`` to define the classes (the canonical approach,
    matching how :class:`SegmentationDataset` is subclassed in
    practice).  ``num_classes`` is a convenience shorthand that
    auto-generates names ``("class_0", "class_1", …)``.

    OoD classes are **not** configured here — use :func:`mark_as_ood`
    on the resulting dataset instead.
    """

    def __init__(
        self,
        num_samples: int = 10,
        num_classes: int | None = None,
        height: int = 8,
        width: int = 8,
        ignore_index: int = 255,
        class_names: tuple[str, ...] | None = None,
        seed: int = 0,
    ) -> None:
        if (
            class_names is not None
            and num_classes is not None
            and len(class_names) != num_classes
        ):
            raise ValueError(
                f"num_classes={num_classes} does not match "
                f"len(class_names)={len(class_names)}. "
                "Provide one or the other (or ensure they agree)."
            )

        if class_names is not None:
            self._class_names = class_names
        else:
            n = num_classes if num_classes is not None else 5
            self._class_names = tuple(f"class_{i}" for i in range(n))

        self._ignore_index = ignore_index

        n = len(self._class_names)
        gen = torch.Generator().manual_seed(seed)
        self._images = torch.rand(num_samples, 3, height, width, generator=gen)
        self._labels = torch.randint(0, n, (num_samples, height, width), generator=gen)

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._class_names

    @property
    def ignore_index(self) -> int:
        return self._ignore_index

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._images[index], self._labels[index]
