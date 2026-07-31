"""Bridges the SegmentationDataset protocol into PyTorch's data loading."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.utils.data

from segspicious.dataset import SegmentationDataset


class TorchDatasetAdapter(torch.utils.data.Dataset):
    """Wraps a :class:`SegmentationDataset` for use with a PyTorch DataLoader.

    Converts images to ``(C, H, W)`` float32 tensors in [0, 1] and labels to
    int64 tensors.  An optional *transform* receives ``(image, labels)`` tensors
    and returns the transformed pair (for augmentation, resizing, etc.).

    Parameters
    ----------
    dataset:
        Any object satisfying the :class:`SegmentationDataset` protocol.
    transform:
        Optional callable ``(image: Tensor, labels: Tensor) -> (Tensor, Tensor)``.
    """

    def __init__(
        self,
        dataset: SegmentationDataset,
        transform: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
    ) -> None:
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._dataset[index]

        # (H, W, C) uint8 → (C, H, W) float32 [0, 1]
        image = torch.from_numpy(sample.image.copy()).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(sample.labels.copy()).long()

        if self._transform is not None:
            image, labels = self._transform(image, labels)

        return image, labels
