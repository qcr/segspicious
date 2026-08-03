"""Base class for segmentation datasets."""

from __future__ import annotations

from abc import abstractmethod

from torch import Tensor
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Abstract base class for segmentation datasets.

    Returns (image, labels) tensor pairs where:

    - image: (C, H, W) float tensor in [0, 1], channels-first.
    - labels: (H, W) long tensor with:
        - [0, num_classes): in-distribution class labels
        - [num_classes, num_classes + num_ood_classes): OoD class labels
        - ignore_index: pixels excluded from all evaluation
    """

    # -- abstract (subclasses must implement) ------------------------------

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of in-distribution classes."""
        ...

    @property
    @abstractmethod
    def all_class_names(self) -> tuple[str, ...]:
        """Names of all classes (ID then OoD). Length >= num_classes.

        ``all_class_names[:num_classes]`` are in-distribution,
        ``all_class_names[num_classes:]`` are out-of-distribution.
        ``all_class_names[label]`` gives the name for any non-ignore label.
        """
        ...

    @property
    @abstractmethod
    def ignore_index(self) -> int:
        """Label value for pixels excluded from all evaluation."""
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: ...

    # -- derived (free for all subclasses) ---------------------------------

    @property
    def class_names(self) -> tuple[str, ...]:
        """Names of in-distribution classes."""
        return self.all_class_names[: self.num_classes]

    @property
    def ood_class_names(self) -> tuple[str, ...]:
        """Names of out-of-distribution classes."""
        return self.all_class_names[self.num_classes :]

    @property
    def num_ood_classes(self) -> int:
        """Number of out-of-distribution classes."""
        return len(self.all_class_names) - self.num_classes

    @property
    def has_ood_classes(self) -> bool:
        return self.num_ood_classes > 0
