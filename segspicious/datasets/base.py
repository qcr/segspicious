"""Base class for segmentation datasets."""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum

from torch import Tensor
from torch.utils.data import Dataset


class Split(Enum):
    """Standard dataset splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


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
    def name(self) -> str:
        """Human-readable identifier for this dataset.

        Used for checkpoint path derivation.  Modifier functions append
        suffixes so the name captures the full transformation chain.
        """
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of in-distribution classes."""
        ...

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        """Names of in-distribution classes. Length must equal ``num_classes``."""
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

    # -- optional overrides ------------------------------------------------

    def get_labels(self, index: int) -> Tensor:
        """Return only the label tensor for sample *index*.

        The default implementation calls ``__getitem__`` and discards the
        image.  Disk-backed datasets should override this to avoid
        decoding the image when only labels are needed (e.g. during
        filter scans).
        """
        _, labels = self[index]
        return labels

    def get_classes_present(self, index: int) -> frozenset[int]:
        """Return the set of class indices present in sample *index*.

        The default implementation calls :meth:`get_labels` and computes
        ``torch.unique``.  Datasets can override this with a much
        cheaper implementation — e.g. a cached lookup table built once
        from label metadata — to avoid decoding label images entirely
        during filter scans.
        """
        labels = self.get_labels(index)
        return frozenset(labels.unique().tolist())

    # -- derived (free for all subclasses) ---------------------------------

    @property
    def all_class_names(self) -> tuple[str, ...]:
        """Names of all classes (ID then OoD). Length >= num_classes.

        ``all_class_names[:num_classes]`` are in-distribution,
        ``all_class_names[num_classes:]`` are out-of-distribution.
        ``all_class_names[label]`` gives the name for any non-ignore label.

        Defaults to ``class_names`` (no OoD classes). Overridden by
        modifiers that introduce OoD classes (e.g. ``mark_as_ood``).
        """
        return self.class_names

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
