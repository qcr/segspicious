"""WildScenes 2D semantic segmentation dataset."""

from __future__ import annotations

import csv
from enum import Enum
from importlib import resources
from pathlib import Path

import torch
from torch import Tensor
from torchvision.io import ImageReadMode, decode_image

from segspicious.datasets import ClassIndexCache, SegmentationDataset, Split

# -- Class definition ------------------------------------------------------

# 19 raw classes (index 0–18). Index 0 is "unlabelled" → ignore_index.
# Indices 1–18 are the 18 actual classes, remapped to 0–17.
# Names follow the WildScenes mmseg reference implementation.
_RAW_CLASS_NAMES: tuple[str, ...] = (
    "unlabelled",
    "asphalt",
    "dirt",
    "mud",
    "water",
    "gravel",
    "other-terrain",
    "tree-trunk",
    "tree-foliage",
    "bush",
    "fence",
    "structure",
    "pole",
    "vehicle",
    "rock",
    "log",
    "other-object",
    "sky",
    "grass",
)

_NUM_CLASSES = len(_RAW_CLASS_NAMES) - 1  # 18 (excluding "unlabelled")
_CLASS_NAMES = _RAW_CLASS_NAMES[1:]  # ("asphalt", "dirt", …, "grass")
_IGNORE_INDEX = 255

# Lookup table: raw label → dataset label.
# 0 (unlabelled) → 255, 1–18 → 0–17.
_LABEL_REMAP = torch.zeros(256, dtype=torch.long)
_LABEL_REMAP[0] = _IGNORE_INDEX
for _i in range(1, 19):
    _LABEL_REMAP[_i] = _i - 1


class Sequence(Enum):
    """WildScenes 2D recording sequences."""

    K01 = "K-01"
    K03 = "K-03"
    V01 = "V-01"
    V02 = "V-02"
    V03 = "V-03"
    ALL = "all"


def _load_split_csv(split: Split) -> list[tuple[Path, Path]]:
    """Load a bundled split CSV and return (image_path, label_path) pairs.

    Paths in the CSV are prefixed with ``WildScenes2d/<sequence>/…``.
    We strip the leading ``WildScenes2d/`` so paths are relative to
    the dataset root (which *is* the ``WildScenes2d/`` directory).
    """
    csv_name = f"{split.value}.csv"
    ref = resources.files(__package__) / "splits" / csv_name
    with resources.as_file(ref) as csv_path, open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        pairs: list[tuple[Path, Path]] = []
        for row in reader:
            im = Path(row["im_path"].removeprefix("WildScenes2d/"))
            lbl = Path(row["label_path"].removeprefix("WildScenes2d/"))
            pairs.append((im, lbl))
    return pairs


class Wildscenes2dDataset(SegmentationDataset):
    """WildScenes 2D semantic segmentation dataset.

    Wraps the WildScenes2d image/indexLabel data with bundled
    train/val/test split definitions.

    Args:
        root: Path to the ``WildScenes2d/`` directory containing
            sequence subdirectories (``K-01/``, ``K-03/``, etc.).
        split: Dataset split.
        sequence: Restrict to a single recording sequence, or use
            ``Sequence.ALL`` (default) for all sequences in the split.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split = Split.TRAIN,
        sequence: Sequence = Sequence.ALL,
    ) -> None:
        self._root = Path(root)

        sample_paths = _load_split_csv(split)

        if sequence is not Sequence.ALL:
            seq_dir = Path(sequence.value)
            sample_paths = [
                (im_path, lbl_path)
                for im_path, lbl_path in sample_paths
                if im_path.is_relative_to(seq_dir)
            ]

        if not sample_paths:
            raise ValueError(
                f"No samples found for split={split.value!r}, "
                f"sequence={sequence.value!r}."
            )

        self._sample_paths = sample_paths
        self._class_cache = ClassIndexCache(
            path=self._root / ".segspicious_cache" / "classes_present.json",
            index_to_key=[str(lbl_path) for _, lbl_path in sample_paths],
            get_labels=self.get_labels,
        )

    @property
    def num_classes(self) -> int:
        return _NUM_CLASSES

    @property
    def class_names(self) -> tuple[str, ...]:
        return _CLASS_NAMES

    @property
    def ignore_index(self) -> int:
        return _IGNORE_INDEX

    def __len__(self) -> int:
        return len(self._sample_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, label_path = self._sample_paths[index]

        image = (
            decode_image(str(self._root / image_path), mode=ImageReadMode.RGB).float()
            / 255.0
        )

        raw_labels = decode_image(
            str(self._root / label_path), mode=ImageReadMode.GRAY
        ).squeeze(0)
        labels = _LABEL_REMAP[raw_labels.long()]

        return image, labels

    def get_labels(self, index: int) -> Tensor:
        """Decode only the label PNG, skipping the RGB image."""
        _, label_path = self._sample_paths[index]
        raw = decode_image(
            str(self._root / label_path), mode=ImageReadMode.GRAY
        ).squeeze(0)
        return _LABEL_REMAP[raw.long()]

    def get_classes_present(self, index: int) -> frozenset[int]:
        return self._class_cache.get_classes_present(index)
