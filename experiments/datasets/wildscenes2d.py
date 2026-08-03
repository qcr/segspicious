"""WildScenes 2D semantic segmentation dataset."""

from __future__ import annotations

import csv
from enum import Enum
from importlib import resources
from pathlib import Path

import torch
from torch import Tensor
from torchvision.io import decode_image, ImageReadMode

from segspicious.datasets import SegmentationDataset, Split

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


def _load_split_csv(split: Split) -> list[tuple[str, str]]:
    """Load a bundled split CSV and return (image_path, label_path) pairs.

    Paths in the CSV are prefixed with ``WildScenes2d/<sequence>/…``.
    We strip the leading ``WildScenes2d/`` so paths are relative to
    the dataset root (which *is* the ``WildScenes2d/`` directory).
    """
    csv_name = f"{split.value}.csv"
    ref = resources.files(__package__) / "wildscenes2d_splits" / csv_name
    with resources.as_file(ref) as csv_path:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            pairs: list[tuple[str, str]] = []
            for row in reader:
                im = row["im_path"].removeprefix("WildScenes2d/")
                lbl = row["label_path"].removeprefix("WildScenes2d/")
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

        pairs = _load_split_csv(split)

        if sequence is not Sequence.ALL:
            prefix = sequence.value + "/"
            pairs = [(im, lbl) for im, lbl in pairs if im.startswith(prefix)]

        if not pairs:
            raise ValueError(
                f"No samples found for split={split.value!r}, "
                f"sequence={sequence.value!r}."
            )

        self._pairs = pairs

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
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        im_rel, lbl_rel = self._pairs[index]

        image = decode_image(
            str(self._root / im_rel), mode=ImageReadMode.RGB
        ).float() / 255.0

        raw_labels = decode_image(
            str(self._root / lbl_rel), mode=ImageReadMode.GRAY
        ).squeeze(0)
        labels = _LABEL_REMAP[raw_labels.long()]

        return image, labels
