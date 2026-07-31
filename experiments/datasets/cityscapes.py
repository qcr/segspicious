"""Cityscapes wrapper that satisfies the SegmentationDataset protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torchvision.datasets import Cityscapes

from segspicious import SegmentationSample, Split

# ---------------------------------------------------------------------------
# Cityscapes label mapping.  34 raw classes → 19 training classes.
# Everything not listed maps to ignore_index (255).
#
# raw_id  train_id  name
# ------  --------  ----
#  0      255       unlabeled
#  1      255       ego vehicle
#  2      255       rectification border
#  3      255       out of roi
#  4      255       static
#  5      255       dynamic
#  6      255       ground
#  7        0       road
#  8        1       sidewalk
#  9      255       parking
# 10      255       rail track
# 11        2       building
# 12        3       wall
# 13        4       fence
# 14      255       guard rail
# 15      255       bridge
# 16      255       tunnel
# 17        5       pole
# 18      255       polegroup
# 19        6       traffic light
# 20        7       traffic sign
# 21        8       vegetation
# 22        9       terrain
# 23       10       sky
# 24       11       person
# 25       12       rider
# 26       13       car
# 27       14       truck
# 28       15       bus
# 29      255       caravan
# 30      255       trailer
# 31       16       train
# 32       17       motorcycle
# 33       18       bicycle
# ---------------------------------------------------------------------------

_NUM_CLASSES = 19
_IGNORE_INDEX = 255

# fmt: off
_CLASS_NAMES = (
    "road",           #  0
    "sidewalk",       #  1
    "building",       #  2
    "wall",           #  3
    "fence",          #  4
    "pole",           #  5
    "traffic light",  #  6
    "traffic sign",   #  7
    "vegetation",     #  8
    "terrain",        #  9
    "sky",            # 10
    "person",         # 11
    "rider",          # 12
    "car",            # 13
    "truck",          # 14
    "bus",            # 15
    "train",          # 16
    "motorcycle",     # 17
    "bicycle",        # 18
)

_REMAP_LUT = np.array([
    255, 255, 255, 255, 255, 255, 255,   0,   1, 255,
    255,   2,   3,   4, 255, 255, 255,   5, 255,   6,
      7,   8,   9,  10,  11,  12,  13,  14,  15, 255,
    255,  16,  17,  18
], dtype=np.uint8)
# fmt: on


class CityscapesDataset:
    """Wraps torchvision Cityscapes, returning :class:`SegmentationSample`.

    Always uses ``mode="fine"`` and ``target_type="semantic"``.  Raw label
    ids (0-33) are remapped to the standard 19 training classes; all other
    pixels become ``ignore_index`` (255).

    Parameters
    ----------
    root:
        Path to the Cityscapes root directory (the one containing
        ``leftImg8bit/`` and ``gtFine/``).
    split:
        Which dataset split to load.
    """

    def __init__(self, root: str | Path, split: Split = Split.TRAIN) -> None:
        root = Path(root)

        images_dir = root / "leftImg8bit" / split.value
        targets_dir = root / "gtFine" / split.value

        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Expected images directory at {images_dir}\n"
                f"Download Cityscapes from https://www.cityscapes-dataset.com/ "
                f"and extract it so that {root}/leftImg8bit/{split.value}/ exists."
            )
        if not targets_dir.is_dir():
            raise FileNotFoundError(
                f"Expected targets directory at {targets_dir}\n"
                f"Download Cityscapes from https://www.cityscapes-dataset.com/ "
                f"and extract it so that {root}/gtFine/{split.value}/ exists."
            )

        self._dataset = Cityscapes(
            root=str(root),
            split=split.value,
            mode="fine",
            target_type="semantic",
        )

    # -- SegmentationDataset protocol --------------------------------------

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
        return len(self._dataset)

    def __getitem__(self, index: int) -> SegmentationSample:
        image_pil, target_pil = self._dataset[index]

        image = np.asarray(image_pil, dtype=np.uint8)  # (H, W, 3)
        raw_labels = np.asarray(target_pil, dtype=np.int64)  # (H, W)

        # Remap raw ids → train ids.  Clip to LUT range so out-of-range ids
        # (e.g. 255 used by some label files) become ignore_index.
        clipped = np.clip(raw_labels, 0, len(_REMAP_LUT) - 1)
        labels = _REMAP_LUT[clipped].astype(np.int64)

        return SegmentationSample(image=image, labels=labels, ood_mask=None)
