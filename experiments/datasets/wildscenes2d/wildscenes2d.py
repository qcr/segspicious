"""WildScenes2d wrapper that satisfies the SegmentationDataset protocol.

Wraps the 2D portion of the WildScenes dataset (IJRR 2024) — semantic
segmentation in natural/unstructured environments (Australian forests).

Uses the official benchmark class set (15 classes) which merges rare classes
(asphalt → other-terrain, pole → other-object) and ignores unlabelled and
vehicle pixels.  Class ordering is alphabetical, matching the official
mmsegmentation config.

Reference:
    Vidanapathirana et al., "WildScenes: A Benchmark for 2D and 3D Semantic
    Segmentation in Large-scale Natural Environments", IJRR 2024.
    https://csiro-robotics.github.io/WildScenes/
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from segspicious import SegmentationSample, Split

# ---------------------------------------------------------------------------
# WildScenes raw classes (19 total, indices 0–18 in label PNGs)
#
# raw_id  raw_name         benchmark_name     train_id
# ------  --------         --------------     --------
#  0      unlabelled       (ignored)          255
#  1      asphalt          other-terrain        8
#  2      dirt             dirt                 1
#  3      mud              mud                  6
#  4      water            water               14
#  5      gravel           gravel               4
#  6      other-terrain    other-terrain        8
#  7      tree-trunk       tree-trunk          13
#  8      tree-foliage     tree-foliage        12
#  9      bush             bush                 0
# 10      fence            fence                2
# 11      structure        structure           11
# 12      pole             other-object         7
# 13      vehicle          (ignored)          255
# 14      rock             rock                 9
# 15      log              log                  5
# 16      other-object     other-object         7
# 17      sky              sky                 10
# 18      grass            grass                3
# ---------------------------------------------------------------------------

_NUM_CLASSES = 15
_IGNORE_INDEX = 255

# Benchmark classes in alphabetical order (matching official mmseg config).
# fmt: off
_CLASS_NAMES = (
    "bush",           #  0
    "dirt",           #  1
    "fence",          #  2
    "grass",          #  3
    "gravel",         #  4
    "log",            #  5
    "mud",            #  6
    "other-object",   #  7  (includes pole)
    "other-terrain",  #  8  (includes asphalt)
    "rock",           #  9
    "sky",            # 10
    "structure",      # 11
    "tree-foliage",   # 12
    "tree-trunk",     # 13
    "water",          # 14
)

# Look-up table: raw label index → benchmark train ID.
# Unmapped classes (unlabelled=0, vehicle=13) default to 255 (ignore).
_REMAP_LUT = np.array([
    255,  #  0  unlabelled  → ignore
      8,  #  1  asphalt     → other-terrain
      1,  #  2  dirt
      6,  #  3  mud
     14,  #  4  water
      4,  #  5  gravel
      8,  #  6  other-terrain
     13,  #  7  tree-trunk
     12,  #  8  tree-foliage
      0,  #  9  bush
      2,  # 10  fence
     11,  # 11  structure
      7,  # 12  pole        → other-object
    255,  # 13  vehicle     → ignore
      9,  # 14  rock
      5,  # 15  log
      7,  # 16  other-object
     10,  # 17  sky
      3,  # 18  grass
], dtype=np.uint8)
# fmt: on

_SPLITS_DIR = Path(__file__).parent / "splits"


def _load_split_csv(csv_path: Path) -> list[tuple[str, str]]:
    """Parse a WildScenes opt2d split CSV into (image_relpath, label_relpath).

    CSV columns: ``id, im_path, label_path``.  Paths in the CSV are relative
    to the dataset root and start with ``WildScenes2d/...``.
    """
    pairs: list[tuple[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["im_path"], row["label_path"]))
    return pairs


class WildScenes2dDataset:
    """Wraps the WildScenes 2D dataset, returning :class:`SegmentationSample`.

    Labels are remapped to the official 15-class benchmark set.  Raw classes
    ``unlabelled`` and ``vehicle`` become ``ignore_index`` (255).  ``asphalt``
    is merged into ``other-terrain`` and ``pole`` into ``other-object``.

    Parameters
    ----------
    root:
        Path to the WildScenes root directory — the parent of
        ``WildScenes2d/`` (e.g. ``/data/WildScenes``).
    split:
        Which dataset split to load.
    """

    def __init__(self, root: str | Path, split: Split = Split.TRAIN) -> None:
        root = Path(root)

        ws2d_dir = root / "WildScenes2d"
        if not ws2d_dir.is_dir():
            raise FileNotFoundError(
                f"Expected WildScenes2d directory at {ws2d_dir}\n"
                f"Download WildScenes from "
                f"https://data.csiro.au/collection/csiro:61541 and extract "
                f"it so that {root}/WildScenes2d/ exists."
            )

        csv_path = _SPLITS_DIR / f"{split.value}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Split CSV not found at {csv_path}. "
                f"Expected bundled split files in {_SPLITS_DIR}."
            )

        self._root = root
        self._pairs = _load_split_csv(csv_path)

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
        return len(self._pairs)

    def __getitem__(self, index: int) -> SegmentationSample:
        img_rel, lbl_rel = self._pairs[index]

        img_path = self._root / img_rel
        lbl_path = self._root / lbl_rel

        # Load RGB image.
        with Image.open(img_path) as img_pil:
            image = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)  # (H, W, 3)

        # Load label map — grayscale PNG, pixel values are raw class indices.
        with Image.open(lbl_path) as lbl_pil:
            raw_labels = np.asarray(lbl_pil, dtype=np.int64)  # (H, W)

        # Remap raw indices → benchmark train IDs.
        clipped = np.clip(raw_labels, 0, len(_REMAP_LUT) - 1)
        labels = _REMAP_LUT[clipped].astype(np.int64)

        return SegmentationSample(image=image, labels=labels, ood_mask=None)
