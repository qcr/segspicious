"""Tests for the WildScenes2dDataset wrapper."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from segspicious import SegmentationDataset, SegmentationSample, Split

from .wildscenes2d import (
    WildScenes2dDataset,
    _CLASS_NAMES,
    _IGNORE_INDEX,
    _NUM_CLASSES,
    _REMAP_LUT,
    _SPLITS_DIR,
)


# ---------------------------------------------------------------------------
# Helpers to build a tiny fake dataset on disk
# ---------------------------------------------------------------------------

def _make_fake_dataset(
    tmp_path: Path,
    n_samples: int = 3,
    height: int = 8,
    width: int = 12,
    raw_classes: list[int] | None = None,
) -> Path:
    """Create a minimal WildScenes-style directory with fake PNGs + CSV."""
    root = tmp_path / "WildScenes"
    seq_dir = root / "WildScenes2d" / "K-01"
    img_dir = seq_dir / "image"
    lbl_dir = seq_dir / "indexLabel"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    if raw_classes is None:
        raw_classes = [0, 2, 7, 8, 9, 13, 17, 18]

    rng = np.random.default_rng(42)
    rows: list[dict[str, str]] = []
    for i in range(n_samples):
        name = f"sample_{i:04d}"
        # RGB image
        img_arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(img_dir / f"{name}.png")

        # Label map — pick from raw_classes
        lbl_arr = rng.choice(raw_classes, size=(height, width)).astype(np.uint8)
        Image.fromarray(lbl_arr, mode="L").save(lbl_dir / f"{name}.png")

        rows.append(
            {
                "id": name,
                "im_path": f"WildScenes2d/K-01/image/{name}.png",
                "label_path": f"WildScenes2d/K-01/indexLabel/{name}.png",
            }
        )

    # Write CSV for all three splits (just reuse the same samples).
    for split_name in ("train", "val", "test"):
        csv_path = _SPLITS_DIR / f"{split_name}.csv"
        # We can't write to the bundled splits, so we monkeypatch _SPLITS_DIR
        # in the test instead.  Here we write to the tmp dir.
        csv_out = tmp_path / "splits" / f"{split_name}.csv"
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "im_path", "label_path"])
            writer.writeheader()
            writer.writerows(rows)

    return root


# ---------------------------------------------------------------------------
# Unit tests — use fake data (no external dataset required)
# ---------------------------------------------------------------------------


class TestRemapLUT:
    """Verify the label remapping table is correct."""

    def test_lut_length(self):
        assert len(_REMAP_LUT) == 19

    def test_unlabelled_ignored(self):
        assert _REMAP_LUT[0] == 255

    def test_vehicle_ignored(self):
        assert _REMAP_LUT[13] == 255

    def test_asphalt_merged_into_other_terrain(self):
        assert _REMAP_LUT[1] == _REMAP_LUT[6]  # both → other-terrain

    def test_pole_merged_into_other_object(self):
        assert _REMAP_LUT[12] == _REMAP_LUT[16]  # both → other-object

    def test_all_valid_ids_in_range(self):
        for v in _REMAP_LUT:
            assert v < _NUM_CLASSES or v == _IGNORE_INDEX

    def test_all_classes_covered(self):
        """Every train ID in [0, num_classes) is produced by at least one raw class."""
        covered = set(_REMAP_LUT[_REMAP_LUT != 255])
        assert covered == set(range(_NUM_CLASSES))


class TestClassNames:
    def test_count(self):
        assert len(_CLASS_NAMES) == _NUM_CLASSES

    def test_sorted(self):
        assert _CLASS_NAMES == tuple(sorted(_CLASS_NAMES))


class TestWildScenes2dDatasetFake:
    """Test the dataset wrapper with synthetic data."""

    @pytest.fixture()
    def fake_ds(self, tmp_path, monkeypatch):
        root = _make_fake_dataset(tmp_path, n_samples=5, height=8, width=12)
        # Point the module-level _SPLITS_DIR at our tmp CSV dir.
        monkeypatch.setattr(
            f"{WildScenes2dDataset.__module__}._SPLITS_DIR",
            tmp_path / "splits",
        )
        return WildScenes2dDataset(root, split=Split.TRAIN)

    def test_protocol_compliance(self, fake_ds):
        assert isinstance(fake_ds, SegmentationDataset)

    def test_len(self, fake_ds):
        assert len(fake_ds) == 5

    def test_num_classes(self, fake_ds):
        assert fake_ds.num_classes == 15

    def test_class_names(self, fake_ds):
        assert fake_ds.class_names == _CLASS_NAMES

    def test_ignore_index(self, fake_ds):
        assert fake_ds.ignore_index == 255

    def test_getitem_returns_sample(self, fake_ds):
        sample = fake_ds[0]
        assert isinstance(sample, SegmentationSample)

    def test_image_shape_and_dtype(self, fake_ds):
        sample = fake_ds[0]
        assert sample.image.shape == (8, 12, 3)
        assert sample.image.dtype == np.uint8

    def test_labels_shape_and_dtype(self, fake_ds):
        sample = fake_ds[0]
        assert sample.labels.shape == (8, 12)
        assert sample.labels.dtype == np.int64

    def test_labels_in_valid_range(self, fake_ds):
        for i in range(len(fake_ds)):
            labels = fake_ds[i].labels
            unique = np.unique(labels)
            for v in unique:
                assert v < _NUM_CLASSES or v == _IGNORE_INDEX

    def test_ood_mask_is_none(self, fake_ds):
        assert fake_ds[0].ood_mask is None


class TestWildScenes2dDatasetErrors:
    """Test error handling."""

    def test_missing_root(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="WildScenes2d"):
            WildScenes2dDataset(tmp_path / "nonexistent")

    def test_missing_ws2d_subdir(self, tmp_path):
        (tmp_path / "exists").mkdir()
        with pytest.raises(FileNotFoundError, match="WildScenes2d"):
            WildScenes2dDataset(tmp_path / "exists")


# ---------------------------------------------------------------------------
# Integration tests — require real data on disk (skipped in CI)
# ---------------------------------------------------------------------------

_WILDSCENES_ROOT = os.environ.get(
    "WILDSCENES_ROOT", "/home/alistair/datasets/WildScenes"
)
_has_data = Path(_WILDSCENES_ROOT).joinpath("WildScenes2d").is_dir()


@pytest.mark.skipif(not _has_data, reason="WildScenes data not found")
class TestWildScenes2dIntegration:
    """Integration tests against the real WildScenes dataset."""

    def test_train_loads(self):
        ds = WildScenes2dDataset(_WILDSCENES_ROOT, split=Split.TRAIN)
        assert len(ds) > 5000
        sample = ds[0]
        assert sample.image.shape[2] == 3
        assert sample.image.dtype == np.uint8

    def test_val_loads(self):
        ds = WildScenes2dDataset(_WILDSCENES_ROOT, split=Split.VAL)
        assert len(ds) > 100

    def test_test_loads(self):
        ds = WildScenes2dDataset(_WILDSCENES_ROOT, split=Split.TEST)
        assert len(ds) > 1000

    def test_labels_remapped(self):
        ds = WildScenes2dDataset(_WILDSCENES_ROOT, split=Split.TRAIN)
        sample = ds[0]
        unique = np.unique(sample.labels)
        for v in unique:
            assert v < _NUM_CLASSES or v == _IGNORE_INDEX, (
                f"unexpected label value {v}"
            )
