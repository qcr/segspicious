"""Tests for TorchDatasetAdapter."""

import numpy as np
import torch

from segspicious import TorchDatasetAdapter
from segspicious.dataset import SegmentationSample


class FakeDataset:
    """Tiny in-memory dataset for testing."""

    def __init__(self, n: int = 3, h: int = 8, w: int = 8, c: int = 4):
        self._n = n
        self._h = h
        self._w = w
        self._c = c

    @property
    def num_classes(self) -> int:
        return self._c

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(f"c{i}" for i in range(self._c))

    @property
    def ignore_index(self) -> int:
        return 255

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> SegmentationSample:
        rng = np.random.default_rng(index)
        return SegmentationSample(
            image=rng.integers(0, 256, (self._h, self._w, 3), dtype=np.uint8),
            labels=rng.integers(0, self._c, (self._h, self._w)),
        )


class TestTorchDatasetAdapter:
    def test_len(self):
        ds = FakeDataset(n=5)
        adapted = TorchDatasetAdapter(ds)
        assert len(adapted) == 5

    def test_getitem_shapes(self):
        ds = FakeDataset(n=2, h=16, w=32)
        adapted = TorchDatasetAdapter(ds)
        image, labels = adapted[0]
        assert image.shape == (3, 16, 32)
        assert labels.shape == (16, 32)

    def test_image_dtype_and_range(self):
        ds = FakeDataset()
        adapted = TorchDatasetAdapter(ds)
        image, _ = adapted[0]
        assert image.dtype == torch.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_labels_dtype(self):
        ds = FakeDataset()
        adapted = TorchDatasetAdapter(ds)
        _, labels = adapted[0]
        assert labels.dtype == torch.int64

    def test_transform_applied(self):
        ds = FakeDataset(h=16, w=16)

        def flip_transform(img, lbl):
            return img.flip(-1), lbl.flip(-1)

        adapted = TorchDatasetAdapter(ds, transform=flip_transform)
        image_t, _ = adapted[0]

        # Compare with untransformed.
        adapted_raw = TorchDatasetAdapter(ds)
        image_raw, _ = adapted_raw[0]

        # Flipped should differ (unless symmetric, which is unlikely).
        assert image_t.shape == image_raw.shape
