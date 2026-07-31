"""Tests for protocol compliance."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from segspicious.candidate import Candidate, SegmentationOutput, UncertaintyOutput
from segspicious.dataset import SegmentationDataset, SegmentationSample

# ---------------------------------------------------------------------------
# Minimal concrete implementations used only for testing
# ---------------------------------------------------------------------------


class FakeDataset:
    """Minimal concrete class satisfying SegmentationDataset."""

    def __init__(
        self,
        n_samples: int = 5,
        h: int = 4,
        w: int = 4,
        n_classes: int = 3,
    ):
        self._n_samples = n_samples
        self._h = h
        self._w = w
        self._n_classes = n_classes

    @property
    def num_classes(self) -> int:
        return self._n_classes

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(f"class_{i}" for i in range(self._n_classes))

    @property
    def ignore_index(self) -> int:
        return 255

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, index: int) -> SegmentationSample:
        rng = np.random.default_rng(index)
        return SegmentationSample(
            image=rng.integers(0, 256, (self._h, self._w, 3), dtype=np.uint8),
            labels=rng.integers(0, self._n_classes, (self._h, self._w)),
        )


class FakeCandidate:
    """Minimal concrete class satisfying Candidate."""

    @property
    def name(self) -> str:
        return "fake"

    def train(self, dataset: SegmentationDataset) -> None:
        pass

    def predict(self, image: np.ndarray) -> SegmentationOutput:
        h, w = image.shape[:2]
        return SegmentationOutput(prediction=np.zeros((h, w), dtype=int))

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


class FakeUQCandidate:
    """Minimal concrete class that returns UncertaintyOutput."""

    @property
    def name(self) -> str:
        return "fake_uq"

    def train(self, dataset: SegmentationDataset) -> None:
        pass

    def predict(self, image: np.ndarray) -> UncertaintyOutput:
        h, w = image.shape[:2]
        c = 3
        return UncertaintyOutput(
            prediction=np.zeros((h, w), dtype=int),
            class_probs=np.ones((h, w, c)) / c,
            predictive_uncertainty=np.zeros((h, w)),
        )

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSegmentationDatasetProtocol:
    def test_isinstance_check(self):
        ds = FakeDataset()
        assert isinstance(ds, SegmentationDataset)

    def test_metadata(self):
        ds = FakeDataset(n_classes=5)
        assert ds.num_classes == 5
        assert len(ds.class_names) == 5
        assert ds.ignore_index == 255

    def test_len_and_getitem(self):
        ds = FakeDataset(n_samples=3, h=8, w=8, n_classes=4)
        assert len(ds) == 3
        sample = ds[0]
        assert isinstance(sample, SegmentationSample)
        assert sample.image.shape == (8, 8, 3)
        assert sample.labels.shape == (8, 8)


class TestCandidateProtocol:
    def test_isinstance_check(self):
        c = FakeCandidate()
        assert isinstance(c, Candidate)

    def test_name(self):
        c = FakeCandidate()
        assert c.name == "fake"

    def test_train_predict_lifecycle(self):
        ds = FakeDataset()
        c = FakeCandidate()
        c.train(ds)
        sample = ds[0]
        out = c.predict(sample.image)
        assert isinstance(out, SegmentationOutput)
        assert out.prediction.shape == sample.image.shape[:2]

    def test_uq_candidate_satisfies_protocol(self):
        c = FakeUQCandidate()
        assert isinstance(c, Candidate)

    def test_uq_candidate_returns_uncertainty_output(self):
        c = FakeUQCandidate()
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = c.predict(img)
        assert isinstance(out, UncertaintyOutput)
        assert isinstance(out, SegmentationOutput)
        assert out.class_probs is not None
        assert out.predictive_uncertainty is not None
        assert out.epistemic_uncertainty is None
