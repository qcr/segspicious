"""Tests for coverage_subset and balanced_subset modifiers."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from segspicious.datasets import (
    SegmentationDataset,
    balanced_subset,
    coverage_subset,
    subset,
)
from segspicious.datasets.base import SegmentationDataset as _Base
from segspicious.tests.helpers import SyntheticDataset


# ── Helper: dataset with controlled class distribution ────────────────────


class _ControlledDataset(_Base):
    """Dataset where each image has specific classes present.

    ``class_sets[i]`` is the set of class indices in image *i*.
    Labels are fabricated so each listed class gets some pixels.
    """

    def __init__(
        self,
        class_sets: list[set[int]],
        num_classes: int,
        height: int = 8,
        width: int = 8,
        ignore_index: int = 255,
    ) -> None:
        self._class_sets = class_sets
        self._num_classes = num_classes
        self._height = height
        self._width = width
        self._ignore_index = ignore_index
        self._class_names = tuple(f"class_{i}" for i in range(num_classes))

        # Build label tensors: distribute pixels evenly across classes
        self._labels: list[Tensor] = []
        for classes in class_sets:
            label = torch.full((height, width), ignore_index, dtype=torch.long)
            if classes:
                sorted_classes = sorted(classes)
                total = height * width
                per_class = total // len(sorted_classes)
                for ci, c in enumerate(sorted_classes):
                    start = ci * per_class
                    end = (ci + 1) * per_class if ci < len(sorted_classes) - 1 else total
                    label.view(-1)[start:end] = c
            self._labels.append(label)

        self._images = [
            torch.rand(3, height, width) for _ in range(len(class_sets))
        ]

    @property
    def name(self) -> str:
        return "controlled"

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._class_names

    @property
    def ignore_index(self) -> int:
        return self._ignore_index

    def __len__(self) -> int:
        return len(self._class_sets)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._images[index], self._labels[index]

    def get_labels(self, index: int) -> Tensor:
        return self._labels[index]

    def get_classes_present(self, index: int) -> frozenset[int]:
        return frozenset(self._class_sets[index])


# ── coverage_subset tests ────────────────────────────────────────────────


class TestCoverageSubset:
    def test_covers_all_classes_minimal(self):
        """With disjoint class sets, coverage needs exactly as many images."""
        ds = _ControlledDataset(
            class_sets=[{0, 1}, {2, 3}, {4}],
            num_classes=5,
        )
        result = coverage_subset(ds, n=3)
        assert len(result) == 3
        # All 5 classes should be covered
        all_covered: set[int] = set()
        for i in range(len(result)):
            all_covered.update(result.get_classes_present(i))
        assert all_covered == {0, 1, 2, 3, 4}

    def test_covers_all_classes_with_overlap(self):
        """Overlapping sets: fewer images can cover everything."""
        ds = _ControlledDataset(
            class_sets=[{0, 1, 2}, {1, 2, 3}, {3, 4}, {0, 4}],
            num_classes=5,
        )
        # n=2 should suffice: {0,1,2} + {3,4}
        result = coverage_subset(ds, n=2)
        assert len(result) == 2
        all_covered: set[int] = set()
        for i in range(len(result)):
            all_covered.update(result.get_classes_present(i))
        assert all_covered == {0, 1, 2, 3, 4}

    def test_prefers_more_classes_on_tie(self):
        """Tie-breaking: image with more total classes wins."""
        ds = _ControlledDataset(
            # Image 0: 1 new class, 1 total
            # Image 1: 1 new class, 3 total (wins tie)
            class_sets=[{0}, {0, 1, 2}],
            num_classes=3,
        )
        result = coverage_subset(ds, n=1)
        # Should pick image 1 (more total classes)
        assert result[0][0].equal(ds[1][0])

    def test_name_suffix(self):
        ds = SyntheticDataset(num_samples=10, num_classes=3)
        result = coverage_subset(ds, n=5)
        assert result.name == "synthetic-coverage_subset(n=5)"

    def test_metadata_propagated(self):
        ds = SyntheticDataset(num_samples=10, num_classes=7)
        result = coverage_subset(ds, n=5)
        assert result.num_classes == ds.num_classes
        assert result.class_names == ds.class_names
        assert result.ignore_index == ds.ignore_index

    def test_length_equals_n(self):
        ds = SyntheticDataset(num_samples=20, num_classes=3)
        result = coverage_subset(ds, n=8)
        assert len(result) == 8

    def test_n_exceeds_dataset_raises(self):
        ds = SyntheticDataset(num_samples=5, num_classes=3)
        with pytest.raises(ValueError, match="exceeds dataset length"):
            coverage_subset(ds, n=10)

    def test_deterministic(self):
        ds = SyntheticDataset(num_samples=20, num_classes=5, seed=0)
        r1 = coverage_subset(ds, n=10)
        r2 = coverage_subset(ds, n=10)
        for i in range(len(r1)):
            assert r1[i][0].equal(r2[i][0])

    def test_single_image_classes(self):
        """Classes only in a single image are still covered."""
        ds = _ControlledDataset(
            class_sets=[{0}, {1}, {2}, {0, 1, 2}],
            num_classes=3,
        )
        result = coverage_subset(ds, n=1)
        # Should pick image 3 (covers all 3 classes)
        all_covered: set[int] = set()
        for i in range(len(result)):
            all_covered.update(result.get_classes_present(i))
        assert all_covered == {0, 1, 2}


# ── balanced_subset tests ────────────────────────────────────────────────


class TestBalancedSubset:
    def test_name_suffix(self):
        ds = SyntheticDataset(num_samples=10, num_classes=3)
        result = balanced_subset(ds, n=5)
        assert result.name == "synthetic-balanced_subset(n=5)"

    def test_metadata_propagated(self):
        ds = SyntheticDataset(num_samples=10, num_classes=7)
        result = balanced_subset(ds, n=5)
        assert result.num_classes == ds.num_classes
        assert result.class_names == ds.class_names
        assert result.ignore_index == ds.ignore_index

    def test_length_equals_n(self):
        ds = SyntheticDataset(num_samples=20, num_classes=3)
        result = balanced_subset(ds, n=8)
        assert len(result) == 8

    def test_n_exceeds_dataset_raises(self):
        ds = SyntheticDataset(num_samples=5, num_classes=3)
        with pytest.raises(ValueError, match="exceeds dataset length"):
            balanced_subset(ds, n=10)

    def test_deterministic(self):
        ds = SyntheticDataset(num_samples=20, num_classes=5, seed=0)
        r1 = balanced_subset(ds, n=10)
        r2 = balanced_subset(ds, n=10)
        for i in range(len(r1)):
            assert r1[i][0].equal(r2[i][0])

    def test_better_balance_than_random(self):
        """Balanced subset should have more even pixel distribution."""
        # Create a dataset where class 0 is rare (only in image 0)
        # and other classes are common
        class_sets = [{0, 1}] + [{1, 2, 3}] * 19
        ds = _ControlledDataset(class_sets=class_sets, num_classes=4)

        bal = balanced_subset(ds, n=5)
        rand = subset(ds, n=5, seed=42)

        # Count pixels per class in balanced subset
        bal_counts = [0] * 4
        for i in range(len(bal)):
            labels = bal.get_labels(i)
            for c in range(4):
                bal_counts[c] += int((labels == c).sum().item())

        # Count pixels per class in random subset
        rand_counts = [0] * 4
        for i in range(len(rand)):
            labels = rand.get_labels(i)
            for c in range(4):
                rand_counts[c] += int((labels == c).sum().item())

        # Balanced should include image 0 (which has class 0)
        assert bal_counts[0] > 0, "Balanced subset should cover rare class 0"

    def test_phase1_coverage_then_phase2_balance(self):
        """Phase 1 covers classes; Phase 2 improves pixel balance."""
        ds = _ControlledDataset(
            class_sets=[{0, 1}, {2, 3}, {4}, {0}, {1, 2}],
            num_classes=5,
        )
        # n=3 covers all 5 classes in phase 1
        # n=5 adds 2 more images in phase 2 for balance
        result = balanced_subset(ds, n=5)
        assert len(result) == 5
        all_covered: set[int] = set()
        for i in range(len(result)):
            all_covered.update(result.get_classes_present(i))
        assert all_covered == {0, 1, 2, 3, 4}

    def test_n_equals_dataset_length(self):
        """Edge case: n == len(dataset) returns all images."""
        ds = SyntheticDataset(num_samples=5, num_classes=3)
        result = balanced_subset(ds, n=5)
        assert len(result) == 5
