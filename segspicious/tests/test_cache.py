"""Tests for ClassIndexCache."""

import json

import pytest
import torch

from segspicious.datasets.cache import ClassIndexCache


def _make_labels(num_samples: int, num_classes: int = 3):
    """Return a list of small label tensors with known classes."""
    gen = torch.Generator().manual_seed(0)
    return [
        torch.randint(0, num_classes, (4, 4), generator=gen)
        for _ in range(num_samples)
    ]


class TestClassIndexCache:
    def test_builds_cache_from_scratch(self, tmp_path):
        labels = _make_labels(5)
        path = tmp_path / "cache" / "classes_present.json"

        cache = ClassIndexCache(
            path=path,
            index_to_key=[f"sample_{i}.png" for i in range(5)],
            get_labels=lambda i: labels[i],
        )

        for i in range(5):
            expected = frozenset(labels[i].unique().tolist())
            assert cache.get_classes_present(i) == expected

    def test_writes_json_to_disk(self, tmp_path):
        labels = _make_labels(3)
        path = tmp_path / "classes_present.json"

        ClassIndexCache(
            path=path,
            index_to_key=["a.png", "b.png", "c.png"],
            get_labels=lambda i: labels[i],
        )

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert set(data.keys()) == {"a.png", "b.png", "c.png"}

    def test_loads_existing_cache(self, tmp_path):
        """Second construction should not call get_labels at all."""
        labels = _make_labels(3)
        path = tmp_path / "classes_present.json"
        keys = ["a.png", "b.png", "c.png"]

        # First construction — builds cache.
        ClassIndexCache(path=path, index_to_key=keys, get_labels=lambda i: labels[i])

        # Second construction — should not call get_labels.
        call_count = 0

        def exploding_get_labels(i):
            nonlocal call_count
            call_count += 1
            return labels[i]

        cache = ClassIndexCache(
            path=path, index_to_key=keys, get_labels=exploding_get_labels
        )
        assert call_count == 0

        for i in range(3):
            expected = frozenset(labels[i].unique().tolist())
            assert cache.get_classes_present(i) == expected

    def test_incremental_fill(self, tmp_path):
        """Adding new keys to an existing cache only computes the new ones."""
        labels = _make_labels(5)
        path = tmp_path / "classes_present.json"

        # Build cache for first 3 samples.
        ClassIndexCache(
            path=path,
            index_to_key=["s0", "s1", "s2"],
            get_labels=lambda i: labels[i],
        )

        # Now construct with all 5 — only s3, s4 should be computed.
        calls = []

        def tracking_get_labels(i):
            calls.append(i)
            return labels[i]

        cache = ClassIndexCache(
            path=path,
            index_to_key=["s0", "s1", "s2", "s3", "s4"],
            get_labels=tracking_get_labels,
        )

        assert sorted(calls) == [3, 4]
        for i in range(5):
            expected = frozenset(labels[i].unique().tolist())
            assert cache.get_classes_present(i) == expected

    def test_shared_across_subsets(self, tmp_path):
        """Different subsets of the same data share one cache file."""
        labels = _make_labels(5)
        path = tmp_path / "classes_present.json"

        # First subset uses samples 0, 1, 2.
        ClassIndexCache(
            path=path,
            index_to_key=["s0", "s1", "s2"],
            get_labels=lambda i: labels[i],
        )

        # Second subset uses samples 2, 3, 4 (s2 already cached).
        calls = []

        def tracking_get_labels(i):
            calls.append(i)
            # indices are relative to this subset: 0→s2, 1→s3, 2→s4
            return labels[i + 2]

        cache = ClassIndexCache(
            path=path,
            index_to_key=["s2", "s3", "s4"],
            get_labels=tracking_get_labels,
        )

        # Only s3 (index 1) and s4 (index 2) should have been computed.
        assert sorted(calls) == [1, 2]
        assert cache.get_classes_present(0) == frozenset(labels[2].unique().tolist())

    def test_returns_frozenset(self, tmp_path):
        labels = _make_labels(1)
        cache = ClassIndexCache(
            path=tmp_path / "c.json",
            index_to_key=["x"],
            get_labels=lambda i: labels[i],
        )
        assert isinstance(cache.get_classes_present(0), frozenset)

    def test_handles_write_failure_gracefully(self, tmp_path):
        """If the cache path is not writable, construction still succeeds."""
        labels = _make_labels(2)
        # Point at a path inside a file (not a directory) to force OSError.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        path = blocker / "subdir" / "classes_present.json"

        cache = ClassIndexCache(
            path=path,
            index_to_key=["a", "b"],
            get_labels=lambda i: labels[i],
        )

        # Should still work in memory.
        for i in range(2):
            expected = frozenset(labels[i].unique().tolist())
            assert cache.get_classes_present(i) == expected
