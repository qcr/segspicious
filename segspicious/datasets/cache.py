"""Per-sample class index cache for segmentation datasets."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from torch import Tensor

log = logging.getLogger(__name__)


class ClassIndexCache:
    """Cache mapping sample index → set of class indices present.

    On construction, loads an existing JSON cache from *path* (if it
    exists), computes any missing entries by calling *get_labels*, and
    writes the updated cache back to disk.  After construction,
    :meth:`get_classes_present` is a pure in-memory lookup.

    Args:
        path: File path for the JSON cache.
        index_to_key: Maps each sample index to a stable string key
            used in the JSON file (e.g. a label file's relative path).
            The key must be stable across different splits/subsets of
            the same underlying data so that cache entries are reusable.
        get_labels: Callable that takes a sample index and returns the
            label tensor for that sample.  Only called for samples not
            already in the cache.
    """

    def __init__(
        self,
        path: Path,
        index_to_key: list[str],
        get_labels: Callable[[int], Tensor],
    ) -> None:
        self._index_to_key = index_to_key
        self._cache = self._load_and_fill(path, index_to_key, get_labels)

    @staticmethod
    def _load_and_fill(
        path: Path,
        index_to_key: list[str],
        get_labels: Callable[[int], Tensor],
    ) -> dict[str, list[int]]:
        """Load existing cache, compute missing entries, write back."""
        cache: dict[str, list[int]] = {}
        if path.exists():
            with open(path) as f:
                cache = json.load(f)

        missing = [
            (i, key) for i, key in enumerate(index_to_key) if key not in cache
        ]

        if missing:
            log.info(
                "Building class index for %d samples (one-time cost)…",
                len(missing),
            )
            for i, key in missing:
                classes = get_labels(i).unique().tolist()
                cache[key] = sorted(classes)

            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    json.dump(cache, f)
                log.info("Class index cached to %s", path)
            except OSError:
                log.warning(
                    "Could not write class index cache to %s; "
                    "will rebuild next time.",
                    path,
                )

        return cache

    def get_classes_present(self, index: int) -> frozenset[int]:
        """Return the set of class indices present in sample *index*."""
        key = self._index_to_key[index]
        return frozenset(self._cache[key])
