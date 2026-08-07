"""Dataset modifiers that compose with SegmentationDataset."""

from __future__ import annotations

import bisect
import random
from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset as _TorchConcatDataset

from segspicious.datasets.base import SegmentationDataset

# ---------------------------------------------------------------------------
# Metadata-propagating wrappers around stdlib dataset utilities
# ---------------------------------------------------------------------------


class _Subset(SegmentationDataset):
    """Random or explicit subset of a dataset. Propagates metadata."""

    def __init__(
        self,
        dataset: SegmentationDataset,
        *,
        name: str,
        indices: Sequence[int] | None = None,
        n: int | None = None,
        seed: int = 0,
    ) -> None:
        if indices is not None and n is not None:
            raise ValueError("Provide either 'indices' or 'n', not both.")
        if indices is None and n is None:
            raise ValueError("Provide either 'indices' or 'n'.")

        self._name = name
        self._dataset = dataset
        if indices is not None:
            self._indices = list(indices)
        else:
            assert n is not None  # for type checker
            if n > len(dataset):
                raise ValueError(f"n={n} exceeds dataset length {len(dataset)}.")
            rng = random.Random(seed)
            self._indices = rng.sample(range(len(dataset)), n)

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_classes(self) -> int:
        return self._dataset.num_classes

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._dataset.class_names

    @property
    def all_class_names(self) -> tuple[str, ...]:
        return self._dataset.all_class_names

    @property
    def ignore_index(self) -> int:
        return self._dataset.ignore_index

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._dataset[self._indices[index]]

    def get_labels(self, index: int) -> Tensor:
        return self._dataset.get_labels(self._indices[index])

    def get_classes_present(self, index: int) -> frozenset[int]:
        return self._dataset.get_classes_present(self._indices[index])


class _ConcatDataset(SegmentationDataset):
    """Concatenation of multiple datasets. Validates and propagates metadata."""

    def __init__(self, datasets: Sequence[SegmentationDataset], *, name: str) -> None:
        if not datasets:
            raise ValueError("Need at least one dataset.")

        first = datasets[0]
        for i, ds in enumerate(datasets[1:], 1):
            if ds.num_classes != first.num_classes:
                raise ValueError(
                    f"num_classes mismatch: dataset 0 has {first.num_classes}, dataset {i} has {ds.num_classes}."
                )
            if ds.all_class_names != first.all_class_names:
                raise ValueError(f"class_names mismatch between dataset 0 and dataset {i}.")
            if ds.ignore_index != first.ignore_index:
                raise ValueError(
                    f"ignore_index mismatch: dataset 0 has {first.ignore_index}, dataset {i} has {ds.ignore_index}."
                )

        self._name = name
        self._first = first
        self._datasets = list(datasets)
        self._concat = _TorchConcatDataset(datasets)

    def _resolve_index(self, index: int) -> tuple[SegmentationDataset, int]:
        """Map a global index to (dataset, local_index)."""
        sizes = self._concat.cumulative_sizes
        ds_idx = bisect.bisect_right(sizes, index)
        local = index if ds_idx == 0 else index - sizes[ds_idx - 1]
        return self._datasets[ds_idx], local

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_classes(self) -> int:
        return self._first.num_classes

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._first.class_names

    @property
    def all_class_names(self) -> tuple[str, ...]:
        return self._first.all_class_names

    @property
    def ignore_index(self) -> int:
        return self._first.ignore_index

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._concat[index]

    def get_labels(self, index: int) -> Tensor:
        ds, local = self._resolve_index(index)
        return ds.get_labels(local)

    def get_classes_present(self, index: int) -> frozenset[int]:
        ds, local = self._resolve_index(index)
        return ds.get_classes_present(local)


# ---------------------------------------------------------------------------
# Remapped dataset (shared by mark_as_ood, remap_classes)
# ---------------------------------------------------------------------------


class _RemappedDataset(SegmentationDataset):
    """Dataset with a label lookup table applied per sample."""

    def __init__(
        self,
        dataset: SegmentationDataset,
        remap: Tensor,
        num_classes: int,
        all_class_names: tuple[str, ...],
        ignore_index: int,
        *,
        name: str,
    ) -> None:
        self._name = name
        self._dataset = dataset
        self._remap = remap
        self._num_classes = num_classes
        self._all_class_names = all_class_names
        self._ignore_index = ignore_index

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._all_class_names[: self._num_classes]

    @property
    def all_class_names(self) -> tuple[str, ...]:
        return self._all_class_names

    @property
    def ignore_index(self) -> int:
        return self._ignore_index

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, labels = self._dataset[index]
        return image, self._remap[labels]

    def get_labels(self, index: int) -> Tensor:
        return self._remap[self._dataset.get_labels(index)]

    def get_classes_present(self, index: int) -> frozenset[int]:
        inner = self._dataset.get_classes_present(index)
        return frozenset(self._remap[c].item() for c in inner)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def subset(
    dataset: SegmentationDataset,
    *,
    indices: Sequence[int] | None = None,
    n: int | None = None,
    seed: int = 0,
) -> SegmentationDataset:
    """Create a subset of a dataset.

    Provide either ``indices`` (explicit) or ``n`` + ``seed`` (random
    subsample of fixed size).
    """
    if indices is not None:
        joined = "+".join(str(i) for i in indices)
        suffix = f"-subset(indices={joined})"
    else:
        suffix = f"-subset(n={n},seed={seed})"
    name = f"{dataset.name}{suffix}"
    return _Subset(dataset, name=name, indices=indices, n=n, seed=seed)


def concat_datasets(
    datasets: Sequence[SegmentationDataset],
) -> SegmentationDataset:
    """Concatenate multiple datasets.

    All datasets must share the same ``num_classes``, ``class_names``,
    and ``ignore_index``.
    """
    name = "+".join(ds.name for ds in datasets)
    return _ConcatDataset(datasets, name=name)


def filter_samples(
    dataset: SegmentationDataset,
    predicate: Callable[[Tensor, Tensor], bool],
    *,
    label: str,
) -> SegmentationDataset:
    """Keep only samples where ``predicate(image, labels)`` is True.

    Scans the full dataset at construction time to compute kept indices.

    Args:
        dataset: Source dataset.
        predicate: A function ``(image, labels) -> bool``.
        label: Human-readable description of what the predicate does.
            Used in the dataset name suffix.

    .. note::

       This loads every sample (image + labels) to evaluate the
       predicate.  If your predicate only inspects labels, use
       :func:`filter_by_labels` instead.  If it only needs the *set* of
       classes present, prefer :func:`select_classes` /
       :func:`hold_out_classes` which use
       :meth:`~SegmentationDataset.get_classes_present` and can be
       nearly free on datasets that cache class metadata.
    """
    indices = [i for i in range(len(dataset)) if predicate(*dataset[i])]
    name = f"{dataset.name}-filter({label})"
    return _Subset(dataset, name=name, indices=indices)


def filter_by_labels(
    dataset: SegmentationDataset,
    predicate: Callable[[Tensor], bool],
    *,
    label: str,
) -> SegmentationDataset:
    """Keep only samples where ``predicate(labels)`` is True.

    Like :func:`filter_samples` but the predicate receives only the
    label tensor.  Uses :meth:`~SegmentationDataset.get_labels` so
    disk-backed datasets can skip image decoding during the scan.

    Args:
        dataset: Source dataset.
        predicate: A function ``(labels) -> bool``.
        label: Human-readable description of what the predicate does.
            Used in the dataset name suffix.
    """
    indices = [i for i in range(len(dataset)) if predicate(dataset.get_labels(i))]
    name = f"{dataset.name}-filter({label})"
    return _Subset(dataset, name=name, indices=indices)


def select_classes(
    dataset: SegmentationDataset,
    classes: Sequence[str] | Sequence[int],
) -> SegmentationDataset:
    """Return only samples containing at least one pixel of the specified classes.

    Pure filter — no label remapping, metadata unchanged.

    Args:
        dataset: Source dataset.
        classes: Class names or indices. A sample is kept if it contains
            at least one pixel of any listed class.

    Returns:
        A filtered dataset (subset of the original).
    """
    if not classes:
        raise ValueError("classes must not be empty.")

    if isinstance(classes[0], str):
        try:
            class_indices = frozenset(dataset.class_names.index(str(c)) for c in classes)
        except ValueError as e:
            raise ValueError(f"Class name not found in dataset: {e}. Available: {dataset.class_names}") from e
    else:
        class_indices = frozenset(int(c) for c in classes)
        for idx in class_indices:
            if idx < 0 or idx >= dataset.num_classes:
                raise ValueError(f"Class index {idx} out of range [0, {dataset.num_classes}).")

    indices = [i for i in range(len(dataset)) if class_indices & dataset.get_classes_present(i)]
    # Build name suffix with sorted class names
    if isinstance(classes[0], str):
        sorted_names = sorted(str(c) for c in classes)
    else:
        sorted_names = sorted(dataset.class_names[int(c)] for c in classes)
    suffix = "-select(" + "+".join(sorted_names) + ")"
    name = f"{dataset.name}{suffix}"
    return _Subset(dataset, name=name, indices=indices)


def mark_as_ood(
    dataset: SegmentationDataset,
    classes: Sequence[str] | Sequence[int],
) -> SegmentationDataset:
    """Mark classes as out-of-distribution.

    Keeps all samples. The specified classes are remapped to OoD labels
    (``>= num_classes``). Remaining in-distribution classes are remapped
    to contiguous ``[0, n)``. Existing OoD labels and ``ignore_index``
    are preserved.

    For training (remove samples entirely), use :func:`hold_out_classes`.

    Args:
        dataset: Source dataset.
        classes: Class names or indices to mark as OoD.

    Returns:
        A new dataset with updated ``num_classes``, ``class_names``,
        and remapped labels.
    """
    if not classes:
        raise ValueError("classes must not be empty.")

    # Normalise to indices -------------------------------------------------
    if isinstance(classes[0], str):
        ood_names = set(str(c) for c in classes)
        for name in ood_names:
            if name not in dataset.class_names:
                raise ValueError(f"Class name {name!r} not found in dataset. Available: {dataset.class_names}")
        ood_indices = frozenset(dataset.class_names.index(n) for n in ood_names)
    else:
        ood_indices = frozenset(int(c) for c in classes)
        for idx in ood_indices:
            if idx < 0 or idx >= dataset.num_classes:
                raise ValueError(f"Class index {idx} out of range [0, {dataset.num_classes}).")

    # Derive kept classes (preserving original order) ----------------------
    keep_indices = [i for i in range(dataset.num_classes) if i not in ood_indices]
    n = len(keep_indices)
    keep_names = tuple(dataset.all_class_names[i] for i in keep_indices)
    newly_ood_names = tuple(dataset.all_class_names[i] for i in sorted(ood_indices))
    all_names = keep_names + newly_ood_names + dataset.ood_class_names

    # Build lookup table: remap[old_label] → new_label ---------------------
    #
    # Identity-initialised so labels above num_classes (existing OoD and
    # ignore_index) pass through unchanged — they are already >= n and
    # remain valid under the new, smaller num_classes.
    remap = torch.arange(dataset.ignore_index + 1, dtype=torch.long)

    for new_idx, old_idx in enumerate(keep_indices):
        remap[old_idx] = new_idx

    ood_counter = n
    for old_idx in sorted(ood_indices):
        remap[old_idx] = ood_counter
        ood_counter += 1

    # Build name suffix with sorted class names
    if isinstance(classes[0], str):
        sorted_names = sorted(str(c) for c in classes)
    else:
        sorted_names = sorted(dataset.class_names[int(c)] for c in classes)
    suffix = "-mark_ood(" + "+".join(sorted_names) + ")"
    name = f"{dataset.name}{suffix}"

    return _RemappedDataset(dataset, remap, n, all_names, dataset.ignore_index, name=name)


def hold_out_classes(
    dataset: SegmentationDataset,
    classes: Sequence[str] | Sequence[int],
) -> SegmentationDataset:
    """Remove samples containing any pixel of the specified classes.

    Pure filter — no label remapping, metadata unchanged. The inverse
    of :func:`select_classes`.

    Args:
        dataset: Source dataset.
        classes: Class names or indices. A sample is removed if it
            contains at least one pixel of any listed class.

    Returns:
        A filtered dataset (subset of the original).
    """
    if not classes:
        raise ValueError("classes must not be empty.")

    if isinstance(classes[0], str):
        try:
            class_indices = frozenset(dataset.class_names.index(str(c)) for c in classes)
        except ValueError as e:
            raise ValueError(f"Class name not found in dataset: {e}. Available: {dataset.class_names}") from e
    else:
        class_indices = frozenset(int(c) for c in classes)
        for idx in class_indices:
            if idx < 0 or idx >= dataset.num_classes:
                raise ValueError(f"Class index {idx} out of range [0, {dataset.num_classes}).")

    indices = [i for i in range(len(dataset)) if not (class_indices & dataset.get_classes_present(i))]
    # Build name suffix with sorted class names
    if isinstance(classes[0], str):
        sorted_names = sorted(str(c) for c in classes)
    else:
        sorted_names = sorted(dataset.class_names[int(c)] for c in classes)
    suffix = "-hold_out(" + "+".join(sorted_names) + ")"
    name = f"{dataset.name}{suffix}"
    return _Subset(dataset, name=name, indices=indices)


def hold_out_ood(
    dataset: SegmentationDataset,
) -> SegmentationDataset:
    """Remove samples containing any out-of-distribution pixel.

    Pure filter — no label remapping, metadata unchanged. Uses the
    current ``num_classes`` boundary to identify OoD pixels.

    Typical workflow: :func:`mark_as_ood` to set up OoD classes, then
    ``hold_out_ood`` to filter for training.

    Returns:
        A filtered dataset with no OoD pixels.
    """
    num_cls = dataset.num_classes
    ignore = dataset.ignore_index
    indices = [
        i for i in range(len(dataset)) if not any(c >= num_cls and c != ignore for c in dataset.get_classes_present(i))
    ]
    name = f"{dataset.name}-hold_out_ood"
    return _Subset(dataset, name=name, indices=indices)


def coverage_subset(
    dataset: SegmentationDataset,
    n: int,
) -> SegmentationDataset:
    """Greedy set-cover subset that maximises class coverage.

    Iteratively picks the image that covers the most not-yet-represented
    classes.  Ties are broken by the total number of classes present
    (more is better).  Uses only :meth:`get_classes_present` — fast,
    no pixel data loaded.

    Deterministic (no randomness).

    Args:
        dataset: Source dataset.
        n: Number of images to select.

    Returns:
        A subset of the dataset with at most *n* images.
    """
    if n > len(dataset):
        raise ValueError(f"n={n} exceeds dataset length {len(dataset)}.")

    indices = _coverage_phase(dataset, n)
    name = f"{dataset.name}-coverage_subset(n={n})"
    return _Subset(dataset, name=name, indices=indices)


def balanced_subset(
    dataset: SegmentationDataset,
    n: int,
) -> SegmentationDataset:
    """Class-balanced subset: coverage phase + pixel-count balancing.

    Phase 1 (coverage): identical to :func:`coverage_subset` — greedy
    set-cover using :meth:`get_classes_present`.

    Phase 2 (balance): fills the remaining budget by picking images that
    contribute the most pixels to the least-represented class.  Uses
    :meth:`get_labels` to count pixels per class — only called on
    candidates not yet selected, only during this phase.

    Deterministic.

    Args:
        dataset: Source dataset.
        n: Number of images to select.

    Returns:
        A subset of the dataset with at most *n* images.
    """
    if n > len(dataset):
        raise ValueError(f"n={n} exceeds dataset length {len(dataset)}.")

    num_classes = dataset.num_classes
    ignore_index = dataset.ignore_index

    # Phase 1: coverage
    selected = _coverage_phase(dataset, n)
    selected_set = set(selected)

    # Phase 2: pixel-count balancing
    # Build pixel profiles for selected images
    profiles: dict[int, list[int]] = {}  # idx -> per-class pixel counts
    for idx in selected:
        profiles[idx] = _pixel_profile(dataset, idx, num_classes, ignore_index)

    while len(selected) < n:
        # Current pixel totals per class across selected images
        current_totals = [0] * num_classes
        for idx in selected:
            for c in range(num_classes):
                current_totals[c] += profiles[idx][c]

        best_idx = -1
        best_score = -1.0

        for i in range(len(dataset)):
            if i in selected_set:
                continue
            # Lazily compute profile for this candidate
            if i not in profiles:
                profiles[i] = _pixel_profile(dataset, i, num_classes, ignore_index)

            # Score: prioritise uncovered classes heavily, then weight by
            # inverse of current representation
            score = 0.0
            for c in range(num_classes):
                px = profiles[i][c]
                if px == 0:
                    continue
                if current_totals[c] == 0:
                    score += px * 1000
                else:
                    score += px / current_totals[c]

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            break

        selected.append(best_idx)
        selected_set.add(best_idx)

    name = f"{dataset.name}-balanced_subset(n={n})"
    return _Subset(dataset, name=name, indices=selected)


def _coverage_phase(
    dataset: SegmentationDataset,
    n: int,
) -> list[int]:
    """Greedy set-cover: pick images to maximise class coverage."""
    num_classes = dataset.num_classes
    selected: list[int] = []
    selected_set: set[int] = set()
    covered_classes: set[int] = set()

    # Pre-fetch classes present for all images
    all_classes = [dataset.get_classes_present(i) for i in range(len(dataset))]

    while len(selected) < n and len(covered_classes) < num_classes:
        best_idx = -1
        best_new_count = -1
        best_total_count = -1

        for i in range(len(dataset)):
            if i in selected_set:
                continue
            present = all_classes[i]
            # Only consider valid in-distribution classes
            id_present = frozenset(c for c in present if c < num_classes)
            new_classes = id_present - covered_classes
            new_count = len(new_classes)
            total_count = len(id_present)

            if new_count > best_new_count or (new_count == best_new_count and total_count > best_total_count):
                best_idx = i
                best_new_count = new_count
                best_total_count = total_count

        if best_idx < 0:
            break

        selected.append(best_idx)
        selected_set.add(best_idx)
        id_present = frozenset(c for c in all_classes[best_idx] if c < num_classes)
        covered_classes.update(id_present)

    # If we still have budget but all classes are covered, fill greedily
    # by total class count (most diverse images first)
    if len(selected) < n:
        remaining = [i for i in range(len(dataset)) if i not in selected_set]
        # Sort by number of ID classes present, descending
        remaining.sort(
            key=lambda i: len(frozenset(c for c in all_classes[i] if c < num_classes)),
            reverse=True,
        )
        for i in remaining:
            if len(selected) >= n:
                break
            selected.append(i)
            selected_set.add(i)

    return selected


def _pixel_profile(
    dataset: SegmentationDataset,
    index: int,
    num_classes: int,
    ignore_index: int,
) -> list[int]:
    """Count pixels per in-distribution class for one sample."""
    labels = dataset.get_labels(index)
    counts = [0] * num_classes
    for c in range(num_classes):
        counts[c] = int((labels == c).sum().item())
    return counts


def remap_classes(
    dataset: SegmentationDataset,
    mapping: dict[str, str],
) -> SegmentationDataset:
    """Apply an explicit class remapping by name.

    Each key is an existing class name, each value is the new class name.
    Multiple old classes mapping to the same new name are merged.
    Unmapped ID classes are preserved (still in-distribution) with
    indices shifted after the mapped classes. Existing OoD classes are
    also preserved. No class changes its ID/OoD/ignore status — to do
    that, use :func:`mark_as_ood`.

    New class indices are assigned in order of first appearance in the
    mapping values.

    Args:
        dataset: Source dataset.
        mapping: ``{old_name: new_name}`` for each class to keep or merge.

    Returns:
        A new dataset with updated ``num_classes``, ``class_names``,
        and remapped labels.
    """
    if not mapping:
        raise ValueError("mapping must not be empty.")

    # Resolve names to indices and collect new class names -----------------
    new_name_to_idx: dict[str, int] = {}
    new_names: list[str] = []
    index_mapping: dict[int, int] = {}

    for old_name, new_name in mapping.items():
        try:
            old_idx = dataset.class_names.index(old_name)
        except ValueError:
            raise ValueError(
                f"Class name {old_name!r} not found in dataset. Available: {dataset.class_names}"
            ) from None
        if new_name not in new_name_to_idx:
            new_name_to_idx[new_name] = len(new_names)
            new_names.append(new_name)
        index_mapping[old_idx] = new_name_to_idx[new_name]

    # Build lookup table ---------------------------------------------------
    #
    # Mapped ID classes get new indices [0, n).  Unmapped ID classes
    # stay ID at [n, n + num_unmapped).  Existing OoD classes are
    # shifted to be contiguous after all ID classes.
    n = len(new_names)
    ignore = dataset.ignore_index
    remap = torch.full((ignore + 1,), fill_value=ignore, dtype=torch.long)

    for old_idx, new_idx in index_mapping.items():
        remap[old_idx] = new_idx

    mapped_set = set(index_mapping)
    unmapped_names: list[str] = []
    unmapped_counter = n
    for old_idx in range(dataset.num_classes):
        if old_idx not in mapped_set:
            remap[old_idx] = unmapped_counter
            unmapped_counter += 1
            unmapped_names.append(dataset.all_class_names[old_idx])

    for i in range(dataset.num_ood_classes):
        remap[dataset.num_classes + i] = unmapped_counter + i

    new_num_classes = n + len(unmapped_names)
    all_names = tuple(new_names) + tuple(unmapped_names) + dataset.ood_class_names

    # Build name suffix with mapping entries sorted by key
    sorted_entries = sorted(mapping.items(), key=lambda kv: kv[0])
    parts = [f"{old}={new}" for old, new in sorted_entries]
    suffix = "-remap(" + "+".join(parts) + ")"
    name = f"{dataset.name}{suffix}"

    return _RemappedDataset(dataset, remap, new_num_classes, all_names, ignore, name=name)


__all__ = [
    "balanced_subset",
    "concat_datasets",
    "coverage_subset",
    "filter_by_labels",
    "filter_samples",
    "hold_out_classes",
    "hold_out_ood",
    "mark_as_ood",
    "remap_classes",
    "select_classes",
    "subset",
]
