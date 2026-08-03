"""Dataset modifiers that compose with SegmentationDataset."""

from __future__ import annotations

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
        indices: Sequence[int] | None = None,
        n: int | None = None,
        seed: int = 0,
    ) -> None:
        if indices is not None and n is not None:
            raise ValueError("Provide either 'indices' or 'n', not both.")
        if indices is None and n is None:
            raise ValueError("Provide either 'indices' or 'n'.")

        self._dataset = dataset
        if indices is not None:
            self._indices = list(indices)
        else:
            assert n is not None  # for type checker
            if n > len(dataset):
                raise ValueError(
                    f"n={n} exceeds dataset length {len(dataset)}."
                )
            rng = random.Random(seed)
            self._indices = rng.sample(range(len(dataset)), n)

    @property
    def num_classes(self) -> int:
        return self._dataset.num_classes

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


class _ConcatDataset(SegmentationDataset):
    """Concatenation of multiple datasets. Validates and propagates metadata."""

    def __init__(self, datasets: Sequence[SegmentationDataset]) -> None:
        if not datasets:
            raise ValueError("Need at least one dataset.")

        first = datasets[0]
        for i, ds in enumerate(datasets[1:], 1):
            if ds.num_classes != first.num_classes:
                raise ValueError(
                    f"num_classes mismatch: dataset 0 has {first.num_classes}, "
                    f"dataset {i} has {ds.num_classes}."
                )
            if ds.all_class_names != first.all_class_names:
                raise ValueError(
                    f"class_names mismatch between dataset 0 and dataset {i}."
                )
            if ds.ignore_index != first.ignore_index:
                raise ValueError(
                    f"ignore_index mismatch: dataset 0 has {first.ignore_index}, "
                    f"dataset {i} has {ds.ignore_index}."
                )

        self._first = first
        self._concat = _TorchConcatDataset(datasets)

    @property
    def num_classes(self) -> int:
        return self._first.num_classes

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
    ) -> None:
        self._dataset = dataset
        self._remap = remap
        self._num_classes = num_classes
        self._all_class_names = all_class_names
        self._ignore_index = ignore_index

    @property
    def num_classes(self) -> int:
        return self._num_classes

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
    return _Subset(dataset, indices=indices, n=n, seed=seed)


def concat_datasets(
    datasets: Sequence[SegmentationDataset],
) -> SegmentationDataset:
    """Concatenate multiple datasets.

    All datasets must share the same ``num_classes``, ``class_names``,
    and ``ignore_index``.
    """
    return _ConcatDataset(datasets)


def filter_samples(
    dataset: SegmentationDataset,
    predicate: Callable[[Tensor, Tensor], bool],
) -> SegmentationDataset:
    """Keep only samples where ``predicate(image, labels)`` is True.

    Scans the full dataset at construction time to compute kept indices.
    """
    indices = [i for i in range(len(dataset)) if predicate(*dataset[i])]
    return _Subset(dataset, indices=indices)


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
            class_indices = frozenset(
                dataset.class_names.index(str(c)) for c in classes
            )
        except ValueError as e:
            raise ValueError(
                f"Class name not found in dataset: {e}. "
                f"Available: {dataset.class_names}"
            ) from e
    else:
        class_indices = frozenset(int(c) for c in classes)
        for idx in class_indices:
            if idx < 0 or idx >= dataset.num_classes:
                raise ValueError(
                    f"Class index {idx} out of range "
                    f"[0, {dataset.num_classes})."
                )

    return filter_samples(
        dataset,
        predicate=lambda _img, lbl: any(
            (lbl == idx).any() for idx in class_indices
        ),
    )


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
                raise ValueError(
                    f"Class name {name!r} not found in dataset. "
                    f"Available: {dataset.class_names}"
                )
        ood_indices = frozenset(
            dataset.class_names.index(n) for n in ood_names
        )
    else:
        ood_indices = frozenset(int(c) for c in classes)
        for idx in ood_indices:
            if idx < 0 or idx >= dataset.num_classes:
                raise ValueError(
                    f"Class index {idx} out of range "
                    f"[0, {dataset.num_classes})."
                )

    # Derive kept classes (preserving original order) ----------------------
    keep_indices = [
        i for i in range(dataset.num_classes) if i not in ood_indices
    ]
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

    return _RemappedDataset(
        dataset, remap, n, all_names, dataset.ignore_index
    )


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
            class_indices = frozenset(
                dataset.class_names.index(str(c)) for c in classes
            )
        except ValueError as e:
            raise ValueError(
                f"Class name not found in dataset: {e}. "
                f"Available: {dataset.class_names}"
            ) from e
    else:
        class_indices = frozenset(int(c) for c in classes)
        for idx in class_indices:
            if idx < 0 or idx >= dataset.num_classes:
                raise ValueError(
                    f"Class index {idx} out of range "
                    f"[0, {dataset.num_classes})."
                )

    return filter_samples(
        dataset,
        predicate=lambda _img, lbl: not any(
            (lbl == idx).any() for idx in class_indices
        ),
    )


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
    return filter_samples(
        dataset,
        predicate=lambda _img, lbl: not (
            (lbl >= num_cls) & (lbl != ignore)
        ).any(),
    )


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
                f"Class name {old_name!r} not found in dataset. "
                f"Available: {dataset.class_names}"
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
    return _RemappedDataset(dataset, remap, new_num_classes, all_names, ignore)


__all__ = [
    "concat_datasets",
    "filter_samples",
    "hold_out_classes",
    "hold_out_ood",
    "mark_as_ood",
    "remap_classes",
    "select_classes",
    "subset",
]
