"""Dataset base class and modifiers for segmentation benchmarking."""

from segspicious.datasets.base import SegmentationDataset, Split
from segspicious.datasets.cache import ClassIndexCache
from segspicious.datasets.modifiers import (
    concat_datasets,
    filter_by_labels,
    filter_samples,
    hold_out_classes,
    hold_out_ood,
    mark_as_ood,
    remap_classes,
    select_classes,
    subset,
)

__all__ = [
    "ClassIndexCache",
    "SegmentationDataset",
    "Split",
    "concat_datasets",
    "filter_by_labels",
    "filter_samples",
    "hold_out_classes",
    "hold_out_ood",
    "mark_as_ood",
    "remap_classes",
    "select_classes",
    "subset",
]
