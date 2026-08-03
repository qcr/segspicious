"""Dataset base class and modifiers for segmentation benchmarking."""

from segspicious.datasets.base import SegmentationDataset
from segspicious.datasets.modifiers import (
    concat_datasets,
    filter_samples,
    hold_out_classes,
    hold_out_ood,
    mark_as_ood,
    remap_classes,
    select_classes,
    subset,
)

__all__ = [
    "SegmentationDataset",
    "concat_datasets",
    "filter_samples",
    "hold_out_classes",
    "hold_out_ood",
    "mark_as_ood",
    "remap_classes",
    "select_classes",
    "subset",
]
