"segspicious — protocols, modifiers, and metrics for segmentation UQ benchmarking."

from segspicious.protocols import Candidate, SegmentationDataset
from segspicious.types import (
    SegmentationOutput,
    SegmentationSample,
    UncertaintyOutput,
)

__all__ = [
    "Candidate",
    "SegmentationDataset",
    "SegmentationOutput",
    "SegmentationSample",
    "UncertaintyOutput",
]
