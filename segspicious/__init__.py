"segspicious — protocols, modifiers, and metrics for segmentation UQ benchmarking."

from segspicious.candidate import Candidate, SegmentationOutput, UncertaintyOutput
from segspicious.dataset import SegmentationDataset, SegmentationSample, Split
from segspicious.torch_adapter import TorchDatasetAdapter

__all__ = [
    "Candidate",
    "SegmentationDataset",
    "SegmentationOutput",
    "SegmentationSample",
    "Split",
    "TorchDatasetAdapter",
    "UncertaintyOutput",
]
