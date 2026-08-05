"""Segspicious: benchmarking segmentation uncertainty quantification."""

from segspicious.candidate import Candidate
from segspicious.model import Model
from segspicious.outputs import SegmentationOutput, UncertaintyOutput
from segspicious.training import load, train, train_or_load

__all__ = [
    "Candidate",
    "Model",
    "SegmentationOutput",
    "UncertaintyOutput",
    "load",
    "train",
    "train_or_load",
]
