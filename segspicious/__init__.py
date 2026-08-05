"""Segspicious: benchmarking segmentation uncertainty quantification."""

from segspicious.candidate import Candidate
from segspicious.model import Model
from segspicious.outputs import SegmentationOutput, UncertaintyOutput

__all__ = ["Candidate", "Model", "SegmentationOutput", "UncertaintyOutput"]
