"""Concrete models for segmentation experiments."""

from experiments.models.deeplabv3_rn50 import DeepLabV3RN50
from experiments.models.pastel_dinov2 import PastelDINOv2, PastelDINOv2_N20

__all__ = ["DeepLabV3RN50", "PastelDINOv2", "PastelDINOv2_N20"]
