"""Candidate: a (model, dataset) pair ready for prediction."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from segspicious.datasets.base import SegmentationDataset
from segspicious.model import Model
from segspicious.outputs import SegmentationOutput


@dataclass
class Candidate:
    """A trained model bound to the dataset it was trained on.

    Not user-implemented.  Created by framework functions
    (e.g. ``train``, ``load``, ``train_or_load``) or directly by
    experiment code.

    The ``name`` property encodes both the model identity and the full
    dataset transformation chain, giving a unique, human-readable
    identifier suitable for results tables and checkpoint paths.
    """

    model: Model
    dataset: SegmentationDataset

    def predict(self, images: Tensor) -> SegmentationOutput:
        """Forward prediction to the underlying model.

        Args:
            images: ``(B, C, H, W)`` float tensor in ``[0, 1]``.

        Returns:
            ``SegmentationOutput`` (or a subclass like ``UncertaintyOutput``).
        """
        return self.model.predict(images)

    @property
    def name(self) -> str:
        """Unique identifier: ``{model.name}/{dataset.name}``."""
        return f"{self.model.name}/{self.dataset.name}"
