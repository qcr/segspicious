"""Candidate interface for segmentation UQ benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from torch import Tensor

from segspicious.datasets.base import SegmentationDataset
from segspicious.outputs import SegmentationOutput


@runtime_checkable
class Candidate(Protocol):
    """A complete pipeline from input image to segmentation output.

    The candidate is the unit of comparison in an experiment. It owns its
    full training procedure (architecture, optimiser, augmentation, etc.)
    and its inference procedure (how raw model output becomes a
    ``SegmentationOutput`` or ``UncertaintyOutput``).

    Lifecycle: **construct → train → save → load → predict**.

    The experiment script constructs the candidate with its configuration,
    calls ``train()`` with a dataset, and later calls ``predict()`` with
    batched image tensors. ``save``/``load`` enable training on one machine
    and evaluating on another.
    """

    @property
    def name(self) -> str:
        """Identifier for results tables and saved state."""
        ...

    def train(self, dataset: SegmentationDataset) -> None:
        """Train on the given dataset.

        The candidate owns its full training procedure: architecture,
        optimiser, schedule, augmentation, epochs, DataLoader construction,
        everything. The experiment only provides data.

        A pre-trained candidate may implement this as a no-op.
        """
        ...

    def predict(self, images: Tensor) -> SegmentationOutput:
        """Produce output for a batch of images.

        Args:
            images: ``(B, C, H, W)`` float tensor in ``[0, 1]``.

        Returns:
            ``SegmentationOutput`` or ``UncertaintyOutput`` (which extends
            it). The framework inspects which fields are populated to
            determine compatible evaluation tests.
        """
        ...

    def save(self, path: Path) -> None:
        """Serialise learned state to disk.

        Only learned state — the candidate's configuration (architecture,
        hyperparameters) lives in the experiment code that constructs the
        candidate object.
        """
        ...

    def load(self, path: Path) -> None:
        """Load learned state from disk.

        The candidate object must already exist (constructed with matching
        configuration). This mirrors PyTorch's
        ``model.load_state_dict(torch.load(path))`` pattern.
        """
        ...
