"""Model interface for segmentation UQ benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from torch import Tensor

from segspicious.datasets.base import SegmentationDataset
from segspicious.outputs import SegmentationOutput


@runtime_checkable
class Model(Protocol):
    """A segmentation model: architecture + training recipe + inference logic.

    The class *is* the configuration — no constructor arguments.
    Different hyperparameters = different class (use inheritance to share
    machinery).  Dataset-dependent values like ``num_classes`` are
    discovered from the dataset at train time.

    Lifecycle: **construct → train → save → load → predict**.

    The experiment script constructs the model (no arguments), hands it
    to a framework function like ``train_or_load``, and later calls
    ``predict()`` with batched image tensors.
    """

    @property
    def name(self) -> str:
        """Identifier for checkpoint paths and results tables."""
        ...

    def train(
        self,
        dataset: SegmentationDataset,
        validation_data: SegmentationDataset | None = None,
    ) -> None:
        """Train on the given dataset.

        The model owns its full training procedure: architecture,
        optimiser, schedule, augmentation, epochs, DataLoader construction,
        everything. The experiment only provides data.

        Args:
            dataset: Training data.
            validation_data: Optional held-out split used for monitoring
                training progress (e.g. early stopping, best-checkpoint
                selection). The model should restore its best weights
                before returning. ``None`` means no validation monitoring.

        A pre-trained model may implement this as a no-op.
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

        Only learned state — the model's configuration (architecture,
        hyperparameters) lives in the class definition itself.
        """
        ...

    def load(self, path: Path) -> None:
        """Load learned state from disk.

        The model object must already exist (constructed with matching
        class). This mirrors PyTorch's
        ``model.load_state_dict(torch.load(path))`` pattern.
        """
        ...
