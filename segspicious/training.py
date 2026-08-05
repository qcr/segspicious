"""Framework-level train / load / train_or_load functions.

These functions manage the lifecycle of a :class:`~segspicious.model.Model`:
training, checkpoint persistence, and cache-aware reuse.  They return
:class:`~segspicious.candidate.Candidate` instances ready for prediction.

Checkpoint layout::

    {root}/{model.name}/{hash}/checkpoint.pt
    {root}/{model.name}/{hash}/dataset_name.txt

where *hash* is a deterministic short hash of ``dataset.name`` (SHA-256
truncated to 16 hex chars).  This keeps model names human-readable in the
directory tree while avoiding filesystem issues with long or
special-character dataset names.  The sidecar ``dataset_name.txt`` stores
the full dataset name for human inspection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from segspicious.candidate import Candidate
from segspicious.datasets.base import SegmentationDataset
from segspicious.model import Model


def _dataset_hash(dataset_name: str) -> str:
    """Return a deterministic 16-hex-char hash of *dataset_name*."""
    return hashlib.sha256(dataset_name.encode("utf-8")).hexdigest()[:16]


def _checkpoint_dir(model: Model, dataset: SegmentationDataset, root: str | Path) -> Path:
    """Return the checkpoint directory for a (model, dataset) pair."""
    return Path(root) / model.name / _dataset_hash(dataset.name)


def _checkpoint_path(model: Model, dataset: SegmentationDataset, root: str | Path) -> Path:
    """Return the checkpoint file path for a (model, dataset) pair."""
    return _checkpoint_dir(model, dataset, root) / "checkpoint.pt"


def _write_dataset_sidecar(model: Model, dataset: SegmentationDataset, root: str | Path) -> None:
    """Write the ``dataset_name.txt`` sidecar file."""
    d = _checkpoint_dir(model, dataset, root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "dataset_name.txt").write_text(dataset.name, encoding="utf-8")


def train(
    model: Model,
    dataset: SegmentationDataset,
    validation_data: SegmentationDataset | None = None,
    root: str | Path = "trained",
) -> Candidate:
    """Train the model, save a checkpoint, return a Candidate.

    Args:
        model: An uninitialised model instance (no-arg constructed).
        dataset: Training data.
        validation_data: Optional validation split forwarded to
            ``model.train()``.
        root: Root directory for checkpoint storage.

    Returns:
        A :class:`Candidate` binding the now-trained model to *dataset*.
    """
    model.train(dataset, validation_data=validation_data)

    path = _checkpoint_path(model, dataset, root)
    model.save(path)
    _write_dataset_sidecar(model, dataset, root)

    return Candidate(model=model, dataset=dataset)


def load(
    model: Model,
    dataset: SegmentationDataset,
    root: str | Path = "trained",
) -> Candidate:
    """Load an existing checkpoint into the model, return a Candidate.

    Args:
        model: An uninitialised model instance whose class matches the
            one used at training time.
        dataset: The dataset the model was trained on (used to derive the
            checkpoint path).
        root: Root directory for checkpoint storage.

    Returns:
        A :class:`Candidate` binding the now-loaded model to *dataset*.

    Raises:
        FileNotFoundError: If no checkpoint exists at the derived path.
    """
    path = _checkpoint_path(model, dataset, root)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {path} for model={model.name!r}, "
            f"dataset={dataset.name!r}"
        )
    model.load(path)
    return Candidate(model=model, dataset=dataset)


def train_or_load(
    model: Model,
    dataset: SegmentationDataset,
    validation_data: SegmentationDataset | None = None,
    root: str | Path = "trained",
) -> Candidate:
    """Load if a checkpoint exists, otherwise train and save.

    This is the primary entry point for experiment scripts.  Change the
    model class or dataset composition and it trains fresh; leave them
    unchanged and it loads from cache.

    Args:
        model: An uninitialised model instance.
        dataset: Training data.
        validation_data: Optional validation split (only used if training
            is needed).
        root: Root directory for checkpoint storage.

    Returns:
        A :class:`Candidate` binding the model to *dataset*.
    """
    path = _checkpoint_path(model, dataset, root)
    if path.exists():
        model.load(path)
        return Candidate(model=model, dataset=dataset)
    return train(model, dataset, validation_data=validation_data, root=root)
