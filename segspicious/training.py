"""Framework-level train / load / train_or_load functions.

These functions manage the lifecycle of a :class:`~segspicious.model.Model`:
training, checkpoint persistence via Hugging Face Hub, and cache-aware
reuse.  They return :class:`~segspicious.candidate.Candidate` instances
ready for prediction.

Before calling any function here, configure the Hub repository::

    import segspicious
    segspicious.configure(repo_id="myorg/segspicious-checkpoints")

Checkpoint layout inside the repository::

    {model.name}/{dataset.name}/
        <files written by model.save()>

The model is given a local directory to save into (and load from).
The framework uploads/downloads the entire directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from segspicious.candidate import Candidate
from segspicious.config import get_config
from segspicious.datasets.base import SegmentationDataset
from segspicious.model import Model


def _repo_dir(model: Model, dataset: SegmentationDataset) -> str:
    """Return the directory within the HF repo for a (model, dataset) checkpoint."""
    return f"{model.name}/{dataset.name}"


def train(
    model: Model,
    dataset: SegmentationDataset,
    validation_data: SegmentationDataset | None = None,
) -> Candidate:
    """Train the model, upload a checkpoint to HF Hub, return a Candidate.

    Args:
        model: An uninitialised model instance (no-arg constructed).
        dataset: Training data.
        validation_data: Optional validation split forwarded to
            ``model.train()``.

    Returns:
        A :class:`Candidate` binding the now-trained model to *dataset*.
    """
    config = get_config()

    model.train(dataset, validation_data=validation_data)

    with tempfile.TemporaryDirectory() as tmp:
        local_dir = Path(tmp)
        model.save(local_dir)

        api = HfApi()
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo=_repo_dir(model, dataset),
            repo_id=config.repo_id,
        )

    return Candidate(model=model, dataset=dataset)


def load(
    model: Model,
    dataset: SegmentationDataset,
) -> Candidate:
    """Download a checkpoint from HF Hub, load it into the model, return a Candidate.

    Args:
        model: An uninitialised model instance whose class matches the
            one used at training time.
        dataset: The dataset the model was trained on (used to derive the
            checkpoint path).

    Returns:
        A :class:`Candidate` binding the now-loaded model to *dataset*.

    Raises:
        FileNotFoundError: If no checkpoint exists at the derived path
            in the repository.
    """
    config = get_config()
    repo_dir = _repo_dir(model, dataset)

    local_root = Path(snapshot_download(
        repo_id=config.repo_id,
        allow_patterns=repo_dir + "/*",
    ))

    checkpoint_dir = local_root / repo_dir
    if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
        raise FileNotFoundError(
            f"No checkpoint found for model={model.name!r}, "
            f"dataset={dataset.name!r} in repo {config.repo_id!r}"
        )

    model.load(checkpoint_dir)
    return Candidate(model=model, dataset=dataset)


def train_or_load(
    model: Model,
    dataset: SegmentationDataset,
    validation_data: SegmentationDataset | None = None,
) -> Candidate:
    """Load if a checkpoint exists on HF Hub, otherwise train and upload.

    This is the primary entry point for experiment scripts.  Change the
    model class or dataset composition and it trains fresh; leave them
    unchanged and it loads from cache.

    Args:
        model: An uninitialised model instance.
        dataset: Training data.
        validation_data: Optional validation split (only used if training
            is needed).

    Returns:
        A :class:`Candidate` binding the model to *dataset*.
    """
    try:
        return load(model, dataset)
    except FileNotFoundError:
        return train(model, dataset, validation_data=validation_data)

