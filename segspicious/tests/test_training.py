"""Tests for training with Hugging Face Hub checkpoint storage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

from segspicious.candidate import Candidate
from segspicious.config import _config
from segspicious.outputs import SegmentationOutput
from segspicious.tests.helpers import SyntheticDataset
from segspicious.training import _repo_dir, load, train, train_or_load


# ── Stub model that records calls ────────────────────────────────────────


class _RecordingModel:
    """Minimal model that records which lifecycle methods were called."""

    def __init__(self, model_name: str = "stub-model") -> None:
        self._name = model_name
        self.calls: list[str] = []
        self._state: dict | None = None

    @property
    def name(self) -> str:
        return self._name

    def train(self, dataset, validation_data=None) -> None:
        self.calls.append("train")
        self._state = {"num_classes": dataset.num_classes}
        self._validation_data = validation_data

    def predict(self, images: torch.Tensor) -> SegmentationOutput:
        self.calls.append("predict")
        b, _, h, w = images.shape
        return SegmentationOutput(
            prediction=torch.zeros(b, h, w, dtype=torch.long),
        )

    def save(self, directory: Path) -> None:
        self.calls.append("save")
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self._state, directory / "checkpoint.pt")

    def load(self, directory: Path) -> None:
        self.calls.append("load")
        self._state = torch.load(
            directory / "checkpoint.pt", map_location="cpu", weights_only=True,
        )


# ── Fixtures ─────────────────────────────────────────────────────────────

REPO_ID = "test-org/test-checkpoints"


@pytest.fixture(autouse=True)
def _configure():
    """Set config for all tests, then reset afterwards."""
    _config.repo_id = REPO_ID
    _config._authenticated = True
    yield
    _config.repo_id = None
    _config._authenticated = False


@pytest.fixture()
def fake_hub(tmp_path):
    """Mock HfApi and snapshot_download with a local directory as backing store.

    Returns a dict with the mock objects for further assertions.
    """
    # Local storage backing the fake hub
    storage = tmp_path / "hub_storage"
    storage.mkdir()

    def fake_upload_folder(*, folder_path, path_in_repo, repo_id):
        import shutil
        dest = storage / path_in_repo
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(folder_path, dest)

    def fake_snapshot_download(repo_id, allow_patterns=None, **kwargs):
        # Return storage root; the caller appends the sub-path.
        return str(storage)

    mock_api_instance = MagicMock()
    mock_api_instance.upload_folder.side_effect = fake_upload_folder

    with (
        patch("segspicious.training.HfApi", return_value=mock_api_instance),
        patch("segspicious.training.snapshot_download", side_effect=fake_snapshot_download),
    ):
        yield {
            "api": mock_api_instance,
            "storage": storage,
        }


# ── Repo path derivation ────────────────────────────────────────────────


class TestRepoDir:
    def test_uses_model_and_dataset_name(self):
        model = _RecordingModel("deeplabv3-rn50")
        dataset = SyntheticDataset(name="wildscenes2d_train")
        assert _repo_dir(model, dataset) == "deeplabv3-rn50/wildscenes2d_train"

    def test_complex_dataset_name(self):
        model = _RecordingModel("my-model")
        dataset = SyntheticDataset(
            name="wildscenes2d_train-subset(n=100,seed=42)-mark_ood(bicycle+motorcycle)"
        )
        expected = (
            "my-model/"
            "wildscenes2d_train-subset(n=100,seed=42)-mark_ood(bicycle+motorcycle)"
        )
        assert _repo_dir(model, dataset) == expected


# ── train() ──────────────────────────────────────────────────────────────


class TestTrain:
    def test_calls_train_then_save(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        train(model, dataset)

        assert model.calls == ["train", "save"]

    def test_returns_candidate(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        candidate = train(model, dataset)

        assert isinstance(candidate, Candidate)
        assert candidate.model is model
        assert candidate.dataset is dataset

    def test_candidate_name(self, fake_hub):
        model = _RecordingModel("my-model")
        dataset = SyntheticDataset(name="my-dataset")

        candidate = train(model, dataset)

        assert candidate.name == "my-model/my-dataset"

    def test_uploads_checkpoint(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        train(model, dataset)

        fake_hub["api"].upload_folder.assert_called_once()
        call_kwargs = fake_hub["api"].upload_folder.call_args
        assert call_kwargs.kwargs["path_in_repo"] == "stub-model/ds_train"
        assert call_kwargs.kwargs["repo_id"] == REPO_ID

    def test_checkpoint_file_in_storage(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        train(model, dataset)

        cp = fake_hub["storage"] / "stub-model" / "ds_train" / "checkpoint.pt"
        assert cp.exists()

    def test_passes_validation_data(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        val_data = SyntheticDataset(name="ds_val")

        train(model, dataset, validation_data=val_data)

        assert model._validation_data is val_data

    def test_passes_none_validation_data(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        train(model, dataset)

        assert model._validation_data is None


# ── load() ───────────────────────────────────────────────────────────────


class TestLoad:
    def test_calls_load(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        train(model, dataset)

        model2 = _RecordingModel()
        load(model2, dataset)

        assert "load" in model2.calls

    def test_returns_candidate(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        train(model, dataset)

        model2 = _RecordingModel()
        candidate = load(model2, dataset)

        assert isinstance(candidate, Candidate)
        assert candidate.model is model2
        assert candidate.dataset is dataset

    def test_raises_file_not_found(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="nonexistent")

        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            load(model, dataset)

    def test_does_not_call_train(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        train(model, dataset)

        model2 = _RecordingModel()
        load(model2, dataset)

        assert "train" not in model2.calls


# ── train_or_load() ──────────────────────────────────────────────────────


class TestTrainOrLoad:
    def test_trains_when_no_checkpoint(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        candidate = train_or_load(model, dataset)

        assert model.calls == ["train", "save"]
        assert isinstance(candidate, Candidate)

    def test_loads_when_checkpoint_exists(self, fake_hub):
        model1 = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        train_or_load(model1, dataset)

        model2 = _RecordingModel()
        candidate = train_or_load(model2, dataset)

        assert model2.calls == ["load"]
        assert isinstance(candidate, Candidate)
        assert candidate.model is model2

    def test_returns_correct_candidate_name(self, fake_hub):
        model = _RecordingModel("arch-v1")
        dataset = SyntheticDataset(name="wildscenes2d_train")

        candidate = train_or_load(model, dataset)

        assert candidate.name == "arch-v1/wildscenes2d_train"

    def test_different_datasets_different_checkpoints(self, fake_hub):
        ds1 = SyntheticDataset(name="dataset_a")
        ds2 = SyntheticDataset(name="dataset_b")

        model1 = _RecordingModel()
        train_or_load(model1, ds1)

        model2 = _RecordingModel()
        train_or_load(model2, ds2)

        assert model1.calls == ["train", "save"]
        assert model2.calls == ["train", "save"]

    def test_different_models_different_checkpoints(self, fake_hub):
        dataset = SyntheticDataset(name="ds_train")

        model1 = _RecordingModel("model-a")
        train_or_load(model1, dataset)

        model2 = _RecordingModel("model-b")
        train_or_load(model2, dataset)

        assert model1.calls == ["train", "save"]
        assert model2.calls == ["train", "save"]

    def test_validation_data_forwarded(self, fake_hub):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        val_data = SyntheticDataset(name="ds_val")

        train_or_load(model, dataset, validation_data=val_data)

        assert model._validation_data is val_data

    def test_validation_data_not_used_on_load(self, fake_hub):
        model1 = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        train_or_load(model1, dataset)

        model2 = _RecordingModel()
        val_data = SyntheticDataset(name="ds_val")
        train_or_load(model2, dataset, validation_data=val_data)

        assert model2.calls == ["load"]


# ── Config requirement ───────────────────────────────────────────────────


class TestConfigRequired:
    def test_train_raises_without_configure(self):
        _config.repo_id = None
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        with pytest.raises(RuntimeError, match="not configured"):
            train(model, dataset)

    def test_load_raises_without_configure(self):
        _config.repo_id = None
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        with pytest.raises(RuntimeError, match="not configured"):
            load(model, dataset)

    def test_train_or_load_raises_without_configure(self):
        _config.repo_id = None
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        with pytest.raises(RuntimeError, match="not configured"):
            train_or_load(model, dataset)


# ── Export ───────────────────────────────────────────────────────────────


class TestExport:
    def test_train_importable_from_package(self):
        from segspicious import train as t
        assert t is train

    def test_load_importable_from_package(self):
        from segspicious import load as l
        assert l is load

    def test_train_or_load_importable_from_package(self):
        from segspicious import train_or_load as tol
        assert tol is train_or_load

    def test_configure_importable_from_package(self):
        from segspicious import configure
        from segspicious.config import configure as cfg
        assert configure is cfg
