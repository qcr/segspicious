"""Tests for Phase 5: Framework train / load / train_or_load."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from segspicious.candidate import Candidate
from segspicious.outputs import SegmentationOutput
from segspicious.tests.helpers import SyntheticDataset
from segspicious.training import _dataset_hash, load, train, train_or_load


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

    def save(self, path: Path) -> None:
        self.calls.append("save")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._state, path)

    def load(self, path: Path) -> None:
        self.calls.append("load")
        self._state = torch.load(path, map_location="cpu", weights_only=True)


# ── Checkpoint path derivation ───────────────────────────────────────────


class TestDatasetHash:
    def test_deterministic(self):
        """Same name always produces the same hash."""
        assert _dataset_hash("wildscenes2d_train") == _dataset_hash("wildscenes2d_train")

    def test_length_is_16(self):
        assert len(_dataset_hash("wildscenes2d_train")) == 16

    def test_hex_chars_only(self):
        h = _dataset_hash("some-dataset-name")
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_names_different_hashes(self):
        h1 = _dataset_hash("wildscenes2d_train")
        h2 = _dataset_hash("wildscenes2d_val")
        assert h1 != h2

    def test_complex_name(self):
        """Long compound dataset names still produce a valid 16-char hash."""
        name = "wildscenes2d_train-subset[n=100,seed=42]-mark_ood[bicycle+motorcycle]-hold_out_ood"
        h = _dataset_hash(name)
        assert len(h) == 16


# ── train() ──────────────────────────────────────────────────────────────


class TestTrain:
    def test_calls_train_then_save(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        candidate = train(model, dataset, root=tmp_path)

        assert model.calls == ["train", "save"]

    def test_returns_candidate(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        candidate = train(model, dataset, root=tmp_path)

        assert isinstance(candidate, Candidate)
        assert candidate.model is model
        assert candidate.dataset is dataset

    def test_candidate_name(self, tmp_path):
        model = _RecordingModel("my-model")
        dataset = SyntheticDataset(name="my-dataset")

        candidate = train(model, dataset, root=tmp_path)

        assert candidate.name == "my-model/my-dataset"

    def test_checkpoint_written(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        train(model, dataset, root=tmp_path)

        h = _dataset_hash("ds_train")
        cp = tmp_path / "stub-model" / h / "checkpoint.pt"
        assert cp.exists()

    def test_sidecar_written(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        train(model, dataset, root=tmp_path)

        h = _dataset_hash("ds_train")
        sidecar = tmp_path / "stub-model" / h / "dataset_name.txt"
        assert sidecar.exists()
        assert sidecar.read_text(encoding="utf-8") == "ds_train"

    def test_passes_validation_data(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        val_data = SyntheticDataset(name="ds_val")

        train(model, dataset, validation_data=val_data, root=tmp_path)

        assert model._validation_data is val_data

    def test_passes_none_validation_data(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        train(model, dataset, root=tmp_path)

        assert model._validation_data is None


# ── load() ───────────────────────────────────────────────────────────────


class TestLoad:
    def test_calls_load(self, tmp_path):
        """load() calls model.load with the correct checkpoint path."""
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        # First train to create checkpoint
        train(model, dataset, root=tmp_path)

        model2 = _RecordingModel()
        candidate = load(model2, dataset, root=tmp_path)

        assert "load" in model2.calls

    def test_returns_candidate(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        train(model, dataset, root=tmp_path)

        model2 = _RecordingModel()
        candidate = load(model2, dataset, root=tmp_path)

        assert isinstance(candidate, Candidate)
        assert candidate.model is model2
        assert candidate.dataset is dataset

    def test_raises_file_not_found(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="nonexistent")

        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            load(model, dataset, root=tmp_path)

    def test_does_not_call_train(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        train(model, dataset, root=tmp_path)

        model2 = _RecordingModel()
        load(model2, dataset, root=tmp_path)

        assert "train" not in model2.calls


# ── train_or_load() ──────────────────────────────────────────────────────


class TestTrainOrLoad:
    def test_trains_when_no_checkpoint(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        candidate = train_or_load(model, dataset, root=tmp_path)

        assert model.calls == ["train", "save"]
        assert isinstance(candidate, Candidate)

    def test_loads_when_checkpoint_exists(self, tmp_path):
        model1 = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        # First call trains
        train_or_load(model1, dataset, root=tmp_path)

        # Second call loads
        model2 = _RecordingModel()
        candidate = train_or_load(model2, dataset, root=tmp_path)

        assert model2.calls == ["load"]
        assert isinstance(candidate, Candidate)
        assert candidate.model is model2

    def test_returns_correct_candidate_name(self, tmp_path):
        model = _RecordingModel("arch-v1")
        dataset = SyntheticDataset(name="wildscenes2d_train")

        candidate = train_or_load(model, dataset, root=tmp_path)

        assert candidate.name == "arch-v1/wildscenes2d_train"

    def test_different_datasets_different_checkpoints(self, tmp_path):
        ds1 = SyntheticDataset(name="dataset_a")
        ds2 = SyntheticDataset(name="dataset_b")

        model1 = _RecordingModel()
        train_or_load(model1, ds1, root=tmp_path)

        model2 = _RecordingModel()
        train_or_load(model2, ds2, root=tmp_path)

        # Both should have trained (different datasets)
        assert model1.calls == ["train", "save"]
        assert model2.calls == ["train", "save"]

    def test_different_models_different_checkpoints(self, tmp_path):
        dataset = SyntheticDataset(name="ds_train")

        model1 = _RecordingModel("model-a")
        train_or_load(model1, dataset, root=tmp_path)

        model2 = _RecordingModel("model-b")
        train_or_load(model2, dataset, root=tmp_path)

        # Both should have trained (different model names)
        assert model1.calls == ["train", "save"]
        assert model2.calls == ["train", "save"]

    def test_validation_data_forwarded(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")
        val_data = SyntheticDataset(name="ds_val")

        train_or_load(model, dataset, validation_data=val_data, root=tmp_path)

        assert model._validation_data is val_data

    def test_validation_data_not_used_on_load(self, tmp_path):
        """When loading, validation_data is irrelevant."""
        model1 = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        train_or_load(model1, dataset, root=tmp_path)

        model2 = _RecordingModel()
        val_data = SyntheticDataset(name="ds_val")
        candidate = train_or_load(model2, dataset, validation_data=val_data, root=tmp_path)

        assert model2.calls == ["load"]


# ── Checkpoint path determinism ──────────────────────────────────────────


class TestCheckpointPaths:
    def test_path_uses_model_name(self, tmp_path):
        model = _RecordingModel("deeplabv3-rn50")
        dataset = SyntheticDataset(name="ds_train")

        train(model, dataset, root=tmp_path)

        assert (tmp_path / "deeplabv3-rn50").is_dir()

    def test_path_uses_dataset_hash(self, tmp_path):
        model = _RecordingModel()
        dataset = SyntheticDataset(name="wildscenes2d_train-subset[n=100,seed=42]")

        train(model, dataset, root=tmp_path)

        h = _dataset_hash("wildscenes2d_train-subset[n=100,seed=42]")
        assert (tmp_path / "stub-model" / h / "checkpoint.pt").exists()
        assert (tmp_path / "stub-model" / h / "dataset_name.txt").exists()

    def test_sidecar_contains_full_dataset_name(self, tmp_path):
        full_name = "cityscapes_train-mark_ood[bicycle+motorcycle]-hold_out_ood"
        model = _RecordingModel()
        dataset = SyntheticDataset(name=full_name)

        train(model, dataset, root=tmp_path)

        h = _dataset_hash(full_name)
        sidecar = tmp_path / "stub-model" / h / "dataset_name.txt"
        assert sidecar.read_text(encoding="utf-8") == full_name

    def test_string_root_accepted(self, tmp_path):
        """root can be a plain string, not just a Path."""
        model = _RecordingModel()
        dataset = SyntheticDataset(name="ds_train")

        candidate = train(model, dataset, root=str(tmp_path))
        assert isinstance(candidate, Candidate)


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
