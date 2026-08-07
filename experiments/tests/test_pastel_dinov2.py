"""Tests for PastelDINOv2 model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import pytest

from experiments.models import PastelDINOv2
from experiments.models.pastel_dinov2 import PastelDINOv2_N20
from segspicious import Model
from segspicious.outputs import UncertaintyOutput
from segspicious.tests.helpers import SyntheticDataset


# ── Quick subclass for fast tests ─────────────────────────────────────────


class QuickPastel(PastelDINOv2):
    epochs = 1
    batch_size = 2
    num_workers = 0


# ── Protocol conformance ─────────────────────────────────────────────────


class TestProtocolConformance:
    def test_isinstance_check(self):
        model = PastelDINOv2()
        assert isinstance(model, Model)

    def test_no_constructor_args(self):
        """Model protocol: no constructor arguments."""
        model = PastelDINOv2()
        assert model._model is None
        assert model._num_classes is None

    def test_name_property(self):
        model = PastelDINOv2()
        assert model.name == "pastel-dinov2-vitl14"


# ── Class attributes as configuration ────────────────────────────────────


class TestClassAttributes:
    def test_default_hyperparameters(self):
        model = PastelDINOv2()
        assert model.backbone == "dinov2_vitl14"
        assert model.feat_dim == 1024
        assert model.epochs == 150
        assert model.batch_size == 4
        assert model.lr == 1e-3
        assert model.crop_size == 504
        assert model.hard_mining_ratio == 0.2
        assert model.num_train_samples is None
        assert model.num_workers == 4
        assert model.val_interval == 10
        assert model.log_dir == "runs"

    def test_inheritance_n_override(self):
        """Sample count variant via inheritance."""
        model = PastelDINOv2_N20()
        assert model.num_train_samples == 20
        assert model.name == "pastel-dinov2-vitl14-n20"
        assert model.backbone == "dinov2_vitl14"  # inherited default
        assert isinstance(model, Model)

    def test_name_with_num_train_samples(self):
        class PastelN20(PastelDINOv2):
            num_train_samples = 20

        model = PastelN20()
        assert model.name == "pastel-dinov2-vitl14-n20"

    def test_name_without_num_train_samples(self):
        model = PastelDINOv2()
        assert model.name == "pastel-dinov2-vitl14"

    def test_inheritance_multiple_overrides(self):
        class QuickPastelN50(PastelDINOv2):
            epochs = 2
            lr = 0.0001
            batch_size = 2
            num_train_samples = 50

        model = QuickPastelN50()
        assert model.epochs == 2
        assert model.lr == 0.0001
        assert model.batch_size == 2
        assert model.num_train_samples == 50
        assert model.crop_size == 504  # inherited
        assert "n50" in model.name


# ── Predict/save guards ──────────────────────────────────────────────────


class TestGuards:
    def test_predict_before_train_raises(self):
        model = PastelDINOv2()
        images = torch.rand(1, 3, 64, 64)
        with pytest.raises(AssertionError, match="not initialised"):
            model.predict(images)

    def test_save_before_train_raises(self):
        model = PastelDINOv2()
        with pytest.raises(AssertionError, match="not initialised"):
            model.save(Path("/tmp/test_pastel_checkpoint_dir"))


# ── Save / load round-trip ───────────────────────────────────────────────


class TestSaveLoad:
    def test_save_load_preserves_num_classes(self):
        """save() persists num_classes; load() recovers it and rebuilds."""
        model = QuickPastel()
        num_classes = 5
        model._num_classes = num_classes
        model._model = model._build_model(num_classes)

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            model.save(directory)

            model2 = QuickPastel()
            assert model2._model is None
            model2.load(directory)

            assert model2._num_classes == num_classes
            assert model2._model is not None

    def test_save_load_weights_match(self):
        """Loaded model produces the same output as the saved model."""
        model = QuickPastel()
        num_classes = 3
        model._num_classes = num_classes
        model._model = model._build_model(num_classes)

        images = torch.rand(1, 3, 56, 56)  # divisible by 14
        out1 = model.predict(images)

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            model.save(directory)

            model2 = QuickPastel()
            model2.load(directory)
            out2 = model2.predict(images)

        assert torch.allclose(out1.prediction, out2.prediction)
        assert torch.allclose(out1.class_probs, out2.class_probs, atol=1e-6)
        assert torch.allclose(
            out1.predictive_uncertainty, out2.predictive_uncertainty, atol=1e-6
        )

    def test_checkpoint_contains_head_and_num_classes(self):
        """Checkpoint stores head_state_dict and num_classes (not full model)."""
        model = QuickPastel()
        model._num_classes = 5
        model._model = model._build_model(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            model.save(directory)

            checkpoint = torch.load(
                directory / "checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert "head_state_dict" in checkpoint
            assert "num_classes" in checkpoint
            assert checkpoint["num_classes"] == 5
            # Should NOT contain full model state (no encoder weights)
            assert "state_dict" not in checkpoint

    def test_checkpoint_is_small(self):
        """Only head weights are saved — checkpoint should be small."""
        model = QuickPastel()
        model._num_classes = 10
        model._model = model._build_model(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            model.save(directory)

            size_bytes = (directory / "checkpoint.pt").stat().st_size
            # Head is ~600K params → ~2.4MB. Allow generous margin.
            assert size_bytes < 10_000_000, (
                f"Checkpoint too large ({size_bytes / 1e6:.1f}MB) — "
                "backbone weights may have been saved"
            )


# ── Train discovers num_classes ──────────────────────────────────────────


class TestTrainDiscoversNumClasses:
    def test_train_sets_num_classes_and_builds_model(self):
        ds = SyntheticDataset(
            num_samples=4, num_classes=3, height=56, width=56
        )
        model = QuickPastel()

        assert model._num_classes is None
        assert model._model is None

        model.train(ds)

        assert model._num_classes == 3
        assert model._model is not None

    def test_predict_works_after_train(self):
        ds = SyntheticDataset(
            num_samples=4, num_classes=3, height=56, width=56
        )
        model = QuickPastel()
        model.train(ds)

        images = torch.rand(1, 3, 56, 56)
        output = model.predict(images)

        assert isinstance(output, UncertaintyOutput)
        assert output.prediction.shape == (1, 56, 56)
        assert output.class_probs.shape == (1, 3, 56, 56)
        assert output.predictive_uncertainty.shape == (1, 56, 56)

    def test_train_then_save_load_predict(self):
        """Full lifecycle: train → save → load → predict."""
        ds = SyntheticDataset(
            num_samples=4, num_classes=4, height=56, width=56
        )
        model = QuickPastel()
        model.train(ds)

        images = torch.rand(1, 3, 56, 56)
        out_before = model.predict(images)

        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            model.save(directory)

            model2 = QuickPastel()
            model2.load(directory)
            out_after = model2.predict(images)

        assert torch.allclose(out_before.prediction, out_after.prediction)
        assert torch.allclose(
            out_before.class_probs, out_after.class_probs, atol=1e-6
        )


# ── PASTEL-specific ──────────────────────────────────────────────────────


class TestPastelSpecific:
    def test_num_train_samples_reflected_in_name(self):
        class PastelN20(PastelDINOv2):
            num_train_samples = 20

        model = PastelN20()
        assert model.name == "pastel-dinov2-vitl14-n20"

        model_full = PastelDINOv2()
        assert "n" not in model_full.name.split("vitl14")[-1]

    def test_balanced_subset_called_when_n_set(self):
        """When num_train_samples is set, balanced_subset is used internally."""

        class PastelN3(QuickPastel):
            num_train_samples = 3

        ds = SyntheticDataset(
            num_samples=10, num_classes=3, height=56, width=56
        )
        model = PastelN3()

        with patch(
            "experiments.models.pastel_dinov2.balanced_subset",
            wraps=lambda dataset, n: __import__(
                "segspicious.datasets.modifiers", fromlist=["balanced_subset"]
            ).balanced_subset(dataset, n),
        ) as mock_bal:
            model.train(ds)
            mock_bal.assert_called_once()
            call_args = mock_bal.call_args
            assert call_args[1].get("n", call_args[0][1] if len(call_args[0]) > 1 else None) == 3 or \
                   (len(call_args[0]) > 1 and call_args[0][1] == 3)

    def test_balanced_subset_not_called_when_n_none(self):
        """When num_train_samples is None, no subsetting occurs."""
        ds = SyntheticDataset(
            num_samples=4, num_classes=3, height=56, width=56
        )
        model = QuickPastel()

        with patch(
            "experiments.models.pastel_dinov2.balanced_subset",
        ) as mock_bal:
            model.train(ds)
            mock_bal.assert_not_called()
