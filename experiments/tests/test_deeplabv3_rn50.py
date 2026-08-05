"""Tests for Phase 3: DeepLabV3RN50 matches Model protocol."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import pytest

from experiments.models import DeepLabV3RN50
from segspicious import Model
from segspicious.outputs import UncertaintyOutput
from segspicious.tests.helpers import SyntheticDataset


# ── Protocol conformance ─────────────────────────────────────────────────


class TestProtocolConformance:
    def test_isinstance_check(self):
        model = DeepLabV3RN50()
        assert isinstance(model, Model)

    def test_no_constructor_args(self):
        """Model protocol: no constructor arguments."""
        model = DeepLabV3RN50()
        assert model._model is None
        assert model._num_classes is None

    def test_name_property(self):
        model = DeepLabV3RN50()
        assert model.name == "deeplabv3-rn50"


# ── Class attributes as configuration ────────────────────────────────────


class TestClassAttributes:
    def test_default_hyperparameters(self):
        model = DeepLabV3RN50()
        assert model.epochs == 50
        assert model.batch_size == 4
        assert model.lr == 0.01
        assert model.crop_size == 512
        assert model.num_workers == 4

    def test_inheritance_overrides(self):
        """Different hyperparameters = different class via inheritance."""

        class DeepLabV3RN50_LowLR(DeepLabV3RN50):
            lr = 0.001

            @property
            def name(self) -> str:
                return "deeplabv3-rn50-lowlr"

        model = DeepLabV3RN50_LowLR()
        assert model.lr == 0.001
        assert model.epochs == 50  # inherited default
        assert model.name == "deeplabv3-rn50-lowlr"
        assert isinstance(model, Model)

    def test_inheritance_multiple_overrides(self):
        class DeepLabV3RN50_Quick(DeepLabV3RN50):
            epochs = 2
            lr = 0.001
            batch_size = 2

            @property
            def name(self) -> str:
                return "deeplabv3-rn50-quick"

        model = DeepLabV3RN50_Quick()
        assert model.epochs == 2
        assert model.lr == 0.001
        assert model.batch_size == 2
        assert model.crop_size == 512  # inherited
        assert model.name == "deeplabv3-rn50-quick"


# ── Predict guard ────────────────────────────────────────────────────────


class TestPredictGuard:
    def test_predict_before_train_raises(self):
        model = DeepLabV3RN50()
        images = torch.rand(1, 3, 64, 64)
        with pytest.raises(AssertionError, match="not initialised"):
            model.predict(images)

    def test_save_before_train_raises(self):
        model = DeepLabV3RN50()
        with pytest.raises(AssertionError, match="not initialised"):
            model.save(Path("/tmp/test_checkpoint.pt"))


# ── Save / load round-trip ───────────────────────────────────────────────


class TestSaveLoad:
    def test_save_load_preserves_num_classes(self):
        """save() persists num_classes; load() recovers it and rebuilds."""
        model = DeepLabV3RN50()

        # Simulate a trained model (build without full training)
        num_classes = 7
        model._num_classes = num_classes
        model._model = model._build_model(num_classes)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            model.save(path)

            # Load into a fresh instance
            model2 = DeepLabV3RN50()
            assert model2._model is None
            model2.load(path)

            assert model2._num_classes == num_classes
            assert model2._model is not None

    def test_save_load_weights_match(self):
        """Loaded model produces the same output as the saved model."""
        model = DeepLabV3RN50()
        num_classes = 3
        model._num_classes = num_classes
        model._model = model._build_model(num_classes)

        images = torch.rand(1, 3, 64, 64)
        out1 = model.predict(images)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            model.save(path)

            model2 = DeepLabV3RN50()
            model2.load(path)
            out2 = model2.predict(images)

        assert torch.allclose(out1.prediction, out2.prediction)
        assert torch.allclose(out1.class_probs, out2.class_probs, atol=1e-6)
        assert torch.allclose(
            out1.predictive_uncertainty, out2.predictive_uncertainty, atol=1e-6
        )

    def test_checkpoint_contains_num_classes(self):
        """Checkpoint file stores both state_dict and num_classes."""
        model = DeepLabV3RN50()
        model._num_classes = 5
        model._model = model._build_model(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            model.save(path)

            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            assert "state_dict" in checkpoint
            assert "num_classes" in checkpoint
            assert checkpoint["num_classes"] == 5


# ── Train discovers num_classes ──────────────────────────────────────────


class TestTrainDiscoversNumClasses:
    def test_train_sets_num_classes_and_builds_model(self):
        """train() discovers num_classes from the dataset."""

        class QuickDeepLab(DeepLabV3RN50):
            epochs = 1
            batch_size = 2
            num_workers = 0

        ds = SyntheticDataset(num_samples=4, num_classes=3, height=64, width=64)
        model = QuickDeepLab()

        assert model._num_classes is None
        assert model._model is None

        model.train(ds)

        assert model._num_classes == 3
        assert model._model is not None

    def test_predict_works_after_train(self):
        """Model can predict after training."""

        class QuickDeepLab(DeepLabV3RN50):
            epochs = 1
            batch_size = 2
            num_workers = 0

        ds = SyntheticDataset(num_samples=4, num_classes=3, height=64, width=64)
        model = QuickDeepLab()
        model.train(ds)

        images = torch.rand(1, 3, 64, 64)
        output = model.predict(images)

        assert isinstance(output, UncertaintyOutput)
        assert output.prediction.shape == (1, 64, 64)
        assert output.class_probs.shape == (1, 3, 64, 64)
        assert output.predictive_uncertainty.shape == (1, 64, 64)

    def test_train_then_save_load_predict(self):
        """Full lifecycle: train → save → load → predict."""

        class QuickDeepLab(DeepLabV3RN50):
            epochs = 1
            batch_size = 2
            num_workers = 0

        ds = SyntheticDataset(num_samples=4, num_classes=4, height=64, width=64)
        model = QuickDeepLab()
        model.train(ds)

        images = torch.rand(1, 3, 64, 64)
        out_before = model.predict(images)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            model.save(path)

            model2 = QuickDeepLab()
            model2.load(path)
            out_after = model2.predict(images)

        assert torch.allclose(out_before.prediction, out_after.prediction)
        assert torch.allclose(out_before.class_probs, out_after.class_probs, atol=1e-6)
