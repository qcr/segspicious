"""Tests for Phase 4: Candidate dataclass."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch
import pytest

from segspicious.candidate import Candidate
from segspicious.outputs import SegmentationOutput, UncertaintyOutput
from segspicious.tests.helpers import SyntheticDataset


# ── Stub model for testing ───────────────────────────────────────────────


class _StubModel:
    """Minimal model that satisfies the Model protocol for testing."""

    def __init__(self, model_name: str = "stub-model") -> None:
        self._name = model_name
        self._num_classes = 3

    @property
    def name(self) -> str:
        return self._name

    def train(self, dataset, validation_data=None) -> None:
        pass

    def predict(self, images: torch.Tensor) -> SegmentationOutput:
        b, _, h, w = images.shape
        return SegmentationOutput(
            prediction=torch.zeros(b, h, w, dtype=torch.long),
        )

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


# ── Candidate construction ───────────────────────────────────────────────


class TestCandidateConstruction:
    def test_is_dataclass(self):
        """Candidate is a dataclass with model and dataset fields."""
        field_names = {f.name for f in fields(Candidate)}
        assert field_names == {"model", "dataset"}

    def test_construct_with_positional_args(self):
        model = _StubModel()
        dataset = SyntheticDataset()
        candidate = Candidate(model, dataset)
        assert candidate.model is model
        assert candidate.dataset is dataset

    def test_construct_with_keyword_args(self):
        model = _StubModel()
        dataset = SyntheticDataset()
        candidate = Candidate(model=model, dataset=dataset)
        assert candidate.model is model
        assert candidate.dataset is dataset


# ── Name property ────────────────────────────────────────────────────────


class TestCandidateName:
    def test_name_format(self):
        model = _StubModel("deeplabv3-rn50")
        dataset = SyntheticDataset(name="wildscenes2d_train")
        candidate = Candidate(model, dataset)
        assert candidate.name == "deeplabv3-rn50/wildscenes2d_train"

    def test_name_with_modified_dataset(self):
        from segspicious.datasets import subset

        model = _StubModel("deeplabv3-rn50")
        dataset = SyntheticDataset(num_samples=20, name="wildscenes2d_train")
        sub = subset(dataset, n=10, seed=42)
        candidate = Candidate(model, sub)
        assert candidate.name == "deeplabv3-rn50/wildscenes2d_train-subset[n=10,seed=42]"

    def test_name_with_different_model(self):
        model = _StubModel("my-custom-model")
        dataset = SyntheticDataset(name="cityscapes_train")
        candidate = Candidate(model, dataset)
        assert candidate.name == "my-custom-model/cityscapes_train"


# ── Predict delegation ──────────────────────────────────────────────────


class TestCandidatePredict:
    def test_predict_delegates_to_model(self):
        model = _StubModel()
        dataset = SyntheticDataset()
        candidate = Candidate(model, dataset)

        images = torch.rand(2, 3, 8, 8)
        output = candidate.predict(images)

        assert isinstance(output, SegmentationOutput)
        assert output.prediction.shape == (2, 8, 8)

    def test_predict_returns_model_output(self):
        """predict() returns exactly what the model returns."""

        class _UncertaintyStubModel(_StubModel):
            def predict(self, images: torch.Tensor) -> UncertaintyOutput:
                b, _, h, w = images.shape
                return UncertaintyOutput(
                    prediction=torch.ones(b, h, w, dtype=torch.long),
                    class_probs=torch.rand(b, 3, h, w),
                )

        model = _UncertaintyStubModel()
        dataset = SyntheticDataset()
        candidate = Candidate(model, dataset)

        images = torch.rand(1, 3, 8, 8)
        output = candidate.predict(images)

        assert isinstance(output, UncertaintyOutput)
        assert output.class_probs is not None


# ── Export ───────────────────────────────────────────────────────────────


class TestExport:
    def test_importable_from_package(self):
        from segspicious import Candidate as C
        assert C is Candidate
