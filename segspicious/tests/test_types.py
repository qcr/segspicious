"""Tests for core data types."""

import numpy as np

from segspicious.types import (
    SegmentationOutput,
    SegmentationSample,
    UncertaintyOutput,
)


class TestSegmentationOutput:
    def test_construction(self):
        pred = np.array([[0, 1], [2, 0]])
        out = SegmentationOutput(prediction=pred)
        np.testing.assert_array_equal(out.prediction, pred)

    def test_field_access(self):
        pred = np.zeros((4, 4), dtype=int)
        out = SegmentationOutput(prediction=pred)
        assert out.prediction.shape == (4, 4)


class TestUncertaintyOutput:
    def test_inherits_prediction(self):
        pred = np.array([[0, 1], [1, 0]])
        out = UncertaintyOutput(prediction=pred)
        assert isinstance(out, SegmentationOutput)
        np.testing.assert_array_equal(out.prediction, pred)

    def test_optional_fields_default_to_none(self):
        out = UncertaintyOutput(prediction=np.zeros((2, 2), dtype=int))
        assert out.class_probs is None
        assert out.predictive_uncertainty is None
        assert out.aleatoric_uncertainty is None
        assert out.epistemic_uncertainty is None
        assert out.ood_score is None

    def test_populate_subset_of_fields(self):
        pred = np.array([[0, 1], [1, 0]])
        probs = np.random.rand(2, 2, 3)
        entropy = np.random.rand(2, 2)
        out = UncertaintyOutput(
            prediction=pred,
            class_probs=probs,
            predictive_uncertainty=entropy,
        )
        np.testing.assert_array_equal(out.class_probs, probs)
        np.testing.assert_array_equal(out.predictive_uncertainty, entropy)
        assert out.aleatoric_uncertainty is None
        assert out.epistemic_uncertainty is None
        assert out.ood_score is None

    def test_populate_all_fields(self):
        h, w, c = 4, 4, 3
        out = UncertaintyOutput(
            prediction=np.zeros((h, w), dtype=int),
            class_probs=np.ones((h, w, c)) / c,
            predictive_uncertainty=np.zeros((h, w)),
            aleatoric_uncertainty=np.zeros((h, w)),
            epistemic_uncertainty=np.zeros((h, w)),
            ood_score=np.zeros((h, w)),
        )
        assert out.class_probs.shape == (h, w, c)
        assert out.predictive_uncertainty.shape == (h, w)
        assert out.aleatoric_uncertainty.shape == (h, w)
        assert out.epistemic_uncertainty.shape == (h, w)
        assert out.ood_score.shape == (h, w)


class TestSegmentationSample:
    def test_construction(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        labels = np.zeros((4, 4), dtype=int)
        sample = SegmentationSample(image=img, labels=labels)
        assert sample.image.shape == (4, 4, 3)
        assert sample.labels.shape == (4, 4)
        assert sample.ood_mask is None

    def test_with_ood_mask(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        labels = np.zeros((4, 4), dtype=int)
        ood = np.array(
            [[True, False, False, False]] * 4,
            dtype=bool,
        )
        sample = SegmentationSample(image=img, labels=labels, ood_mask=ood)
        assert sample.ood_mask is not None
        assert sample.ood_mask.sum() == 4
