"""Tests for segmentation metrics."""

import torch

from segspicious.metrics import AccuracyResult, IoUResult, IoU, PixelAccuracy
from segspicious.outputs import SegmentationOutput


class TestIoU:
    def test_perfect_prediction(self):
        labels = torch.randint(0, 4, (2, 8, 8))
        output = SegmentationOutput(prediction=labels.clone())

        m = IoU(num_classes=4)
        m.update(output, labels)
        result = m.compute()

        assert isinstance(result, IoUResult)
        assert result.mean_iou == 1.0
        assert all(v == 1.0 for v in result.per_class_iou)

    def test_wrong_prediction(self):
        labels = torch.zeros(2, 8, 8, dtype=torch.long)
        preds = torch.ones(2, 8, 8, dtype=torch.long)
        output = SegmentationOutput(prediction=preds)

        m = IoU(num_classes=4)
        m.update(output, labels)
        result = m.compute()

        assert result.mean_iou < 1.0
        assert result.per_class_iou[0] == 0.0  # class 0: no TP
        assert result.per_class_iou[1] == 0.0  # class 1: no TP either

    def test_ignore_index_excluded(self):
        labels = torch.zeros(1, 4, 4, dtype=torch.long)
        labels[0, :2, :] = 255  # half ignored
        preds = torch.zeros(1, 4, 4, dtype=torch.long)
        preds[0, 2:, :] = 1  # wrong on non-ignored half

        m = IoU(num_classes=2, ignore_index=255)
        m.update(SegmentationOutput(prediction=preds), labels)
        result = m.compute()

        # Only the non-ignored half matters: all zeros predicted as 1 → IoU for class 0 is 0
        assert result.mean_iou < 1.0

    def test_per_class_shape(self):
        m = IoU(num_classes=7)
        labels = torch.randint(0, 7, (2, 8, 8))
        m.update(SegmentationOutput(prediction=labels), labels)
        assert len(m.compute().per_class_iou) == 7

    def test_multi_batch_accumulation(self):
        m = IoU(num_classes=3)

        labels1 = torch.randint(0, 3, (4, 8, 8))
        labels2 = torch.randint(0, 3, (4, 8, 8))
        m.update(SegmentationOutput(prediction=labels1), labels1)
        m.update(SegmentationOutput(prediction=labels2), labels2)

        assert m.compute().mean_iou == 1.0

    def test_reset(self):
        m = IoU(num_classes=3)
        labels = torch.randint(0, 3, (2, 8, 8))
        m.update(SegmentationOutput(prediction=labels), labels)
        m.reset()

        # After reset, feed different data
        labels2 = torch.zeros(2, 8, 8, dtype=torch.long)
        preds2 = torch.ones(2, 8, 8, dtype=torch.long)
        m.update(SegmentationOutput(prediction=preds2), labels2)

        # Should reflect only the second batch (all wrong)
        assert m.compute().mean_iou < 1.0


class TestPixelAccuracy:
    def test_perfect_prediction(self):
        labels = torch.randint(0, 4, (2, 8, 8))
        output = SegmentationOutput(prediction=labels.clone())

        m = PixelAccuracy(num_classes=4)
        m.update(output, labels)
        result = m.compute()

        assert isinstance(result, AccuracyResult)
        assert result.pixel_accuracy == 1.0
        assert result.mean_class_accuracy == 1.0

    def test_all_wrong(self):
        labels = torch.zeros(2, 8, 8, dtype=torch.long)
        preds = torch.ones(2, 8, 8, dtype=torch.long)
        output = SegmentationOutput(prediction=preds)

        m = PixelAccuracy(num_classes=4)
        m.update(output, labels)

        assert m.compute().pixel_accuracy == 0.0

    def test_half_correct(self):
        labels = torch.zeros(1, 2, 4, dtype=torch.long)
        preds = torch.zeros(1, 2, 4, dtype=torch.long)
        preds[0, 1, :] = 1  # bottom row wrong

        m = PixelAccuracy(num_classes=2)
        m.update(SegmentationOutput(prediction=preds), labels)

        assert m.compute().pixel_accuracy == 0.5

    def test_ignore_index_excluded(self):
        labels = torch.zeros(1, 2, 2, dtype=torch.long)
        labels[0, 0, 0] = 255  # one pixel ignored
        preds = torch.zeros(1, 2, 2, dtype=torch.long)

        m = PixelAccuracy(num_classes=2, ignore_index=255)
        m.update(SegmentationOutput(prediction=preds), labels)

        # 3 valid pixels, all correct
        assert m.compute().pixel_accuracy == 1.0

    def test_multi_batch_accumulation(self):
        m = PixelAccuracy(num_classes=3)

        labels1 = torch.randint(0, 3, (4, 8, 8))
        labels2 = torch.randint(0, 3, (4, 8, 8))
        m.update(SegmentationOutput(prediction=labels1), labels1)
        m.update(SegmentationOutput(prediction=labels2), labels2)

        assert m.compute().pixel_accuracy == 1.0

    def test_reset(self):
        m = PixelAccuracy(num_classes=3)
        labels = torch.randint(0, 3, (2, 8, 8))
        m.update(SegmentationOutput(prediction=labels), labels)
        m.reset()

        labels2 = torch.zeros(2, 8, 8, dtype=torch.long)
        preds2 = torch.ones(2, 8, 8, dtype=torch.long)
        m.update(SegmentationOutput(prediction=preds2), labels2)

        assert m.compute().pixel_accuracy == 0.0
