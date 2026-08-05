# Segmentation Design

Evaluating semantic segmentation performance.

## Core Concept

The unit of comparison is the **candidate**: a model trained on a specific dataset (see `model_candidate_design.md`). The experiment script runs candidates against evaluation tests using the metrics described below.

## Segmentation Output

A candidate's `predict()` method receives a batch of images as a `(B, C, H, W)` tensor and returns a `SegmentationOutput` — a dataclass with a single required field:

```
SegmentationOutput:
    prediction:  (B, H, W)  # argmax class map
```

### Field Semantics

- **prediction** — the model's best-guess class per pixel. Long tensor with values in `[0, num_classes)`.

This is the base output type. `UncertaintyOutput` extends it (see `uq_design.md`), so UQ models get segmentation evaluation for free.

## Evaluation Tests

All tests operate on the hard prediction against ground truth class labels. Metrics are computed using torchmetrics, which accumulates results across batches via `.update()` / `.compute()` calls. The experiment script owns the eval loop (see `experiment_design.md`).

### Per-class IoU / mIoU

Intersection over union per class, and the mean across classes.

| | |
|---|---|
| **Requires** | `prediction` |
| **Ground truth** | Class labels `(B, H, W)` |
| **Metrics** | Per-class IoU, mIoU |
| **Implementation** | `torchmetrics` / `torch-uncertainty` `MeanIntersectionOverUnion` |

### Pixel Accuracy

Fraction of pixels with the correct class.

| | |
|---|---|
| **Requires** | `prediction` |
| **Ground truth** | Class labels `(B, H, W)` |
| **Metrics** | Pixel accuracy |
| **Implementation** | `torchmetrics.Accuracy` |

### Mean Class Accuracy

Per-class accuracy (correct pixels / total pixels for that class), averaged across classes.

| | |
|---|---|
| **Requires** | `prediction` |
| **Ground truth** | Class labels `(B, H, W)` |
| **Metrics** | Per-class accuracy, mean class accuracy |
| **Implementation** | `torchmetrics.Accuracy` with `average="macro"` |

## Relationship to UQ

The segmentation interface evaluates prediction quality only — how good is the class map? No probabilistic metrics live here. NLL, Brier score, and ECE evaluate probability distributions, not segmentations, and belong in the UQ calibration test.

A segmentation-only model produces `SegmentationOutput` and is evaluated on the tests above. A UQ model produces `UncertaintyOutput` (which extends `SegmentationOutput`) and gets both segmentation and UQ evaluation.
