# Metrics API Reference

What we're using, how to call it, gotchas.

## torchmetrics

### MulticlassJaccardIndex (IoU)

```python
from torchmetrics.classification import MulticlassJaccardIndex

# Per-class IoU
m = MulticlassJaccardIndex(num_classes=N, ignore_index=255, average='none')
m.update(preds, target)  # preds: (B,H,W) long, target: (B,H,W) long
per_class_iou = m.compute()  # (N,)

# mIoU
m = MulticlassJaccardIndex(num_classes=N, ignore_index=255, average='macro')
m.update(preds, target)
miou = m.compute()  # scalar
```

### MulticlassAccuracy

```python
from torchmetrics.classification import MulticlassAccuracy

# Pixel accuracy
m = MulticlassAccuracy(num_classes=N, ignore_index=255, average='micro')
m.update(preds, target)  # preds: (B,H,W) long, target: (B,H,W) long
pixel_acc = m.compute()  # scalar

# Mean class accuracy
m = MulticlassAccuracy(num_classes=N, ignore_index=255, average='macro')
m.update(preds, target)
mean_class_acc = m.compute()  # scalar
```

### MulticlassCalibrationError (ECE)

```python
from torchmetrics.classification import MulticlassCalibrationError

m = MulticlassCalibrationError(num_classes=N, n_bins=15, norm='l1')
m.update(probs, target)  # probs: (B,C,H,W) float, target: (B,H,W) long
ece = m.compute()  # scalar
```

Handles spatial dimensions natively — no flattening needed.

## torch-uncertainty

### BrierScore

```python
from torch_uncertainty.metrics.classification import BrierScore

m = BrierScore(num_classes=N)
m.update(probs, target)  # probs: (B*H*W, C) float, target: (B*H*W,) long
brier = m.compute()  # scalar
```

**Gotcha: no spatial dims.** Expects `(batch, num_classes)`. Flatten before calling:

```python
B, C, H, W = probs.shape
m.update(probs.permute(0, 2, 3, 1).reshape(-1, C), target.reshape(-1))
```

### CategoricalNLL

```python
from torch_uncertainty.metrics.classification import CategoricalNLL

m = CategoricalNLL(reduction='mean')
m.update(probs, target)  # probs: (B*H*W, C) float, target: (B*H*W,) long
nll = m.compute()  # scalar
```

**Gotcha: no spatial dims.** Same flattening as BrierScore. Technically accepts `(B, C, H, W)` without error, but produces wrong values (sums over spatial dims instead of averaging).

### Segmentation OoD Metrics

```python
from torch_uncertainty.metrics.segmentation import (
    SegmentationBinaryAUROC,
    SegmentationBinaryAveragePrecision,
    SegmentationFPR95,
)

m = SegmentationBinaryAUROC()
m.update(scores, target)  # scores: (B,H,W) float, target: (B,H,W) binary long
auroc = m.compute()  # scalar
```

All three share the same API. Scores are any uncertainty/OoD score (higher = more OoD). Target is binary: 0 = ID, 1 = OoD.

**Image-averaged**: AUROC/FPR95/AP computed per image then averaged. This is the convention in dense OoD-detection literature.

Derive binary targets from our label convention:
```python
ood_target = ((labels >= num_classes) & (labels != ignore_index)).long()
```

### AURC / AUGRC (Selective Prediction)

```python
from torch_uncertainty.metrics.classification import AURC, AUGRC

m = AURC()
m.update(probs, target)  # probs: (B*H*W, C) float, target: (B*H*W,) long
aurc = m.compute()  # scalar
```

Expects class probabilities, not raw uncertainty scores. Internally derives confidence as `probs.max(-1)` and errors as `probs.argmax(-1) != target`.

**Gotcha: no spatial dims.** Flatten before calling (same as BrierScore/NLL).
