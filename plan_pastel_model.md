# Plan: PASTEL DINOv2 Model + Balanced Subset Modifiers

Implement the PASTEL semantic segmentation method (frozen DINOv2 backbone + MLP head)
as a segspicious `Model`, and add two new dataset modifiers for class-balanced subset
selection.

Reference: [PASTEL: A Good Foundation is Worth Many Labels](https://arxiv.org/abs/2405.19035)
— Vödisch et al., RA-L 2025. Local implementation at `~/repos/pastel_ss_on_wildscenes/`.

## Overview

PASTEL uses a frozen DINOv2 ViT-L/14 backbone with a tiny trainable MLP head (~600K params)
for semantic segmentation. It's designed for sample-efficient training — competitive results
from as few as 20 labelled images. The method uses hard pixel mining (top-20% loss), Adam
with cosine annealing, and class-balanced sample selection.

The number of training samples is treated as a model hyperparameter: the model internally
selects a class-balanced subset from whatever dataset it receives. This avoids needing
special benchmarking scenarios with small datasets just for this model.

## Step 1: Add `coverage_subset` and `balanced_subset` modifiers

**Files:**
- Edit `segspicious/datasets/modifiers.py` — add two modifiers
- Edit `segspicious/datasets/__init__.py` — export them
- Create `segspicious/tests/test_balanced_subset.py` — tests

### `coverage_subset(dataset, n) → SegmentationDataset`

Phase 1 only: greedy set-cover that maximises class coverage.

- Uses `get_classes_present(i)` only — fast, no pixel data loaded.
- Iteratively picks the image that covers the most not-yet-represented classes.
- Breaks ties by number of total classes present (more = better).
- Deterministic (no seed needed).
- Name suffix: `-coverage_subset(n=20)`

### `balanced_subset(dataset, n) → SegmentationDataset`

Phase 1 (coverage) + Phase 2 (pixel-count balancing).

- Phase 1: identical to `coverage_subset`.
- Phase 2: fills remaining budget by picking images that contribute the most
  pixels to the least-represented class. Uses `get_labels(i)` to count pixels
  per class — only called on candidates not yet selected, only during Phase 2.
- Deterministic.
- Name suffix: `-balanced_subset(n=20)`

### Tests

- Coverage: on a `SyntheticDataset` with known class distribution, verify that
  `coverage_subset` covers all classes with minimal n.
- Balanced: verify that `balanced_subset` produces better pixel-count balance
  than random `subset` of the same size.
- Both: name suffix correctness, metadata propagation, `len()` equals n,
  error on n > len(dataset).
- Edge cases: n >= len(dataset), dataset with classes only in single images.

## Step 2: Implement `PastelDINOv2` model

**Files:**
- Create `experiments/models/pastel_dinov2.py`

### Architecture (internal `nn.Module`)

Frozen DINOv2 ViT-L/14 backbone + 4-layer MLP head (1×1 convs):

```
1024 → 300 (ReLU) → 300 (ReLU) → 200 (ReLU) → num_classes
```

Only the head is trainable (~600K params). The backbone is loaded from
`torch.hub.load("facebookresearch/dinov2", ...)` and kept in eval mode.

### Class attributes (hyperparameters)

| Attribute | Default | Notes |
|---|---|---|
| `backbone` | `"dinov2_vitl14"` | DINOv2 variant |
| `feat_dim` | `1024` | Must match backbone |
| `epochs` | `150` | From PASTEL paper |
| `batch_size` | `8` | |
| `lr` | `1e-3` | Adam learning rate |
| `crop_size` | `504` | Must be divisible by 14 |
| `hard_mining_ratio` | `0.2` | Top-k% hardest pixels |
| `num_train_samples` | `None` | `None` = use all; int = balanced subset |
| `num_workers` | `4` | DataLoader workers |
| `log_dir` | `"runs"` | TensorBoard log directory |

### `name` property

Format: `pastel-dinov2-vitl14` when `num_train_samples` is None,
or `pastel-dinov2-vitl14-n20` when set to 20.

This encodes the sample count in the model identity, so different sample
budgets produce different checkpoint paths without needing different
dataset scenarios.

### `train(dataset, validation_data=None)`

1. Discover `num_classes` from `dataset.num_classes`.
2. If `num_train_samples` is set and < `len(dataset)`, apply
   `balanced_subset(dataset, n=self.num_train_samples)` to get a
   class-balanced training subset. This is internal — the `Candidate`
   still uses the original dataset for checkpoint path derivation.
3. Build `_TrainAugDataset` wrapper with PASTEL augmentations:
   - Random resized crop to `crop_size × crop_size` (scale 0.2–1.0)
   - Random horizontal flip
   - Colour jitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
   - ImageNet normalisation
4. Build model: frozen backbone + MLP head.
5. Optimizer: `Adam(head.parameters(), lr=self.lr)`.
6. Scheduler: `CosineAnnealingLR(T_max=self.epochs)`.
7. Training loop with hard pixel mining:
   ```python
   loss = F.cross_entropy(logits, labels, ignore_index=..., reduction="none")
   k = int(self.hard_mining_ratio * loss.numel())
   loss = torch.topk(loss.reshape(-1), k).values.mean()
   ```
8. Per-epoch validation (if `validation_data` provided), tracking best
   val loss. Restore best weights before returning.
9. TensorBoard logging: train loss (step + epoch), val loss, val mIoU, lr.

### `predict(images: Tensor) → UncertaintyOutput`

1. Record original spatial size `(H, W)` from input.
2. Bilinear resize input to `(crop_size, crop_size)`.
3. ImageNet normalise.
4. Forward pass (backbone is frozen, in `torch.no_grad()`).
5. Bilinear upsample logits back to `(H, W)`.
6. Compute outputs:
   - `class_probs`: softmax over logits → `(B, C, H, W)`
   - `prediction`: argmax of class_probs → `(B, H, W)`
   - `predictive_uncertainty`: entropy `−Σ p·log(p)` → `(B, H, W)`

Returns `UncertaintyOutput` (softmax baseline — no epistemic/aleatoric/ood_score).

### `save(directory)` / `load(directory)`

Save only the MLP head weights + `num_classes` (the backbone is re-loaded
from torch.hub). Checkpoint is tiny (~2.4MB).

```python
# save
torch.save({"head_state_dict": head.state_dict(), "num_classes": num_classes},
           directory / "checkpoint.pt")

# load
checkpoint = torch.load(directory / "checkpoint.pt")
self._num_classes = checkpoint["num_classes"]
self._model = self._build_model(self._num_classes)
self._model.head.load_state_dict(checkpoint["head_state_dict"])
```

### `_TrainAugDataset`

Similar pattern to DeepLabV3's augmentation wrapper but with PASTEL-specific transforms:

- Joint image/label `RandomResizedCrop` (scale 0.2–1.0, ratio 0.75–1.333) to `crop_size`.
- Random horizontal flip.
- Colour jitter on image only.
- ImageNet normalisation on image.
- Label resize uses nearest-neighbour interpolation.

Key difference from DeepLabV3's wrapper: PASTEL uses `RandomResizedCrop` (PIL-style
crop-then-resize) rather than separate scale → pad → random-crop.

### Inheritance for variants

```python
class PastelDINOv2_ViTB(PastelDINOv2):
    backbone = "dinov2_vitb14"
    feat_dim = 768

    @property
    def name(self) -> str:
        base = "pastel-dinov2-vitb14"
        if self.num_train_samples is not None:
            return f"{base}-n{self.num_train_samples}"
        return base

class PastelDINOv2_N100(PastelDINOv2):
    num_train_samples = 100
```

## Step 3: Tests for `PastelDINOv2`

**Files:**
- Create `experiments/tests/test_pastel_dinov2.py`

Following the same pattern as `test_deeplabv3_rn50.py`:

### Protocol conformance
- `isinstance(model, Model)` passes.
- No constructor arguments.
- `name` property returns expected string.

### Class attributes as configuration
- Default hyperparameters match expected values.
- Inheritance overrides work (e.g. different backbone, different n).

### Predict/save guards
- `predict()` before `train()`/`load()` raises `AssertionError`.
- `save()` before `train()` raises `AssertionError`.

### Save/load round-trip
- Save then load produces identical predictions.
- Checkpoint contains `head_state_dict` and `num_classes`.
- Only head weights are saved (checkpoint is small).

### Train discovers num_classes
- After `train()`, `num_classes` is set from dataset.
- `predict()` works after `train()` with correct output shapes.
- Full lifecycle: train → save → load → predict.

### PASTEL-specific
- `num_train_samples` is reflected in `name`.
- When `num_train_samples` is set, training uses fewer samples than
  the full dataset (verify via a mock/spy on `balanced_subset` or
  by checking DataLoader length).

**Note:** Tests will use a `QuickPastel` subclass with `epochs=1`,
`batch_size=2`, `num_workers=0` and small `SyntheticDataset` instances
to keep tests fast. The DINOv2 backbone download is unavoidable on
first run but is cached by torch.hub thereafter.

## Step 4: Wire into exports and benchmarking

**Files:**
- Edit `experiments/models/__init__.py` — add `PastelDINOv2` export
- Edit `experiments/benchmarking/models.py` — add to `get_models()` registry

The model will appear in the benchmarking matrix alongside DeepLabV3RN50,
evaluated on all existing scenarios (wildscenes2d_subset100, wildscenes2d_full).
Since PASTEL does its own internal balanced sampling, it gets the full
training dataset from the scenario and subsets it internally.

## File change summary

| File | Action |
|---|---|
| `segspicious/datasets/modifiers.py` | Edit — add `coverage_subset`, `balanced_subset`, helpers |
| `segspicious/datasets/__init__.py` | Edit — export new modifiers |
| `segspicious/tests/test_balanced_subset.py` | Create — modifier tests |
| `experiments/models/pastel_dinov2.py` | Create — PASTEL model |
| `experiments/tests/test_pastel_dinov2.py` | Create — model tests |
| `experiments/models/__init__.py` | Edit — add export |
| `experiments/benchmarking/models.py` | Edit — add to registry |
