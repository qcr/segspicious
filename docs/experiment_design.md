# Experiment Design

Datasets, candidates, and experiment composition.

## Philosophy

Experiments are defined in code. There is no configuration language, no YAML schema, no registry of experiment types. The library provides clean building blocks — datasets, modifiers, output types, metrics, and the candidate protocol — and the experiment author composes them in a Python script. The experiment script owns the eval loop: it constructs a DataLoader, iterates batches, calls `predict()`, feeds results into metrics, and prints the output. This keeps the library focused on composable primitives and lets experiment scripts evolve freely without premature abstraction.

## SegmentationDataset

The dataset base class. Extends `torch.utils.data.Dataset` to add required segmentation metadata. Every dataset returns `(image, labels)` tensor pairs.

```python
class SegmentationDataset(torch.utils.data.Dataset):
    @property
    def num_classes(self) -> int: ...

    @property
    def class_names(self) -> tuple[str, ...]: ...

    @property
    def ignore_index(self) -> int: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: ...

    # Derived (free for all subclasses):
    @property
    def all_class_names(self) -> tuple[str, ...]: ...  # defaults to class_names

    @property
    def ood_class_names(self) -> tuple[str, ...]: ...   # all_class_names[num_classes:]

    @property
    def num_ood_classes(self) -> int: ...               # len(all_class_names) - num_classes

    @property
    def has_ood_classes(self) -> bool: ...
```

### Return Format

Each sample is a tuple of two tensors:

- **image** — `(C, H, W)` float tensor. Channels-first, following PyTorch convention. No normalisation, no augmentation — raw pixels scaled to `[0, 1]`. The candidate is responsible for any preprocessing it needs.
- **labels** — `(H, W)` long tensor. Per-pixel class index with three zones:
  - `[0, num_classes)` — valid in-distribution class labels
  - `[num_classes, num_classes + num_ood_classes)` — out-of-distribution class labels (preserving class identity)
  - `ignore_index` — pixels excluded from all evaluation (typically 255)

### OoD Encoding

OoD status is encoded directly in the label tensor rather than as a separate mask. A pixel is OoD if `label >= num_classes` and `label != ignore_index`. This convention:

- Matches the torch-uncertainty convention (used by MUAD, etc.), so their segmentation OoD metrics (`SegmentationBinaryAUROC`, `SegmentationFPR95`, etc.) work with no translation.
- Handles "extra classes" naturally — a dataset with 15 known classes and additional anomaly classes just sets `num_classes=15` and leaves anomaly labels at their original indices.
- Eliminates the collation problem — no `None` vs tensor mismatch in DataLoader batching.
- Preserves OoD class identity — you can analyse which OoD class was hardest to detect because the original index is retained.

At evaluation time, the experiment script derives binary OoD targets from this convention: `ood = (labels >= num_classes) & (labels != ignore_index)`.

### Design Decisions

**PyTorch-native.** The dataset is a `torch.utils.data.Dataset` subclass returning tensors. This means standard PyTorch `DataLoader` works directly — batching, shuffling, parallel loading, pinned memory all come for free with no adapter layer.

**A dataset is a split.** `CityscapesDataset(root, split='train')` and `CityscapesDataset(root, split='test')` are two separate dataset objects. There is no container that holds both. The experiment composes them explicitly. This is simpler, and it composes naturally: if you want to train on Cityscapes train and evaluate on BDD100K test, those are just two independent objects passed to different functions.

**No transforms.** The dataset provides raw, unprocessed data. Augmentation, normalisation, resizing — all of that is the candidate's responsibility. Different candidates need different preprocessing; the dataset should not impose any.

**Metadata is required.** `num_classes`, `class_names`, and `ignore_index` are abstract properties on the dataset, not external configuration. The candidate and evaluation runner need to know these, and they should come from the dataset. `all_class_names`, `ood_class_names`, `num_ood_classes`, and `has_ood_classes` are derived automatically. `all_class_names` defaults to `class_names` (no OoD) and is overridden by modifiers like `mark_as_ood` that introduce OoD classes. Because `SegmentationDataset` is a base class (not a protocol), metadata propagates naturally through dataset modifiers that subclass it.

### Concrete Datasets

Concrete dataset classes wrap existing data sources (torchvision datasets, image directories, HuggingFace datasets, etc.) and extend the base class:

```python
class CityscapesDataset(SegmentationDataset):
    """Wraps torchvision.datasets.Cityscapes."""

    def __init__(self, root: str, split: str = "train"):
        self._dataset = torchvision.datasets.Cityscapes(
            root, split=split, mode="fine", target_type="semantic"
        )

    @property
    def num_classes(self) -> int:
        return 19

    @property
    def class_names(self) -> tuple[str, ...]:
        return ("road", "sidewalk", "building", ...)

    @property
    def ignore_index(self) -> int:
        return 255

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img, seg = self._dataset[index]
        image = to_float_tensor(img)          # → (C, H, W) float
        labels = remap_cityscapes_ids(seg)    # → (H, W) long
        return image, labels
```

Adding a new dataset means extending `SegmentationDataset`. Most implementations are thin wrappers.

## Dataset Modifiers

Modifiers take a `SegmentationDataset` and return a new `SegmentationDataset`. They compose: the output of one modifier is valid input to another.

```
mark_as_ood(subset(cityscapes_train, n=500, seed=42), classes=["motorcycle", "bicycle"])
```

### Filter-Based Modifiers

Several operations are pure filters — they change which samples are included but don't modify labels or class metadata. They inherit metadata from the wrapped dataset unchanged, propagating `num_classes`, `all_class_names`, and `ignore_index`.

**`subset`** — random or explicit subset.

```python
small = subset(dataset, n=500, seed=42)
# small.num_classes == dataset.num_classes (propagated)

explicit = subset(dataset, indices=[0, 3, 7])
```

**`concat_datasets`** — concatenates multiple datasets. All input datasets must share the same class set (`all_class_names` and `ignore_index` must match).

```python
combined = concat_datasets([cityscapes_train, bdd100k_train])
# combined.num_classes == cityscapes_train.num_classes (validated, propagated)
```

**`filter_samples`** — removes samples based on a predicate. Scans the dataset at construction to compute kept indices.

```python
# Keep only samples that have at least 10% valid pixels
filtered = filter_samples(dataset, predicate=lambda img, lbl: (lbl != 255).float().mean() > 0.1)
```

**`select_classes`** — keeps only samples containing at least one pixel of the specified classes. Pure filter, no label remapping.

```python
# Keep only images that contain road or car pixels
with_road_or_car = select_classes(dataset, classes=["road", "car"])
```

**`hold_out_classes`** — removes samples containing any pixel of the specified classes. Inverse of `select_classes`.

```python
# Remove any image containing motorcycle pixels
no_motorcycle = hold_out_classes(dataset, classes=["motorcycle"])
```

**`hold_out_ood`** — removes samples containing any out-of-distribution pixel (label `>= num_classes` and `!= ignore_index`). Typically used after `mark_as_ood` to create clean training sets.

```python
train_ds = hold_out_ood(mark_as_ood(full_train, classes=ood_classes))
```

### Label-Remapping Modifiers

These modify the label space and must update metadata accordingly. They use a `_RemappedDataset` that applies a label lookup table in `__getitem__`.

#### `mark_as_ood`

Marks specified in-distribution classes as out-of-distribution. Keeps all samples. Remaining ID classes are remapped to contiguous `[0, n)`. Marked classes get OoD labels `>= n`. Existing OoD classes and `ignore_index` are preserved.

This serves both training and evaluation in OoD experiments:
- **Evaluation**: the eval dataset has OoD pixels with labels `>= num_classes`. Segmentation metrics exclude them. OoD detection metrics derive binary targets from `label >= num_classes`.
- **Training**: apply `hold_out_ood()` after `mark_as_ood()` to remove samples with OoD pixels, giving a clean training set with the same metadata.

```python
full_train = CityscapesDataset("./data", split="train")
ood_classes = ["motorcycle", "bicycle", "train"]

# For evaluation (keeps all samples, OoD pixels labelled >= num_classes)
eval_ds = mark_as_ood(full_val, classes=ood_classes)
# eval_ds.num_classes == 16
# eval_ds.class_names == ("road", "sidewalk", "building", ...)
# eval_ds.ood_class_names == ("motorcycle", "bicycle", "train")

# For training (same relabelling, then filter out OoD samples)
train_ds = hold_out_ood(mark_as_ood(full_train, classes=ood_classes))
# train_ds.num_classes == 16 (same metadata as eval)
```

| | |
|---|---|
| **Input** | Dataset, list of class names or indices to mark as OoD |
| **`num_classes`** | Number of remaining ID classes |
| **`class_names`** | Names of remaining ID classes |
| **`ood_class_names`** | Names of newly-OoD classes + any existing OoD classes |
| **Labels** | ID classes remapped to `[0, n)`, marked classes to `[n, n+k)` |

#### `remap_classes`

Applies an explicit class remapping by name. Multiple old classes mapping to the same new name are merged. Unmapped ID classes are preserved as ID with shifted indices. Existing OoD classes are preserved. No class changes its ID/OoD/ignore status.

```python
remapped = remap_classes(dataset, mapping={
    "road": "ground", "sidewalk": "ground",
    "car": "vehicle", "truck": "vehicle",
    "building": "structure",
})
# Mapped classes get new indices in order of first appearance.
# Unmapped ID classes stay ID, shifted after mapped ones.
```

| | |
|---|---|
| **Input** | Dataset, mapping dict `{old_name: new_name}` |
| **`num_classes`** | Number of distinct new names + unmapped ID classes |
| **`class_names`** | New names (first-appearance order) + unmapped class names |
| **Labels** | Remapped per the dict, unmapped ID classes shifted |

### Laziness Model

All modifiers are lazy on data: `__getitem__` delegates to the wrapped dataset and transforms per-sample on access. No pixel data is copied at construction time.

Some modifiers need to compute an index mapping at construction (e.g. `filter_samples` needs to scan to find which indices to keep). This is a lightweight O(n) scan that stores a list of integers, not a copy of the data.

## DataLoaders

A `torch.utils.data.DataLoader` wraps a Dataset and handles batching, shuffling, parallel loading, and memory management. The dataset provides individual samples via `__getitem__`; the DataLoader groups them into batched tensors ready for the GPU.

In our design, DataLoaders are constructed in two places:

- **Training** — the candidate constructs its own DataLoader internally. It controls batch size, shuffle, num_workers — these are training decisions that affect results.
- **Evaluation** — the experiment script constructs the DataLoader. Batch size is a system/memory concern, not a method decision. The candidate never sees the DataLoader, just the batched tensors it produces.

## Candidate Lifecycle

A candidate is the unit of comparison (see `segmentation_design.md`, `uq_design.md`). The lifecycle is: **construct → train → save → load → predict → evaluate**.

```python
class Candidate(Protocol):
    @property
    def name(self) -> str:
        """Identifier for results tables and saved state."""
        ...

    def train(self, dataset: SegmentationDataset) -> None:
        """Train on the given dataset. The candidate owns its full training
        procedure: architecture, optimiser, schedule, augmentation, epochs,
        DataLoader construction, everything. The experiment only provides data."""
        ...

    def predict(self, images: Tensor) -> SegmentationOutput:
        """Produce output for a batch of images. Receives a (B, C, H, W)
        tensor. Returns SegmentationOutput or UncertaintyOutput (which
        extends it) with batch dimensions. The experiment script inspects
        which fields are populated to determine compatible metrics."""
        ...

    def save(self, path: Path) -> None:
        """Serialise learned state to disk. Only learned state — the
        candidate's configuration (architecture, hyperparameters) lives
        in the experiment code that constructs the candidate object."""
        ...

    def load(self, path: Path) -> None:
        """Load learned state from disk. The candidate object must already
        exist (constructed with matching configuration). This mirrors
        PyTorch's model.load_state_dict(torch.load(path)) pattern."""
        ...
```

### What the Candidate Owns

- **Architecture** — the model, number of parameters, structure.
- **Training procedure** — optimiser, learning rate schedule, loss function, number of epochs, batch size.
- **Data loading for training** — constructing a `DataLoader` from the provided `SegmentationDataset`, including batch size, shuffle, and num_workers.
- **Augmentation** — training-time transforms applied to the raw data from the dataset.
- **Inference procedure** — how raw model output becomes a `SegmentationOutput` or `UncertaintyOutput`. MC Dropout does N forward passes. An ensemble averages M members. A softmax baseline does a single pass.

### What the Candidate Does Not Own

- **What data to train on** — the experiment decides.
- **What to evaluate** — the experiment script decides which metrics to run and which output fields to use.
- **Where to save state** — the experiment decides the path.
- **Evaluation data loading** — the experiment script owns the eval DataLoader.

### Pre-trained Candidates

A candidate that wraps a pre-trained model implements `train()` as a no-op (or as loading pre-trained weights). The lifecycle is the same — the experiment always calls `train()`, some candidates just don't need it.

### Cross-machine Training and Evaluation

The `save`/`load` interface enables training on one machine and evaluating on another. Only learned state is serialised. The experiment code (which constructs the candidate with its configuration) must be present on both machines.

```python
# train.py — runs on GPU machine
candidate = EnsembleCandidate(backbone="resnet50", n_members=5, epochs=100, lr=0.01)
candidate.train(train_data)
candidate.save(Path("./checkpoints/ensemble_r50"))

# eval.py — runs on a different machine
candidate = EnsembleCandidate(backbone="resnet50", n_members=5, epochs=100, lr=0.01)
candidate.load(Path("./checkpoints/ensemble_r50"))
# candidate is now ready for predict()
```

## Experiments as Code

An experiment is a Python script that composes datasets, candidates, and evaluation. The library provides building blocks; the experiment author writes the eval loop directly.

### Example: Standard Evaluation

Train a candidate on Cityscapes train, evaluate on Cityscapes test.

```python
import torch
from torch.utils.data import DataLoader
from segspicious.metrics import IoU, PixelAccuracy

# ... construct train_data, test_data, candidate ...

candidate.train(train_data)

iou = IoU(num_classes=test_data.num_classes, ignore_index=test_data.ignore_index)
acc = PixelAccuracy(num_classes=test_data.num_classes, ignore_index=test_data.ignore_index)

loader = DataLoader(test_data, batch_size=4)
for images, labels in loader:
    output = candidate.predict(images)
    iou.update(output, labels)
    acc.update(output, labels)

print(iou.compute())
print(acc.compute())
```

### Example: OoD Detection with Held-Out Classes

Train on a reduced class set, evaluate OoD detection on held-out classes.

```python
from segspicious.datasets import CityscapesDataset, mark_as_ood, hold_out_ood

ood_classes = ["motorcycle", "bicycle", "train"]

# Evaluation dataset: all samples, OoD classes relabelled >= num_classes
eval_data = mark_as_ood(
    CityscapesDataset("./data/cityscapes", split="val"),
    classes=ood_classes,
)
# eval_data.num_classes == 16
# eval_data.ood_class_names == ("motorcycle", "bicycle", "train")

# Training dataset: same relabelling, then remove samples with OoD pixels
train_data = hold_out_ood(
    mark_as_ood(
        CityscapesDataset("./data/cityscapes", split="train"),
        classes=ood_classes,
    )
)

candidate.train(train_data)

# Eval loop with both segmentation and OoD metrics
iou = IoU(num_classes=eval_data.num_classes, ignore_index=eval_data.ignore_index)
ood_auroc = OoDDetection()

loader = DataLoader(eval_data, batch_size=4)
for images, labels in loader:
    output = candidate.predict(images)
    iou.update(output, labels)
    ood_target = ((labels >= eval_data.num_classes) & (labels != eval_data.ignore_index)).long()
    if output.predictive_uncertainty is not None:
        ood_auroc.update(output.predictive_uncertainty, ood_target)

print(iou.compute())
print(ood_auroc.compute())
```

### Example: Train Once, Evaluate Multiple Scenarios

```python
from segspicious.datasets import subset

train_data = CityscapesDataset("./data/cityscapes", split="train")

candidate.train(train_data)
candidate.save(Path("./checkpoints/softmax_r50"))

# Scenario 1: in-distribution
eval_id(candidate, CityscapesDataset("./data/cityscapes", split="val"))

# Scenario 2: domain shift
eval_id(candidate, BDD100KDataset("./data/bdd100k", split="val"))
```

The eval loops are short enough to write inline or extract into local helper functions within the experiment. If patterns stabilise across many experiments, they can be promoted into the library later.

## Relationship to Metrics

The metrics (see `segmentation_design.md`, `uq_design.md`, `metrics_api.md`) use torchmetrics and torch-uncertainty.metrics, which accumulate results across batches via `.update()` / `.compute()`. The experiment script owns the eval loop:

1. Constructs a `DataLoader` from the evaluation dataset.
2. Iterates batches. For each batch:
   a. Unpacks `(images, labels)` from the DataLoader.
   b. Calls `candidate.predict(images)` → batched `SegmentationOutput` or `UncertaintyOutput`.
   c. Derives any needed ground truth (e.g. OoD binary targets from the label convention).
   d. Calls `.update()` on the relevant metrics.
3. Calls `.compute()` on each metric to get final aggregated results.

The metrics themselves are stateless between runs — they don't know about datasets or candidates. They accumulate `(output, ground_truth)` pairs batch by batch and compute aggregate statistics at the end.
