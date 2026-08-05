# Experiment Design

Datasets, models, candidates, and experiment composition.

## Philosophy

Experiments are defined in code. There is no configuration language, no YAML schema, no registry of experiment types. The library provides clean building blocks — datasets, modifiers, output types, metrics, and the model protocol — and the experiment author composes them in a Python script. The experiment script owns the eval loop: it constructs a DataLoader, iterates batches, calls `predict()`, feeds results into metrics, and prints the output. This keeps the library focused on composable primitives and lets experiment scripts evolve freely without premature abstraction.

## SegmentationDataset

The dataset base class. Extends `torch.utils.data.Dataset` to add required segmentation metadata. Every dataset returns `(image, labels)` tensor pairs.

```python
class SegmentationDataset(torch.utils.data.Dataset):
    @property
    def name(self) -> str: ...

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

### Dataset Naming

Every dataset has a `name` property used for checkpoint path derivation and results identification. Base datasets set the root name (`cityscapes_train`, `wildscenes2d_K01_val`). Modifiers append suffixes describing the transformation:

```
wildscenes2d_train-subset[n=100,seed=42]-mark_ood[bicycle+motorcycle]-hold_out_ood
```

See `model_candidate_design.md` for the full naming convention.

### Return Format

Each sample is a tuple of two tensors:

- **image** — `(C, H, W)` float tensor. Channels-first, following PyTorch convention. No normalisation, no augmentation — raw pixels scaled to `[0, 1]`. The model is responsible for any preprocessing it needs.
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

**No transforms.** The dataset provides raw, unprocessed data. Augmentation, normalisation, resizing — all of that is the model's responsibility. Different models need different preprocessing; the dataset should not impose any.

**Metadata is required.** `name`, `num_classes`, `class_names`, and `ignore_index` are abstract properties on the dataset, not external configuration. The model and evaluation runner need to know these, and they should come from the dataset. `all_class_names`, `ood_class_names`, `num_ood_classes`, and `has_ood_classes` are derived automatically. `all_class_names` defaults to `class_names` (no OoD) and is overridden by modifiers like `mark_as_ood` that introduce OoD classes. Because `SegmentationDataset` is a base class (not a protocol), metadata propagates naturally through dataset modifiers that subclass it.

### Concrete Datasets

Concrete dataset classes wrap existing data sources (torchvision datasets, image directories, HuggingFace datasets, etc.) and extend the base class:

```python
class CityscapesDataset(SegmentationDataset):
    """Wraps torchvision.datasets.Cityscapes."""

    def __init__(self, root: str, split: str = "train"):
        self._split = split
        self._dataset = torchvision.datasets.Cityscapes(
            root, split=split, mode="fine", target_type="semantic"
        )

    @property
    def name(self) -> str:
        return f"cityscapes_{self._split}"

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

Modifiers take a `SegmentationDataset` and return a new `SegmentationDataset`. They compose: the output of one modifier is valid input to another. Each modifier appends to the dataset's `name`.

```python
ds = mark_as_ood(subset(cityscapes_train, n=500, seed=42), classes=["motorcycle", "bicycle"])
# ds.name == "cityscapes_train-subset[n=500,seed=42]-mark_ood[bicycle+motorcycle]"
```

### Filter-Based Modifiers

Several operations are pure filters — they change which samples are included but don't modify labels or class metadata. They inherit metadata from the wrapped dataset unchanged, propagating `num_classes`, `all_class_names`, and `ignore_index`.

**`subset`** — random or explicit subset.

```python
small = subset(dataset, n=500, seed=42)
# small.name == "cityscapes_train-subset[n=500,seed=42]"

explicit = subset(dataset, indices=[0, 3, 7])
```

**`concat_datasets`** — concatenates multiple datasets. All input datasets must share the same class set (`all_class_names` and `ignore_index` must match).

```python
combined = concat_datasets([cityscapes_train, bdd100k_train])
# combined.name == "cityscapes_train+bdd100k_train"
```

**`filter_samples`** — removes samples based on a predicate. Scans the dataset at construction to compute kept indices. Requires a `label` parameter for naming since the predicate has no string representation.

```python
filtered = filter_samples(
    dataset,
    predicate=lambda img, lbl: (lbl != 255).float().mean() > 0.1,
    label="min10pct_valid",
)
# filtered.name == "cityscapes_train-filter[min10pct_valid]"
```

**`select_classes`** — keeps only samples containing at least one pixel of the specified classes. Pure filter, no label remapping.

```python
with_road_or_car = select_classes(dataset, classes=["road", "car"])
```

**`hold_out_classes`** — removes samples containing any pixel of the specified classes. Inverse of `select_classes`.

```python
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

Dataset construction is cheap. This matters because both train and eval scripts construct the same dataset objects (the code is the configuration) — only `__getitem__` is expensive.

## DataLoaders

A `torch.utils.data.DataLoader` wraps a Dataset and handles batching, shuffling, parallel loading, and memory management. The dataset provides individual samples via `__getitem__`; the DataLoader groups them into batched tensors ready for the GPU.

In our design, DataLoaders are constructed in two places:

- **Training** — the model constructs its own DataLoader internally. It controls batch size, shuffle, num_workers — these are training decisions that affect results.
- **Evaluation** — the experiment script constructs the DataLoader. Batch size is a system/memory concern, not a method decision. The model never sees the DataLoader, just the batched tensors it produces.

## Models and Candidates

A **model** is an architecture + training recipe + inference logic. A **candidate** is a model trained on a specific dataset — the unit of comparison in an experiment. See `model_candidate_design.md` for the full design.

### Model Protocol

Models have no constructor arguments. The class is the configuration.

```python
class Model(Protocol):
    @property
    def name(self) -> str: ...

    def train(
        self,
        dataset: SegmentationDataset,
        validation_data: SegmentationDataset | None = None,
    ) -> None: ...

    def predict(self, images: Tensor) -> SegmentationOutput: ...

    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

### What the Model Owns

- **Architecture** — the model, number of parameters, structure.
- **Training procedure** — optimiser, learning rate schedule, loss function, number of epochs, batch size.
- **Data loading for training** — constructing a `DataLoader` from the provided `SegmentationDataset`, including batch size, shuffle, and num_workers.
- **Augmentation** — training-time transforms applied to the raw data from the dataset.
- **Inference procedure** — how raw model output becomes a `SegmentationOutput` or `UncertaintyOutput`. MC Dropout does N forward passes. An ensemble averages M members. A softmax baseline does a single pass.

### What the Model Does Not Own

- **What data to train on** — the experiment decides.
- **What to evaluate** — the experiment script decides which metrics to run and which output fields to use.
- **Where to save state** — the framework derives checkpoint paths from model and dataset identity.
- **Evaluation data loading** — the experiment script owns the eval DataLoader.

### Candidate

A framework-level object that pairs a trained model with the dataset it was trained on. Not user-implemented. Delegates `predict()` to the model.

```python
candidate = train_or_load(DeepLabV3RN50(), train_data, validation_data=val_data)
output = candidate.predict(images)
```

### Framework Functions

```python
train(model, dataset, validation_data=None) -> Candidate
load(model, dataset) -> Candidate
train_or_load(model, dataset, validation_data=None) -> Candidate
```

`train_or_load` loads from cache if a checkpoint exists, otherwise trains and saves. Checkpoint paths are derived from `model.name` and `dataset.name`. The user never specifies paths.

### Pre-trained Models

A model that wraps a pre-trained network implements `train()` as a no-op (or as loading pre-trained weights). The lifecycle is the same — the framework always calls `train()`, some models just don't need it.

### Cross-machine Training and Evaluation

The code is the configuration. Train and eval scripts import the same data preparation module, constructing the same dataset objects. Same code → same dataset names → same checkpoint paths.

```python
# experiments/ood_bench/data.py — shared module
def cityscapes_ood():
    ood_classes = ["motorcycle", "bicycle", "train"]
    train_ds = hold_out_ood(mark_as_ood(
        CityscapesDataset("./data/cityscapes", split=Split.TRAIN),
        classes=ood_classes,
    ))
    val_ds = mark_as_ood(
        CityscapesDataset("./data/cityscapes", split=Split.VAL),
        classes=ood_classes,
    )
    return train_ds, val_ds

# train.py — GPU machine
from experiments.ood_bench.data import cityscapes_ood
train_ds, val_ds = cityscapes_ood()
candidate = train(DeepLabV3RN50(), train_ds, validation_data=val_ds)

# eval.py — different machine
from experiments.ood_bench.data import cityscapes_ood
train_ds, val_ds = cityscapes_ood()
candidate = load(DeepLabV3RN50(), train_ds)
```

## Experiments as Code

An experiment is a Python script that composes datasets, models, and evaluation. The library provides building blocks; the experiment author writes the eval loop directly.

### Example: Standard Evaluation

```python
import torch
from torch.utils.data import DataLoader
from segspicious.metrics import IoU, PixelAccuracy

train_data = CityscapesDataset("./data/cityscapes", split=Split.TRAIN)
test_data = CityscapesDataset("./data/cityscapes", split=Split.TEST)

candidate = train_or_load(DeepLabV3RN50(), train_data)

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

```python
from segspicious.datasets import CityscapesDataset, mark_as_ood, hold_out_ood

ood_classes = ["motorcycle", "bicycle", "train"]

eval_data = mark_as_ood(
    CityscapesDataset("./data/cityscapes", split="val"),
    classes=ood_classes,
)

train_data = hold_out_ood(
    mark_as_ood(
        CityscapesDataset("./data/cityscapes", split="train"),
        classes=ood_classes,
    )
)

candidate = train_or_load(DeepLabV3RN50(), train_data)

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

### Example: Multiple Models, Same Dataset

```python
train_data = CityscapesDataset("./data/cityscapes", split="train")
val_data = CityscapesDataset("./data/cityscapes", split="val")

candidates = [
    train_or_load(DeepLabV3RN50(), train_data),
    train_or_load(EnsembleDeepLabV3(), train_data),
    train_or_load(MCDropoutDeepLabV3(), train_data),
]

for candidate in candidates:
    eval_segmentation(candidate, val_data)
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

The metrics themselves are stateless between runs — they don't know about datasets or models. They accumulate `(output, ground_truth)` pairs batch by batch and compute aggregate statistics at the end.
