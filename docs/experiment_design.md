# Experiment Design

Datasets, candidates, and experiment composition.

## Philosophy

Experiments are defined in code. There is no configuration language, no YAML schema, no registry of experiment types. The framework provides clean building blocks — datasets, candidates, evaluation — and the experiment author composes them in a Python script. The goal is to make those scripts as simple, readable, and flexible as possible.

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
```

### Return Format

Each sample is a tuple of two tensors:

- **image** — `(C, H, W)` float tensor. Channels-first, following PyTorch convention. No normalisation, no augmentation — raw pixels scaled to `[0, 1]`. The candidate is responsible for any preprocessing it needs.
- **labels** — `(H, W)` long tensor. Per-pixel class index with three zones:
  - `[0, num_classes)` — valid in-distribution class labels
  - `[num_classes, ignore_index)` — out-of-distribution pixels (may preserve original class identity)
  - `ignore_index` — pixels excluded from all evaluation (typically 255)

### OoD Encoding

OoD status is encoded directly in the label tensor rather than as a separate mask. A pixel is OoD if `label >= num_classes` and `label != ignore_index`. This convention:

- Matches the torch-uncertainty convention (used by MUAD, etc.), so their segmentation OoD metrics (`SegmentationBinaryAUROC`, `SegmentationFPR95`, etc.) work with no translation.
- Handles "extra classes" naturally — a dataset with 15 known classes and additional anomaly classes just sets `num_classes=15` and leaves anomaly labels at their original indices.
- Eliminates the collation problem — no `None` vs tensor mismatch in DataLoader batching.
- Preserves OoD class identity — you can analyse which OoD class was hardest to detect because the original index is retained.

At evaluation time, the benchmark runner derives binary OoD targets from this convention: `ood = (labels >= num_classes) & (labels != ignore_index)`.

### Design Decisions

**PyTorch-native.** The dataset is a `torch.utils.data.Dataset` subclass returning tensors. This means standard PyTorch `DataLoader` works directly — batching, shuffling, parallel loading, pinned memory all come for free with no adapter layer.

**A dataset is a split.** `CityscapesDataset(root, split='train')` and `CityscapesDataset(root, split='test')` are two separate dataset objects. There is no container that holds both. The experiment composes them explicitly. This is simpler, and it composes naturally: if you want to train on Cityscapes train and evaluate on BDD100K test, those are just two independent objects passed to different functions.

**No transforms.** The dataset provides raw, unprocessed data. Augmentation, normalisation, resizing — all of that is the candidate's responsibility. Different candidates need different preprocessing; the dataset should not impose any.

**Metadata is required.** `num_classes`, `class_names`, and `ignore_index` are properties on the dataset, not external configuration. The candidate and evaluation runner need to know these, and they should come from the dataset. Because `SegmentationDataset` is a base class (not a protocol), metadata propagates naturally through dataset modifiers that subclass it.

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
select_classes(Subset(cityscapes_train, indices), keep=["road", "car", "person"])
```

### Standard PyTorch Modifiers

Several common operations map directly to PyTorch's built-in dataset utilities. Because these don't change the class structure, they inherit metadata from the wrapped dataset unchanged. We provide thin subclasses that propagate `num_classes`, `class_names`, and `ignore_index` from the source dataset.

**Subsample** — random subset of fixed size. Wraps `torch.utils.data.Subset`.

```python
small = Subset(dataset, n=500, seed=42)
# small.num_classes == dataset.num_classes (propagated)
```

**Combine** — concatenates multiple datasets. Wraps `torch.utils.data.ConcatDataset`. All input datasets must share the same class set.

```python
combined = ConcatDataset([cityscapes_train, bdd100k_train])
# combined.num_classes == cityscapes_train.num_classes (validated, propagated)
```

**Filter** — removes samples based on a predicate. Computes kept indices at construction, then wraps `torch.utils.data.Subset`.

```python
# Keep only samples that have at least 10% valid pixels
filtered = filter_samples(dataset, predicate=lambda img, lbl: (lbl != 255).float().mean() > 0.1)
```

### Custom Modifiers

These modify the label space and must update metadata accordingly. They subclass `SegmentationDataset`, wrap the source dataset, and transform labels in `__getitem__`.

#### `select_classes`

Keeps only the specified classes. Remaining classes are remapped to contiguous indices `[0, n)`. Non-selected class pixels get labels `>= n` (preserving their original identity as OoD), following the OoD encoding convention.

This serves both training and evaluation in OoD experiments:
- **Training**: the candidate sees a reduced-class dataset and trains a model with `num_classes` outputs. OoD pixels are excluded from the loss via standard `label >= num_classes` masking.
- **Evaluation**: segmentation metrics use the remapped labels (OoD pixels excluded). OoD detection metrics derive binary targets from `label >= num_classes`.

```python
full_train = CityscapesDataset("./data", split="train")
reduced_train = select_classes(full_train, keep=["road", "sidewalk", "building", ...])
# reduced_train.num_classes == 16  (if 3 classes were removed from 19)
# reduced_train.class_names == ("road", "sidewalk", "building", ...)
# sample labels: kept classes remapped to [0, 16), others → >= 16
```

| | |
|---|---|
| **Input** | Dataset, list of class names or indices to keep |
| **`num_classes`** | Number of kept classes |
| **`class_names`** | Names of kept classes |
| **Labels** | Kept classes remapped to contiguous `[0, n)`, others to `>= n` |

#### `remap_classes`

Applies an explicit class index mapping. For non-standard remapping that `select_classes` doesn't cover.

```python
remapped = remap_classes(dataset, mapping={0: 0, 1: 0, 2: 1, 3: 1, 4: 2})
# Merges classes 0+1 → 0, 2+3 → 1, 4 → 2
# remapped.num_classes == 3
```

| | |
|---|---|
| **Input** | Dataset, mapping dict `{old_index: new_index}` |
| **`num_classes`** | Number of distinct values in mapping |
| **Labels** | Remapped per the dict, unmapped classes → `ignore_index` |

### Laziness Model

All modifiers are lazy on data: `__getitem__` delegates to the wrapped dataset and transforms per-sample on access. No pixel data is copied at construction time.

Some modifiers need to compute an index mapping at construction (e.g. `filter_samples` needs to scan to find which indices to keep). This is a lightweight O(n) scan that stores a list of integers, not a copy of the data.

## DataLoaders

A `torch.utils.data.DataLoader` wraps a Dataset and handles batching, shuffling, parallel loading, and memory management. The dataset provides individual samples via `__getitem__`; the DataLoader groups them into batched tensors ready for the GPU.

In our design, DataLoaders are constructed in two places:

- **Training** — the candidate constructs its own DataLoader internally. It controls batch size, shuffle, num_workers — these are training decisions that affect results.
- **Evaluation** — `run_benchmark` constructs the DataLoader. Batch size is a system/memory concern, not a method decision. The candidate never sees the DataLoader, just the batched tensors it produces.

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
        extends it) with batch dimensions. The framework inspects which
        fields are populated to determine compatible evaluation tests."""
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
- **What to evaluate** — the framework matches output fields to compatible tests.
- **Where to save state** — the experiment decides the path.
- **Evaluation data loading** — the benchmark runner owns the eval DataLoader.

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

An experiment is a Python script that composes datasets, candidates, and evaluation. The framework provides building blocks; the experiment author decides how to wire them.

### Example: Standard Benchmark

Train all candidates on Cityscapes train, evaluate on Cityscapes test.

```python
from segspicious.datasets import CityscapesDataset
from segspicious.candidates import SoftmaxCandidate, EnsembleCandidate, MCDropoutCandidate
from segspicious.evaluation import run_benchmark

train_data = CityscapesDataset("./data/cityscapes", split="train")
test_data  = CityscapesDataset("./data/cityscapes", split="test")

candidates = [
    SoftmaxCandidate(name="softmax_r50",  backbone="resnet50", epochs=100),
    EnsembleCandidate(name="ensemble_r50", backbone="resnet50", n_members=5, epochs=100),
    MCDropoutCandidate(name="mcdrop_r50",  backbone="resnet50", n_samples=20, epochs=100),
]

for c in candidates:
    c.train(train_data)

results = run_benchmark(candidates, test_data)
results.print_table()
```

### Example: Cross-Dataset Generalisation

Train on Cityscapes, evaluate on BDD100K to measure domain shift.

```python
train_data = CityscapesDataset("./data/cityscapes", split="train")
bdd_test   = BDD100KDataset("./data/bdd100k", split="test")

candidates = [
    SoftmaxCandidate(name="softmax_r50", backbone="resnet50", epochs=100),
    EnsembleCandidate(name="ensemble_r50", backbone="resnet50", n_members=5, epochs=100),
]

for c in candidates:
    c.train(train_data)

results = run_benchmark(candidates, bdd_test)
results.print_table()
```

### Example: OoD Detection with Held-Out Classes

Train on a reduced class set, evaluate OoD detection on held-out classes.

```python
from segspicious.datasets import CityscapesDataset, select_classes

all_classes = CityscapesDataset.CLASS_NAMES
ood_classes = ("motorcycle", "bicycle", "train")
id_classes  = tuple(c for c in all_classes if c not in ood_classes)

train_data = select_classes(CityscapesDataset("./data/cityscapes", split="train"), keep=id_classes)
eval_data  = select_classes(CityscapesDataset("./data/cityscapes", split="val"),   keep=id_classes)
# eval_data.num_classes == 16
# eval_data labels: kept classes remapped to [0, 16), motorcycle/bicycle/train → >= 16
# Segmentation metrics exclude OoD pixels automatically (label >= num_classes)
# OoD detection metrics derive binary targets: label >= num_classes → OoD

candidates = [
    SoftmaxCandidate(name="softmax_r50", backbone="resnet50", epochs=100),
    EnsembleCandidate(name="ensemble_r50", backbone="resnet50", n_members=5, epochs=100),
    DDUCandidate(name="ddu_r50", backbone="resnet50", epochs=100),
]

for c in candidates:
    c.train(train_data)

results = run_benchmark(candidates, eval_data)
results.print_table()
```

### Example: Train Once, Evaluate Multiple Scenarios

```python
train_data = CityscapesDataset("./data/cityscapes", split="train")

candidates = [...]
for c in candidates:
    c.train(train_data)
    c.save(Path(f"./checkpoints/{c.name}"))

# Scenario 1: in-distribution test performance
id_results = run_benchmark(candidates, CityscapesDataset("./data/cityscapes", split="val"))

# Scenario 2: domain shift
shift_results = run_benchmark(candidates, BDD100KDataset("./data/bdd100k", split="val"))

# Scenario 3: reduced dataset to test data efficiency
small_train = Subset(train_data, n=500, seed=42)
for c in candidates:
    c_small = type(c)(**c.config)  # fresh copy, same config
    c_small.train(small_train)
# ...evaluate c_small...
```

## Relationship to Evaluation

The evaluation framework (see `segmentation_design.md`, `uq_design.md`) uses torchmetrics and torch-uncertainty.metrics, which accumulate results across batches via `.update()` / `.compute()`. `run_benchmark` bridges the gap:

1. Constructs a `DataLoader` from the evaluation dataset.
2. Iterates batches. For each batch:
   a. Unpacks `(images, labels)` from the DataLoader.
   b. Calls `candidate.predict(images)` → batched `SegmentationOutput` or `UncertaintyOutput`.
   c. Derives ground truth masks from labels: `ood = (labels >= num_classes) & (labels != ignore_index)`, `ignore = labels == ignore_index`.
   d. Calls `.update(output, ground_truth)` on all compatible metrics.
3. Calls `.compute()` on each metric to get final aggregated results.
4. Returns the results matrix.

The evaluation metrics themselves are stateless between runs — they don't know about datasets or candidates. They accumulate `(output, ground_truth)` pairs batch by batch and compute aggregate statistics at the end.
