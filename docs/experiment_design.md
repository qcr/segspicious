# Experiment Design

Datasets, candidates, and experiment composition.

## Philosophy

Experiments are defined in code. There is no configuration language, no YAML schema, no registry of experiment types. The framework provides clean building blocks — datasets, candidates, evaluation — and the experiment author composes them in a Python script. The goal is to make those scripts as simple, readable, and flexible as possible.

## SegmentationSample

The data unit. Every dataset returns these.

```python
@dataclass
class SegmentationSample:
    image:    np.ndarray                # (H, W, C) uint8 RGB
    labels:   np.ndarray                # (H, W) int, class indices
    ood_mask: np.ndarray | None = None  # (H, W) bool, True = out-of-distribution
```

### Field Semantics

- **image** — the input image. Always numpy, always `(H, W, C)`, always RGB, always uint8. No normalisation, no augmentation — raw pixels. The candidate is responsible for any preprocessing it needs.
- **labels** — per-pixel class index. Values in `[0, num_classes)` for valid pixels, or `ignore_index` for pixels that should be excluded from both training loss and evaluation metrics.
- **ood_mask** — optional binary mask indicating which pixels are out-of-distribution. `None` by default. Set either by the dataset natively (e.g. Fishyscapes) or by a dataset modifier (e.g. marking held-out classes as OoD). The OoD detection evaluation test requires this field.

## SegmentationDataset

The dataset protocol. This is what candidates receive for training and what the evaluation runner iterates over.

```python
class SegmentationDataset(Protocol):
    @property
    def num_classes(self) -> int: ...

    @property
    def class_names(self) -> tuple[str, ...]: ...

    @property
    def ignore_index(self) -> int: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> SegmentationSample: ...
```

### Design Decisions

**Framework-agnostic.** The protocol uses numpy arrays, not torch tensors. It sits above PyTorch. A PyTorch-based candidate wraps it internally — creating a `torch.utils.data.Dataset`, adding transforms, building a `DataLoader`. That translation is trivial (a few lines) and can be provided as a utility. But the protocol itself has no PyTorch dependency.

**A dataset is a split.** `Cityscapes(root, split='train')` and `Cityscapes(root, split='test')` are two separate dataset objects. There is no container that holds both. The experiment composes them explicitly. This is simpler, and it composes naturally: if you want to train on Cityscapes train and evaluate on BDD100K test, those are just two independent objects passed to different functions.

**No transforms.** The dataset provides raw, unprocessed data. Augmentation, normalisation, resizing — all of that is the candidate's responsibility. Different candidates need different preprocessing; the dataset should not impose any.

**Metadata is required.** `num_classes`, `class_names`, and `ignore_index` are properties on the dataset, not external configuration. The candidate and evaluation runner need to know these, and they should come from the dataset.

### Concrete Datasets

Concrete dataset classes wrap existing data sources (torchvision datasets, image directories, HuggingFace datasets, etc.) and expose them through the protocol:

```python
class CityscapesDataset:
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

    def __getitem__(self, index: int) -> SegmentationSample:
        img, seg = self._dataset[index]
        return SegmentationSample(
            image=np.asarray(img),
            labels=remap_cityscapes_ids(np.asarray(seg)),
        )
```

Adding a new dataset means implementing this interface. Most implementations are thin wrappers.

## Dataset Modifiers

Modifiers take a dataset and return a new dataset satisfying the same protocol. They compose: the output of one modifier is valid input to another.

```
select_classes(subsample(cityscapes_train, n=1000), keep=["road", "car", "person"])
```

### Laziness Model

Modifiers are lazy on data: `__getitem__` delegates to the wrapped dataset and transforms per-sample on access. No pixel data is copied at construction time.

Some modifiers need to compute an index mapping at construction (e.g. `filter_samples` needs to scan to find which indices to keep). This is a lightweight O(n) scan that stores a list of integers, not a copy of the data.

### `select_classes`

Keeps only the specified classes. Remaining classes are remapped to contiguous indices `[0, n)`. All other pixels become `ignore_index`. Additionally, sets `ood_mask` on each sample: pixels belonging to non-selected classes are marked as OoD.

This serves both training and evaluation in OoD experiments:
- **Training**: the candidate sees a reduced-class dataset and trains a model with `num_classes` outputs.
- **Evaluation**: segmentation metrics use the remapped labels (non-selected pixels are ignored). OoD metrics use the `ood_mask`.

```python
full_train = CityscapesDataset("./data", split="train")
reduced_train = select_classes(full_train, keep=["road", "sidewalk", "building", ...])
# reduced_train.num_classes == 16  (if 3 classes were removed from 19)
# reduced_train.class_names == ("road", "sidewalk", "building", ...)
# sample.labels: remapped to [0, 15], others → ignore_index
# sample.ood_mask: True where original class was not in keep
```

| | |
|---|---|
| **Input** | Dataset, list of class names or indices to keep |
| **`num_classes`** | Number of kept classes |
| **`class_names`** | Names of kept classes |
| **Labels** | Remapped to contiguous `[0, n)`, others → `ignore_index` |
| **`ood_mask`** | `True` for pixels of non-selected classes |

### `remap_classes`

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

### `filter_samples`

Removes samples based on a predicate. Computes the index mapping at construction by scanning the dataset.

```python
# Keep only samples that have at least 10% valid (non-ignore) pixels
filtered = filter_samples(dataset, predicate=lambda s: (s.labels != 255).mean() > 0.1)
```

| | |
|---|---|
| **Input** | Dataset, predicate function `SegmentationSample → bool` |
| **`num_classes`** | Unchanged |
| **Construction** | Scans dataset once to build index of kept samples |

### `subsample`

Random subset of fixed size.

```python
small = subsample(dataset, n=500, seed=42)
```

| | |
|---|---|
| **Input** | Dataset, number of samples, optional seed |
| **`num_classes`** | Unchanged |
| **Construction** | Selects random indices |

### `combine`

Concatenates multiple datasets. All must share the same class set.

```python
combined = combine(cityscapes_train, bdd100k_train)
```

| | |
|---|---|
| **Input** | Two or more datasets with matching `num_classes` and `class_names` |
| **`num_classes`** | Unchanged |
| **Length** | Sum of input lengths |

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
        everything. The experiment only provides data."""
        ...

    def predict(self, image: np.ndarray) -> SegmentationOutput:
        """Produce output for a single image. Returns SegmentationOutput
        or UncertaintyOutput (which extends it). The framework inspects
        which fields are populated to determine compatible evaluation tests."""
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
- **Augmentation** — training-time transforms applied to the raw data from the dataset.
- **Inference procedure** — how raw model output becomes a `SegmentationOutput` or `UncertaintyOutput`. MC Dropout does N forward passes. An ensemble averages M members. A softmax baseline does a single pass.
- **Internal data loading** — wrapping the `SegmentationDataset` protocol into a PyTorch `DataLoader` (or whatever the framework-specific equivalent is).

### What the Candidate Does Not Own

- **What data to train on** — the experiment decides.
- **What to evaluate** — the framework matches output fields to compatible tests.
- **Where to save state** — the experiment decides the path.

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
# eval_data.ood_mask marks motorcycle/bicycle/train pixels as OoD
# eval_data.labels has OoD pixels set to ignore_index (excluded from mIoU)
# eval_data.num_classes == 16

candidates = [
    SoftmaxCandidate(name="softmax_r50", backbone="resnet50", epochs=100),
    EnsembleCandidate(name="ensemble_r50", backbone="resnet50", n_members=5, epochs=100),
    DDUCandidate(name="ddu_r50", backbone="resnet50", epochs=100),
]

for c in candidates:
    c.train(train_data)

results = run_benchmark(candidates, eval_data)
# Segmentation tests use remapped labels (16 classes, OoD pixels ignored)
# OoD detection tests use ood_mask
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
small_train = subsample(train_data, n=500, seed=42)
for c in candidates:
    c_small = type(c)(**c.config)  # fresh copy, same config
    c_small.train(small_train)
# ...evaluate c_small...
```

## Relationship to Evaluation

The evaluation framework (see `segmentation_design.md`, `uq_design.md`) operates on `(candidate_output, ground_truth)` pairs. `run_benchmark` bridges the gap:

1. Iterates over the evaluation dataset.
2. Calls `candidate.predict(sample.image)` on each sample.
3. Pairs the output with ground truth from the sample (`sample.labels`, `sample.ood_mask`).
4. Runs all compatible evaluation tests (determined by which output fields are populated and which ground truth is available).
5. Aggregates and returns the results matrix.

The evaluation tests themselves are unchanged — they don't know about datasets or candidates. They receive outputs and ground truth and compute metrics.

## PyTorch Interop

Since most candidates will use PyTorch internally, the framework provides a utility to bridge the protocol into PyTorch's data loading:

```python
class TorchDatasetAdapter(torch.utils.data.Dataset):
    """Wraps a SegmentationDataset for use with torch.utils.data.DataLoader."""

    def __init__(self, dataset: SegmentationDataset, transform=None):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]
        image = torch.from_numpy(sample.image).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(sample.labels).long()
        if self._transform:
            image, labels = self._transform(image, labels)
        return image, labels
```

This adapter (or something like it) lives in the framework as a utility. Candidates can use it or roll their own. It is not part of the protocol.
