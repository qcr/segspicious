# Models and Candidates

## Summary

A **Model** is an architecture + training recipe + inference logic. A **Candidate** is a model trained on a specific dataset — the unit of comparison in an experiment.

`Model` is a protocol implemented by the user. `Candidate` is a framework-level object that pairs a trained model with the dataset it was trained on.

## Model Protocol

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

The model owns everything about the method: architecture, optimiser, augmentation, schedule, epochs, inference procedure. Dataset-dependent values like `num_classes` are discovered from the dataset at train time.

Different hyperparameters = different model class. Classes can share machinery via inheritance or composition.

```python
class DeepLabV3RN50:
    """DeepLabV3 ResNet-50, SGD poly, lr=0.01, 50 epochs."""
    ...

class DeepLabV3RN50_LowLR(DeepLabV3RN50):
    """Same architecture, lr=0.001."""
    lr = 0.001
    ...
```

### Validation Data

The model receives optional `validation_data` through `train()`. It can use it for monitoring, early stopping, or best-checkpoint selection internally. The model should restore its best weights before `train()` returns. Validation data does not affect checkpoint identity.

## Candidate

A framework-level dataclass. Not user-implemented.

```python
@dataclass
class Candidate:
    model: Model
    dataset: SegmentationDataset

    def predict(self, images: Tensor) -> SegmentationOutput:
        return self.model.predict(images)

    @property
    def name(self) -> str:
        return f"{self.model.name}/{self.dataset.name}"
```

The candidate is what appears in results tables. Its name combines the model and dataset identity.

## Framework Functions

The framework manages checkpoint paths automatically, derived from model and dataset identity.

```python
def train(model, dataset, validation_data=None) -> Candidate:
    """Train the model and save a checkpoint."""

def load(model, dataset) -> Candidate:
    """Load an existing checkpoint."""

def train_or_load(model, dataset, validation_data=None) -> Candidate:
    """Load if a checkpoint exists, otherwise train and save."""
```

`train_or_load` makes re-runs instant — change the model class or dataset composition and it trains fresh, otherwise it loads from cache.

Checkpoint paths are derived from `model.name` and a hash of `dataset.name`, under a `trained/` directory. The user never specifies paths.

## Dataset Naming

Every dataset has a `name` property. Base datasets set the root name. Modifiers append suffixes.

The format is: `base_name` followed by `-modifier[params]` for each transformation.

```
wildscenes2d_train
wildscenes2d_K01_val
cityscapes_train-subset[n=100,seed=42]
cityscapes_train-mark_ood[bicycle+motorcycle]-hold_out_ood
cityscapes_train-subset[n=500,seed=0]-mark_ood[bicycle+motorcycle]-hold_out_ood
```

Concat joins with `+`:

```
cityscapes_train+bdd100k_train
```

### Modifier Name Suffixes

| Modifier | Suffix |
|---|---|
| `subset(n=100, seed=42)` | `-subset[n=100,seed=42]` |
| `subset(indices=[0,3,7])` | `-subset[indices=0+3+7]` |
| `mark_as_ood(classes=...)` | `-mark_ood[bicycle+motorcycle]` |
| `hold_out_ood()` | `-hold_out_ood` |
| `hold_out_classes(classes=...)` | `-hold_out[bicycle+motorcycle]` |
| `select_classes(classes=...)` | `-select[bicycle+motorcycle]` |
| `remap_classes(mapping=...)` | `-remap[road=ground+sidewalk=ground]` |
| `filter_samples(label=...)` | `-filter[min10pct_valid]` |
| `filter_by_labels(label=...)` | `-filter[min10pct_valid]` |
| `concat_datasets(...)` | `ds1_name+ds2_name` |

`filter_samples` and `filter_by_labels` take an arbitrary predicate with no string representation, so they require a `label` parameter from the caller.

### Implementation

Modifier functions build the name and pass it to the wrapper class. The wrapper just stores and exposes it.

```python
def mark_as_ood(dataset, classes):
    name = f"{dataset.name}-mark_ood[{'+'.join(sorted(ood_names))}]"
    return _RemappedDataset(dataset, remap, ..., name=name)
```

`SegmentationDataset` gains `name` as an abstract property. Concrete datasets implement it. Modifier wrappers receive it at construction.

## Experiment Lifecycle

```python
# experiments/ood_bench/data.py — shared by train and eval
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

# experiments/ood_bench/run.py
train_ds, val_ds = cityscapes_ood()
candidate = train_or_load(DeepLabV3RN50(), train_ds, validation_data=val_ds)

for images, labels in DataLoader(val_ds, batch_size=4):
    output = candidate.predict(images)
    iou.update(output, labels)
```

Both train and eval scripts import the same data module. Same code constructs the same dataset objects, producing the same names, resolving to the same checkpoint paths. The code is the configuration.
