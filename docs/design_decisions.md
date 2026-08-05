# Design Decisions

Summary of key decisions made during the PyTorch-native redesign, and the reasoning behind them.

## PyTorch-Native Over Framework-Agnostic

The original design kept the dataset layer framework-agnostic (numpy arrays, custom protocol, adapter to bridge into PyTorch). We decided to go fully PyTorch-native because every model will be PyTorch-based and the framework-agnostic benefit was theoretical. This unlocks:

- `torch.utils.data.DataLoader` for batching, shuffling, parallel loading, pinned memory — no adapter needed.
- `torch.utils.data.Subset` and `ConcatDataset` for dataset operations — no custom reimplementation.
- torchmetrics and torch-uncertainty.metrics for evaluation — designed for batched `.update()` / `.compute()` accumulation.
- Consistent tensor conventions throughout: channels-first `(B, C, H, W)`, float images, long labels.

## Tensors, Not Numpy

All data is tensors from the dataset layer onwards. Images are `(C, H, W)` float (channels-first, PyTorch convention). Labels are `(H, W)` long. No numpy-to-torch conversion boundaries anywhere in the pipeline.

## Channels-First Convention

All tensor shapes follow PyTorch convention: `(B, C, H, W)` for batched images and class probabilities. The original design had `class_probs` as `(H, W, C)` which is more human-readable but non-standard. We chose convention over readability to avoid transposition bugs and to match what every PyTorch model produces natively.

## OoD Encoded in Labels, Not a Separate Mask

The original design had a separate `ood_mask: np.ndarray | None` field on each sample. We replaced this with a label encoding convention:

- `[0, num_classes)` — in-distribution class labels
- `[num_classes, ignore_index)` — out-of-distribution pixels
- `ignore_index` (typically 255) — pixels excluded from all evaluation

This was driven by looking at how torch-uncertainty handles segmentation OoD detection. Their `SegmentationRoutine` derives OoD masks from `target >= num_classes` at eval time. Their metrics (`SegmentationBinaryAUROC`, `SegmentationFPR95`, `SegmentationBinaryAveragePrecision`) expect `(scores, binary_labels)` where binary labels are derived from this convention.

Benefits:
- **Zero impedance mismatch** with torch-uncertainty metrics.
- **Handles extra classes naturally** — a dataset with anomaly classes beyond the known set just leaves them at their original indices. No remapping or mask construction needed.
- **Preserves OoD class identity** — the original class index is retained, so you can analyse which OoD class was hardest to detect. A binary mask loses this.
- **Eliminates collation issues** — no `None` vs tensor problem in DataLoader batching. Every sample is just `(image, labels)`, two tensors, always present.
- **Composes with `mark_as_ood`** — marking classes as OoD just leaves them at indices `>= num_classes`, which is exactly the convention the label encoding expects. No special logic needed.

## Dataset as Base Class, Not Protocol

The original design used a Protocol for `SegmentationDataset`. We switched to a base class extending `torch.utils.data.Dataset` for one reason: **metadata propagation through modifiers**.

PyTorch's `Subset` and `ConcatDataset` don't know about `num_classes`, `class_names`, or `ignore_index`. With a Protocol, every modifier and every stdlib wrapper would silently lose metadata. With a base class, we can provide thin subclasses of `Subset`/`ConcatDataset` that propagate these properties from the wrapped dataset.

## Batched Predict, Single-Image No More

The original `predict(image: np.ndarray) → SegmentationOutput` was single-image. We switched to `predict(images: Tensor) → SegmentationOutput` operating on `(B, C, H, W)` batches because:

- GPU inference is vastly more efficient in batches.
- The eval loop already iterates a DataLoader producing batches — calling predict per-image inside that loop is the anti-pattern DataLoader exists to solve.
- torchmetrics are designed for batched `.update()` calls — the batch flows straight from predict into metric accumulation with no per-sample loop.

## Train Gets a Dataset, Predict Gets Tensors

There's an asymmetry in the model interface:

- **`train(dataset: SegmentationDataset)`** — the model receives a Dataset and constructs its own DataLoader internally. This is because batch size, shuffle strategy, augmentation, and sampler are all training decisions that affect results. The model must own them.
- **`predict(images: Tensor)`** — the model receives raw batched tensors. The experiment script owns the eval DataLoader because batch size for inference is a system/memory concern, not a method decision. The model just processes whatever batch it receives.

This matches the PyTorch Lightning pattern: `training_step` owns data loading via the DataModule, while `predict_step` just receives a batch from the caller.

## Leaning on torchmetrics and torch-uncertainty.metrics

Rather than implementing metrics ourselves, we use:

- **torchmetrics** — `Accuracy`, `JaccardIndex` / `MeanIntersectionOverUnion` for segmentation. `CalibrationError` for ECE.
- **torch-uncertainty.metrics** — `SegmentationBinaryAUROC`, `SegmentationBinaryAveragePrecision`, `SegmentationFPR95` for OoD detection. `BrierScore`, `CategoricalNLL` for calibration. `AURC`, `AUGRC` for selective prediction.

These are all designed for the `.update(preds, targets)` per batch, `.compute()` at end pattern. The experiment script calls `.update()` on each batch and `.compute()` once after exhausting the DataLoader. No need to hold all predictions in memory.

The torch-uncertainty segmentation OoD metrics are **image-averaged** (AUROC computed per image then averaged), which is the convention in the dense OoD-detection literature.

## Dataset Modifiers: Custom Only Where Needed

The original design had five custom dataset modifiers. With PyTorch-native datasets, several map to stdlib-style patterns:

| Operation | Implementation |
|---|---|
| `subset` | Thin subclass wrapping index selection (explicit or random) |
| `concat_datasets` | Thin subclass wrapping `torch.utils.data.ConcatDataset` |
| `filter_samples` | Compute kept indices, then `subset` |
| `select_classes` | Filter — keeps samples containing specified classes, no remapping |
| `hold_out_classes` | Filter — removes samples containing specified classes (inverse of `select_classes`) |
| `mark_as_ood` | **Custom** — remaps labels, updates `num_classes` / `all_class_names` |
| `hold_out_ood` | Filter — removes samples with any OoD pixels (pairs with `mark_as_ood` for training) |
| `remap_classes` | **Custom** — merges/renames classes via explicit mapping |

The filter-based ones use a thin `_Subset` subclass that propagates `num_classes`, `all_class_names`, and `ignore_index` from the source dataset. The custom ones use a `_RemappedDataset` subclass that applies a label lookup table in `__getitem__`.

## Separation of OoD Marking and Filtering

OoD experiments require two distinct operations: (1) marking classes as out-of-distribution (relabelling), and (2) filtering out samples that contain OoD pixels (for training). The original design conflated these in a single `select_classes` modifier. We separated them:

- **`mark_as_ood(dataset, classes)`** — relabels specified classes as OoD (`>= num_classes`), remaps remaining ID classes to contiguous `[0, n)`. Keeps all samples. Used for both train and eval datasets.
- **`hold_out_ood(dataset)`** — removes any sample containing OoD pixels. Pure filter, no relabelling. Used after `mark_as_ood` to create clean training sets.

This separation means eval and train datasets share the same relabelling (same `mark_as_ood` call), with training getting an additional filter. The metadata (`num_classes`, `all_class_names`) is identical between them, which is essential for consistent metrics.

## Model vs Candidate

A **model** is an architecture + training recipe + inference logic (the `Model` protocol). A **candidate** is a model trained on a specific dataset — the unit of comparison in an experiment. The framework pairs them via `train()` / `load()` / `train_or_load()` functions that return a `Candidate` object.

This separation means the model class has no constructor arguments — the class *is* the configuration. Different hyperparameters = different model class. This makes models self-describing and enables the framework to derive checkpoint paths from `model.name` + `dataset.name` without any external configuration.

See `model_candidate_design.md` for the full design.

## Train/Test Split Is a Dataset Concern

Train and test are separate `SegmentationDataset` objects: `CityscapesDataset(root, split="train")` vs `CityscapesDataset(root, split="test")`. The DataLoader has no opinion about what the data is — it just batches whatever dataset you hand it. The experiment script wires datasets to models and evaluation explicitly. This makes cross-dataset experiments (train on Cityscapes, evaluate on BDD100K) natural — just two different dataset objects passed to different functions.
