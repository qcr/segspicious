# Implementation Plan

## Repo Structure

```
segspicious/          # reusable library — protocols, modifiers, metrics
experiments/          # concrete datasets, candidates, experiment scripts
```

segspicious defines *what things are*. experiments defines *which things*. The dependency arrow is one-way: experiments imports segspicious, never the reverse. segspicious never names a specific dataset, model architecture, or framework.

## Phase 0 — Project Setup

Set up a pixi environment for the project. pixi manages both Python and non-Python dependencies via conda-forge, with a lockfile for reproducibility.

- Initialise pixi in the repo root.
- Add core dependencies: python, numpy.
- Add dev dependencies: pytest.
- torch and torchvision are needed from Phase 2 onwards (TorchDatasetAdapter) and heavily in experiments/. Add them to the default environment from the start — they're unavoidable.
- segspicious itself should be installed in editable mode so imports work cleanly across both segspicious/ and experiments/.
- Add any further dependencies as they come up (e.g. scikit-learn for AUROC in Phase 3).

## Phase 1 — Core Types & Protocols

The foundation. Everything else depends on this.

### Outputs

```python
@dataclass
class SegmentationOutput:
    prediction: np.ndarray               # (H, W) int — argmax class map

@dataclass
class UncertaintyOutput(SegmentationOutput):
    class_probs: np.ndarray | None             # (H, W, C) — probability distribution over classes
    predictive_uncertainty: np.ndarray | None   # (H, W) — total uncertainty
    aleatoric_uncertainty: np.ndarray | None    # (H, W) — data ambiguity
    epistemic_uncertainty: np.ndarray | None    # (H, W) — model ignorance
    ood_score: np.ndarray | None                # (H, W) — input unfamiliarity
```

### Data

```python
@dataclass
class SegmentationSample:
    image: np.ndarray                    # (H, W, C) uint8 RGB
    labels: np.ndarray                   # (H, W) int — class indices
    ood_mask: np.ndarray | None          # (H, W) bool — True = out-of-distribution

class SegmentationDataset(Protocol):
    num_classes: int
    class_names: tuple[str, ...]
    ignore_index: int
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> SegmentationSample: ...
```

### Candidate

```python
class Candidate(Protocol):
    name: str
    def train(self, dataset: SegmentationDataset) -> None: ...
    def predict(self, image: np.ndarray) -> SegmentationOutput: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

### Testing

All testable with hand-crafted numpy arrays. No external data, no GPU. Tests should verify:

- Dataclass construction and field access.
- UncertaintyOutput inherits prediction from SegmentationOutput.
- Optional UQ fields default to None.
- Protocol compliance — a minimal concrete class satisfies SegmentationDataset / Candidate.

## Phase 2 — Dataset Modifiers

Composable wrappers that take a SegmentationDataset and return a new one satisfying the same protocol. Lazy on data — no pixel copying at construction.

### Modifiers

- **`select_classes(dataset, keep)`** — keeps listed classes, remaps to contiguous [0, n), sets ood_mask on non-selected pixels, other pixels become ignore_index.
- **`remap_classes(dataset, mapping)`** — explicit index remapping. For merging classes or non-standard remaps.
- **`filter_samples(dataset, predicate)`** — removes samples based on a predicate. Scans at construction to build index of kept samples.
- **`subsample(dataset, n, seed)`** — random subset of fixed size.
- **`combine(*datasets)`** — concatenates datasets. All must share the same class set.

### TorchDatasetAdapter

Utility that wraps SegmentationDataset into a torch.utils.data.Dataset. Optional transform argument. Lives in segspicious as a convenience, not part of the protocol.

### Testing

Build a tiny in-memory fake dataset (e.g. 5 samples, 4×4 images, 3 classes). Test each modifier in isolation and in composition:

- select_classes: verify remapped labels, ood_mask, num_classes, class_names.
- remap_classes: verify merged classes, unmapped → ignore_index.
- filter_samples: verify length, correct samples retained.
- subsample: verify length, determinism with seed.
- combine: verify length is sum, indexing across boundary works.
- Composition: `select_classes(subsample(dataset, n=3), keep=[...])` works.

## Phase 3 — Evaluation Metrics

Pure functions: (predictions, ground_truth) → metrics. No knowledge of datasets or candidates.

### Segmentation Metrics

- **Per-class IoU / mIoU** — intersection over union per class and mean. Must handle ignore_index.
- **Pixel accuracy** — fraction correct. Must handle ignore_index.
- **Mean class accuracy** — per-class accuracy averaged across classes. Must handle ignore_index.

### UQ Metrics

- **OoD detection** — AUROC, AUPR, FPR@95TPR. Accepts an uncertainty map and a binary OoD mask.
- **Failure detection** — AUROC, AUPR, FPR@95TPR. Accepts an uncertainty map, predictions, and labels. Derives binary error mask internally (prediction ≠ label).
- **Selective prediction** — AURC, E-AURC. Accepts an uncertainty map, predictions, and labels.
- **Calibration** — ECE, classwise-ECE, Brier score, NLL. Accepts class_probs and labels.

Active learning is deferred — it requires retraining loops and is qualitatively different from the other tests.

### Testing

All testable with synthetic arrays:

- Perfect predictions → IoU = 1.0, accuracy = 1.0.
- Known wrong predictions → verify specific metric values.
- Ignore_index pixels excluded from all segmentation metrics.
- Synthetic uncertainty maps with known separation → verify AUROC ≈ 1.0 or ≈ 0.5.
- Perfectly calibrated synthetic probs → ECE ≈ 0.
- Edge cases: single class, all pixels ignored, empty OoD mask.

## Phase 4 — Concrete Dataset

First concrete dataset implementation, living in experiments/.

### Cityscapes

Wraps torchvision.datasets.Cityscapes. Handles the id-to-trainId remapping (34 raw classes → 19 training classes). Returns SegmentationSample with raw uint8 RGB images and remapped labels.

### Testing

Integration test that checks the wrapper satisfies the protocol, returns correct shapes, and remaps labels correctly. Requires Cityscapes data on disk — gate behind an environment variable or marker so CI can skip it.

## Phase 5 — Concrete Candidates

First concrete candidates, living in experiments/.

### Softmax Baseline

Simplest possible candidate. Single-network segmentation model (e.g. DeepLabV3+ with a ResNet backbone). Produces SegmentationOutput with prediction, and UncertaintyOutput with class_probs (softmax) and predictive_uncertainty (entropy). Proves the full pipeline works end-to-end: train on a dataset, predict, feed outputs to metrics.

### First UQ Candidate

After the softmax baseline works, add one real UQ method — MC Dropout or Deep Ensemble. This is the first candidate that populates epistemic_uncertainty and exercises the UQ metrics meaningfully.

### Testing

Smoke tests that verify the candidate satisfies the protocol and produces correctly-shaped outputs. Full training tests are expensive and don't belong in CI — those are the experiment scripts themselves.
