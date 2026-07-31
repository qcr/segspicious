# Uncertainty Quantification Design

Benchmarking uncertainty estimation methods in semantic segmentation.

## Core Concept

The unit of comparison is the **candidate**: a complete pipeline from input image to a standardised uncertainty output. Each candidate populates only the fields it meaningfully provides. The framework runs every candidate against every compatible evaluation test and reports the results.

## Uncertainty Output

Every candidate's `predict()` method receives a batch of images as a `(B, C, H, W)` tensor and returns an `UncertaintyOutput` — a dataclass that extends `SegmentationOutput` (see `segmentation_design.md`) with five optional uncertainty fields. UQ candidates inherit `prediction` and are automatically evaluated on segmentation tests as well.

All tensors follow PyTorch conventions: channels-first, with a batch dimension.

```
SegmentationOutput:
    prediction:              (B, H, W)          # always present — argmax class map

UncertaintyOutput(SegmentationOutput):
    class_probs:             (B, C, H, W) | -   # class probability vector (channels-first)
    predictive_uncertainty:  (B, H, W)    | -   # total uncertainty (responds to all sources)
    aleatoric_uncertainty:   (B, H, W)    | -   # irreducible data ambiguity
    epistemic_uncertainty:   (B, H, W)    | -   # reducible model ignorance
    ood_score:               (B, H, W)    | -   # how out-of-distribution the input is
```

### Field Semantics

- **prediction** — inherited from `SegmentationOutput`. The model's best-guess class per pixel.
- **class_probs** — a probability distribution over classes per pixel. `(B, C, H, W)` channels-first, following PyTorch convention. Required for calibration evaluation. Must be interpretable as probabilities (sums to 1 along C, non-negative).
- **predictive_uncertainty** — total uncertainty scalar. High when the model is uncertain for any reason (ambiguous data, unseen input, model ignorance). Typically entropy of the predictive distribution.
- **aleatoric_uncertainty** — uncertainty due to inherent ambiguity in the data. Cannot be reduced by collecting more training data. Boundary regions, label noise, overlapping classes.
- **epistemic_uncertainty** — uncertainty due to limited training data or model capacity. Can be reduced by training on more data. Measures what the model doesn't know.
- **ood_score** — how unlike training data this input is. Not a measure of prediction quality — a measure of input familiarity. Includes density-based methods (Mahalanobis, KNN) and energy-based scores.

### Candidate Responsibility

The candidate is the semantic bridge between a model's raw output and the standardised fields. It must only populate fields it can meaningfully provide:

- An ensemble computes MI from member disagreement → `epistemic_uncertainty`. It does not put MI into `ood_score` even though MI correlates with OoD-ness.
- An SSN samples model aleatoric variation, not parameter variation → it populates `aleatoric_uncertainty` but not `epistemic_uncertainty`, even though the underlying math (sample disagreement) looks similar.
- An energy score computes -LogSumExp(logits) → `ood_score`. It does not populate `predictive_uncertainty` even though it has access to the same logits.

The candidate author knows what their method measures. The framework trusts that labelling.

## Evaluation Tests

Each test declares which fields it requires. The framework checks compatibility and runs all valid (candidate, test) pairings. All metrics use torchmetrics and torch-uncertainty.metrics, which accumulate across batches via `.update()` / `.compute()`.

### OoD Detection

Separates in-distribution pixels from out-of-distribution pixels. OoD ground truth is encoded in the label tensor: pixels with `label >= num_classes` are OoD (see `experiment_design.md`). The benchmark runner derives binary OoD targets from this convention and passes `(ood_scores, binary_labels)` to the metrics.

| | |
|---|---|
| **Accepts** | `predictive_uncertainty`, `epistemic_uncertainty`, `ood_score` |
| **Ground truth** | Derived from label tensor: `label >= num_classes` → OoD |
| **Metrics** | AUROC, AUPR, FPR@95TPR |
| **Implementation** | `torch-uncertainty` `SegmentationBinaryAUROC`, `SegmentationBinaryAveragePrecision`, `SegmentationFPR95` |

Runs independently for each populated field — a candidate with both `epistemic_uncertainty` and `ood_score` produces two sets of metrics.

The torch-uncertainty segmentation OoD metrics are **image-averaged**: AUROC / FPR95 is computed per image then averaged across the batch. This is the convention in the dense OoD-detection literature and behaves better than computing over flattened pixels when image sizes or OoD prevalences vary.

### Failure Detection

Detects pixels where the model's prediction is wrong.

| | |
|---|---|
| **Accepts** | `predictive_uncertainty`, `aleatoric_uncertainty`, `epistemic_uncertainty`, `ood_score` |
| **Ground truth** | Class labels `(B, H, W)` (binary error mask derived from `prediction ≠ label`) |
| **Metrics** | AUROC, AUPR, FPR@95TPR |

### Selective Prediction (Risk-Coverage)

Measures accuracy improvement when the model abstains on uncertain pixels.

| | |
|---|---|
| **Accepts** | `predictive_uncertainty`, `aleatoric_uncertainty`, `epistemic_uncertainty`, `ood_score` |
| **Ground truth** | Class labels `(B, H, W)` |
| **Metrics** | AURC, E-AURC |
| **Implementation** | `torch-uncertainty` `AURC`, `AUGRC` |

### Calibration

Evaluates whether predicted probabilities match empirical correctness frequency.

| | |
|---|---|
| **Accepts** | `class_probs` |
| **Ground truth** | Class labels `(B, H, W)` |
| **Metrics** | ECE, classwise-ECE, Brier score, NLL |
| **Implementation** | `torchmetrics` `CalibrationError`, `torch-uncertainty` `BrierScore`, `CategoricalNLL` |

### Active Learning

Evaluates whether uncertainty scores identify the most informative samples to label.

| | |
|---|---|
| **Accepts** | `predictive_uncertainty`, `epistemic_uncertainty`, `ood_score` |
| **Ground truth** | Evaluated indirectly via learning curves after retraining |
| **Metrics** | Learning curve AUC, performance at fixed budget |

## Candidate Examples

```
Softmax Baseline:
    prediction:              argmax(softmax(logits))
    class_probs:             softmax(logits)                ✓
    predictive_uncertainty:  entropy(softmax(logits))       ✓

Energy Score:
    prediction:              argmax(logits)
    ood_score:               -LogSumExp(logits)             ✓

Deep Ensemble (M=5):
    prediction:              argmax(mean_probs)
    class_probs:             mean(member_probs)             ✓
    predictive_uncertainty:  H[mean(member_probs)]          ✓
    aleatoric_uncertainty:   mean(H[each member])           ✓
    epistemic_uncertainty:   predictive - aleatoric (MI)    ✓

MC Dropout (N=20):
    prediction:              argmax(mean_probs)
    class_probs:             mean(sample_probs)             ✓
    predictive_uncertainty:  H[mean(sample_probs)]          ✓
    aleatoric_uncertainty:   mean(H[each sample])           ✓
    epistemic_uncertainty:   predictive - aleatoric (MI)    ✓

Evidential (Dirichlet):
    prediction:              argmax(expected_probs)
    class_probs:             α / sum(α)                    ✓
    predictive_uncertainty:  H[expected_probs]              ✓
    aleatoric_uncertainty:   expected entropy (analytic)    ✓
    epistemic_uncertainty:   MI (analytic)                  ✓

Mahalanobis:
    prediction:              argmax(logits)
    ood_score:               mahalanobis distance           ✓

DDU:
    prediction:              argmax(softmax(logits))
    class_probs:             softmax(logits)                ✓
    predictive_uncertainty:  entropy(softmax(logits))       ✓
    ood_score:               GMM log-likelihood             ✓
```

## Results Matrix

The benchmark output is a table of candidates × (test, field) pairs:

```
                     OoD     OoD     OoD    Fail    Fail     Calib   ...
                    (pred)  (epist) (ood)  (pred)  (aleat)  (probs)
Softmax Baseline     0.81     -       -     0.74     -       0.12
Energy Score          -       -      0.85     -       -        -
Deep Ensemble        0.83    0.91     -     0.79    0.68     0.04
Evidential           0.80    0.86   0.88    0.76    0.65     0.06
Mahalanobis           -       -     0.92     -       -        -
DDU                  0.82     -     0.90    0.75     -       0.05
```

Empty cells are documented incompatibilities, not missing data.

## Shared Inference

Multiple candidates may share the same underlying model. An ensemble computes member predictions once — separate candidates for MI, predictive entropy, etc. all draw from the same forward passes. This is an efficiency optimisation handled by grouping candidates by model at runtime. Conceptually, each candidate is independent.
