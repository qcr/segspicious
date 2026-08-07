# Uncertainty Metrics Recommendations

Which metrics to use, why, and where they come from.

## Context

This document recommends a set of uncertainty evaluation metrics for
segspicious, informed by:

- The **ValUES** framework (Kahl et al., ICLR 2024) — a systematic
  evaluation of uncertainty methods for semantic segmentation across five
  downstream tasks.
- The **torch-uncertainty** library — which already implements most of what
  we need.

Our use case is infrastructure inspection (corrosion, defects). We care
about **pixel-level** evaluation: identifying which regions the model is
uncertain about, not whether to reject a whole image. Image-level
aggregation strategies (mean, max-patch, threshold) are not needed.

## Tasks and Metrics

Three evaluation tasks, five metrics, all pixel-level.

### OoD Detection

*"Which pixels belong to classes the model hasn't seen before?"*

A novel defect type or material appears in part of an otherwise normal
image. We need to know which pixels are unfamiliar.

| Metric | What it measures | Higher/lower is better |
|---|---|---|
| **AUROC** | Overall ranking quality — can the score separate ID from OoD pixels? | Higher |
| **FPR@95TPR** | At 95% OoD recall, what fraction of ID pixels are falsely flagged? | Lower |
| **AUPR** | Precision-recall summary — better than AUROC when OoD pixels are rare. | Higher |

**Visualisation:** ROC curve (TPR vs FPR).

Ground truth: derived from the label tensor. Pixels with `label >= num_classes`
are OoD (existing segspicious convention).

**torch-uncertainty implementations:**
- `SegmentationBinaryAUROC` — per-image pixel-level AUROC, averaged across images.
- `SegmentationFPR95` — per-image pixel-level FPR@95TPR, averaged across images.
- `SegmentationBinaryAveragePrecision` — per-image pixel-level AUPR, averaged across images.

**Accepts:** `predictive_uncertainty`, `epistemic_uncertainty`, `ood_score` —
whichever the model provides. Each field produces an independent set of
results.

### Acceptance Risk

*"If I trust this pixel's prediction, what's my expected cost — accounting
for both misclassification and OoD?"*

OoD detection and selective prediction evaluate failure modes in isolation.
Acceptance risk evaluates them jointly: a single score ranks pixels for a
reject/accept decision, and the cost function penalises both accepted
misclassified ID pixels and accepted OoD pixels.

| Metric | What it measures | Higher/lower is better |
|---|---|---|
| **SCOD-AURC** | Area under the risk-coverage curve using the blended SCOD cost. Measures how well one score handles both failure modes. | Lower |

The cost parameter `c_OOD` (default 0.5) controls the relative penalty.
At 0.5, accepting an OoD pixel and accepting a misclassified ID pixel are
equally bad. Tuning it reflects domain priorities — e.g. in inspection,
missing an unknown defect type may be worse than misclassifying a known one.

**torch-uncertainty implementation:**
- `SCODAURC` — exists for classification. Needs pixel-level image-averaged
  adaptation for segmentation.

**Accepts:** `predictive_uncertainty`, `epistemic_uncertainty`, `ood_score` —
whichever the model provides. Each field produces an independent set of
results. Also requires `prediction` and class labels to derive
classification errors on ID pixels.

### Calibration

*"Are the predicted class probabilities trustworthy?"*

If the model says 80% confidence, it should be correct ~80% of the time.

| Metric | What it measures | Higher/lower is better |
|---|---|---|
| **ACE** | Average Calibration Error — like ECE but with equally-sized bins. Avoids bias from dominant background pixels. | Lower |

ACE is recommended over ECE for segmentation by ValUES (Kahl et al.) because
ECE's equal-width bins overweight the high-confidence background class that
dominates most images. ACE uses equal-count bins so every confidence range
gets equal weight.

**torch-uncertainty implementation:**
- `AdaptiveCalibrationError`

**Accepts:** `class_probs` only.

### Selective Prediction

*"Does removing uncertain pixels actually remove errors?"*

Rank pixels by uncertainty, progressively remove the most uncertain ones, and
check whether the error on the remaining pixels drops.

| Metric | What it measures | Higher/lower is better |
|---|---|---|
| **AUSE** | Area Under the Sparsification Error curve — gap between the model's uncertainty ranking and an oracle ranking. | Lower |

AUSE isolates uncertainty quality from model quality: a perfect model has zero
error everywhere so sparsification doesn't help, but AUSE measures whether
the *ranking* of uncertainties matches the ranking of actual errors. This is
more informative than AURC at pixel level, which conflates model performance
with uncertainty quality.

**Visualisation:** Sparsification plot (fraction of pixels removed vs error
on remaining pixels), showing both the model curve and the oracle curve.

**torch-uncertainty implementation:**
- `AUSE`

**Accepts:** `predictive_uncertainty`, `aleatoric_uncertainty`,
`epistemic_uncertainty`, `ood_score` — whichever the model provides.

## Summary

| Task | Metrics | Level | Visualisation | torch-uncertainty class |
|---|---|---|---|---|
| OoD Detection | AUROC, FPR@95, AUPR | Pixel (image-averaged) | ROC curve | `SegmentationBinaryAUROC`, `SegmentationFPR95`, `SegmentationBinaryAveragePrecision` |
| Acceptance Risk | SCOD-AURC | Pixel (image-averaged) | Risk-coverage curve | Adapted from `SCODAURC` |
| Calibration | ACE | Pixel | Reliability diagram | `AdaptiveCalibrationError` |
| Selective Prediction | AUSE | Pixel | Sparsification plot | `AUSE` |

Six metrics. Four tasks. All pixel-level. All based on torch-uncertainty
(SCOD-AURC needs pixel-level adaptation).

## What We Chose Not To Include

| Metric/Task | Why not |
|---|---|
| **Image-level AURC / E-AURC** (failure detection) | Our use case doesn't need whole-image rejection. Pixel-level selective prediction (AUSE) covers the same ground. |
| **Image-level aggregation strategies** (mean, max-patch, threshold) | Only needed for image-level tasks. ValUES showed these are critical but dataset-dependent — since we're staying pixel-level, they're irrelevant. |
| **NCC / GED** (ambiguity modelling) | Requires multi-rater annotations. Not available for our datasets. |
| **Active learning metrics** | A protocol, not a metric. Out of scope for the metrics module. |
| **ECE** | ACE is strictly better for segmentation due to background pixel bias in equal-width bins. |
| **Brier score / NLL** | Useful diagnostics but not tied to a clear downstream task. Can be added later if needed. |
| **PAvPU** (patch accuracy vs uncertainty) | Interesting but a niche metric. Our three tasks already cover the key questions. |

## References

- Kahl, K.-C., Lüth, C. T., Zenk, M., Maier-Hein, K., & Jaeger, P. F.
  (2024). ValUES: A Framework for Systematic Validation of Uncertainty
  Estimation in Semantic Segmentation. ICLR 2024.
- Xia, H. & Bouganis, C.-S. (2022). Augmenting Softmax Information for
  Selective Classification with Out-of-Distribution Data. ACCV 2022.
- Narasimhan, H., Jitkrittum, W., Menon, A. K., Rawat, A., & Kumar, S.
  (2023). Plugin Estimators for Selective Classification with
  Out-of-Distribution Detection.
- torch-uncertainty: https://github.com/ENSTA-U2IS-AI/torch-uncertainty
