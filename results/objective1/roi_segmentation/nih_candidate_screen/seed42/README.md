---
license: mit
tags:
- medical-imaging
- chest-xray
- segmentation
- domain-shift
- active-learning
- explainable-ai
---

# NIH ChestX-ray14 ROI Candidate Screen — Seed 42

This package contains aggregate results from a deterministic
2,000-patient target-domain screen using the frozen Objective 1
lung ROI segmentation model.

## Design

- 1,000 NIH development-train cases
- 1,000 NIH development-validation cases
- 2,000 unique patients and images
- Stratification by source split, PA/AP view, sex, and
  No Finding versus abnormal
- Official NIH test split excluded
- Frozen decision threshold: 0.55
- No target-domain threshold tuning

## Main screening findings

- 2,000/2,000 cases processed
- 0 inference failures
- 30 automatic ROI plausibility failures
- 2 empty predicted masks
- 3 masks touching the model-space border
- 311 masks with more than two retained components
- 41 masks with five or more retained components

The high-priority failures were concentrated in AP radiographs,
providing evidence of target-domain shift and motivating manual
annotation and domain-adaptive fine-tuning.

## Interpretation

These results are an active-QC and cohort-selection experiment.
They are not a ground-truth segmentation accuracy evaluation.
Entropy and threshold-margin measurements are deterministic
confidence proxies, not calibrated uncertainty estimates.

## Privacy and licensing boundary

This public package contains no patient identifiers, image
identifiers, image paths, original radiographs, generated masks,
or private patient-level manifests.

The NIH ChestX-ray14 images are not redistributed.
