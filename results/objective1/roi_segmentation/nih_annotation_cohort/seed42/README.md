---
license: mit
tags:
- medical-imaging
- chest-xray
- segmentation
- domain-adaptation
- active-learning
- reproducibility
---

# NIH ROI Manual-Annotation Cohort Design — Seed 42

This artifact documents the aggregate design of a private,
patient-disjoint 200-case NIH ChestX-ray14 lung ROI annotation
cohort.

## Cohort roles

- 120 adaptation-training cases
- 40 target-validation cases
- 40 locked target-test cases
- 200 unique patients and images
- Zero patient or image overlap between roles
- Eight balanced view × sex × finding strata per role
- Official NIH test split excluded

## Selection safeguards

The locked 40-case target test was selected using deterministic
SHA-256 ordering from identifiers only. Selection occurred before
the prediction audit was joined.

Therefore, locked-test membership did not depend on:

- Predicted masks
- ROI size or shape
- Confidence or entropy proxies
- Automatic QC results
- Active-QC risk rankings

The locked-test annotation bundle contains images only and no
model-generated pre-annotations.

Adaptation-training and target-validation cases intentionally mix
high-risk predictions with cases spanning the remaining risk
distribution.

## Interpretation

This release documents cohort design and reproducibility only.
It does not contain completed manual annotations, target-domain
accuracy results, medical images, predicted masks, or patient-level
manifests.

## Privacy boundary

This public artifact contains no patient identifiers, image
identifiers, source paths, medical images, predicted masks,
completed masks, or annotation-bundle contents.
