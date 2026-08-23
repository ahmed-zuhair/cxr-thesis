---
library_name: pytorch
pipeline_tag: image-segmentation
tags:
- medical-imaging
- chest-xray
- lung-segmentation
- unet
- explainable-ai
- graph-neural-networks
- thesis
- pytorch
license: other
---

# Objective 1: Lung ROI Segmentation

This artifact is part of the thesis:

**Explainable Quantum-Enhanced Graph-Based Deep Learning for
Multimodal Medical Image Analysis and Clinical Report Generation**

## Intended purpose

The model produces union lung-region masks from frontal chest
radiographs. It was developed as the ROI-segmentation component of
the thesis preprocessing and feature-extraction pipeline.

This is a research artifact and is **not approved for clinical use**.

## Model

- Architecture: compact 2D U-Net
- Input: one-channel chest radiograph
- Input resolution: 224 × 224
- Output: binary union lung mask
- Best epoch: 48
- Training seed: 42
- Validation-selected threshold: 0.55
- Checkpoint SHA-256: `94a20eee892f9240df4a4534ced85e55f3677975c627ee336db7eb51a98e5642`

## Experimental protocol

- Dataset: Montgomery–Shenzhen lung-mask collection
- Train cases: 492
- Validation cases: 106
- Locked test cases: 106
- Patient leakage detected: 0
- Test threshold tuning performed: no
- Connected-component postprocessing on test predictions: no
- Bootstrap repetitions: 10,000
- Bootstrap seed: 42

## Locked test results

| Metric | Result |
|---|---:|
| Dice | 0.9623 |
| Dice 95% CI | 0.9566–0.9672 |
| IoU | 0.9287 |
| IoU 95% CI | 0.9188–0.9373 |
| HD95 at 224×224 | 4.77 pixels |
| HD95 95% CI | 3.79–5.89 pixels |
| Precision | 0.9695 |
| Recall | 0.9568 |
| Empty-mask failure rate | 0.0% |

## Validation-to-test consistency

- Best validation Dice: 0.9632
- Locked test Dice: 0.9623
- Absolute difference: 0.0009

The close validation and locked-test results provide no obvious
evidence of progressive overfitting in this experiment.

## Subgroup observations

Montgomery cases achieved a higher mean Dice than Shenzhen cases.
This should be treated as a domain/acquisition signal rather than
proof of demographic bias because the subgroup sizes and acquisition
conditions differ.

PTB-positive and PTB-negative cases showed similar mean Dice.

Detailed subgroup results are provided in:

`metrics/locked_test_subgroup_metrics.csv`

## Published files

- `checkpoint/best.pt`: protected trained checkpoint
- `checkpoint/best.sha256`: checkpoint checksum
- `metrics/training_history.csv`
- `metrics/locked_test_summary.json`
- `metrics/locked_test_subgroup_metrics.csv`
- `metrics/locked_test_per_case_anonymized.csv`
- `figures/training_history.png`
- `figures/locked_test_worst_cases.png`
- `figures/locked_test_seeded_random_cases.png`
- `configuration/`: frozen experiment configuration
- `source/`: source snapshot used by the experiment
- `reproducibility/`: environment and locked-test metadata

## Excluded data

The publication package intentionally excludes:

- original medical images;
- original segmentation masks;
- clinical remarks;
- patient, study, and image identifiers;
- the original patient-level manifest;
- target-mask and probability arrays.

## Limitations

- The model was evaluated at 224×224 resolution.
- HD95 is reported in resized-image pixels, not millimetres.
- External validation on NIH, CheXpert, and PadChest remains pending.
- Performance differences between source domains require further
  investigation and motivate the thesis domain-harmonisation work.
- Qualitative images originate from de-identified public research
  datasets and remain subject to their original dataset terms.

## Code

Source repository:

https://github.com/ahmed-zuhair/cxr-thesis

## Version

- Result version: v1.0.0
- Publication timestamp UTC:
  2026-08-23T00:45:57.949497+00:00


## Published checkpoint

The checkpoint and complete sanitized result package are mirrored at:

- Hugging Face: https://huggingface.co/ahmed-zuhair/cxr-thesis-checkpoints/tree/main/objective1/roi_segmentation/seed42
- GitHub Release tag: `objective1-roi-segmentation-v1.0.0`

Publication bundle SHA-256:

`b66ab657836d18bdae9eb91a6f14eea885c53fcd839eab6a4346ae82a2be8bf6`
