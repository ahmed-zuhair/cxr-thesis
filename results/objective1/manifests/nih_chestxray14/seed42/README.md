# NIH ChestX-ray14 Canonical Manifest Summary

This artifact records the deterministic NIH ChestX-ray14 data
manifest used for Objective 1 of the thesis:

**Explainable Quantum-Enhanced Graph-Based Deep Learning for
Multimodal Medical Image Analysis and Clinical Report Generation**

## Data split

| Split | Images | Patients |
|---|---:|---:|
| Train | 77719 | 25207 |
| Validation | 8805 | 2801 |
| Test | 25596 | 2797 |
| Total | 112120 | 30805 |

## Split protocol

- The official NIH test list is preserved exactly.
- No test image was moved into training or validation.
- Validation patients were selected only from official development
  patients.
- Validation fraction: 10% of development patients.
- Split seed: 42.
- Patients appearing in multiple splits: 0.
- Missing image paths: 0.
- Duplicate image identifiers: 0.

## Label counts

| Finding | Train | Validation | Test |
|---|---:|---:|---:|
| Atelectasis | 7345 | 935 | 3279 |
| Cardiomegaly | 1523 | 184 | 1069 |
| Consolidation | 2543 | 309 | 1815 |
| Edema | 1244 | 134 | 925 |
| Effusion | 7679 | 980 | 4658 |
| Emphysema | 1295 | 128 | 1093 |
| Fibrosis | 1120 | 131 | 435 |
| Hernia | 127 | 14 | 86 |
| Infiltration | 12305 | 1477 | 6112 |
| Mass | 3695 | 339 | 1748 |
| Nodule | 4199 | 509 | 1623 |
| Pleural_Thickening | 2007 | 235 | 1143 |
| Pneumonia | 770 | 106 | 555 |
| Pneumothorax | 2370 | 267 | 2665 |

Finding labels are not mutually exclusive.

## Reproducibility

- Code commit: `78efed71d40c182d3a222ab164ce89408bb5847d`
- Private manifest SHA-256:
  `c7457b90eb48472995ac9f7b8a00590f4beb2176db3d02aa1d72198260aba8e9`
- Metadata SHA-256:
  `88f75094e25ccc0c6f1f9cdfd4b2f94f9379a0ae07d5ff4dcf94242707b07462`
- Official development-list SHA-256:
  `61fbe896321c1c1c8b75f3e4f3a08e4fef6486d95ef8a667c31d4d60dca6cb81`
- Official test-list SHA-256:
  `38ca5ef7f756092946f57c1a59faca882ed589a1ab1f72590b45dc06c6d5e1cc`

## Privacy and licensing boundary

The patient-level CSV manifest is intentionally excluded from this
public artifact. This publication contains only aggregate statistics,
methodology, and cryptographic hashes.

It does not contain:

- medical images;
- segmentation masks;
- patient identifiers;
- study or image identifiers;
- ages, sex values, or clinical text;
- absolute patient-level image paths.

The original NIH ChestX-ray14 data remain subject to their source
dataset terms.

## Status

This manifest is ready for frozen ROI-mask inference. The next stage
is a controlled 32-image target-domain smoke audit before any
full-dataset inference.

Generated UTC: 2026-08-23T01:14:14.677268+00:00
