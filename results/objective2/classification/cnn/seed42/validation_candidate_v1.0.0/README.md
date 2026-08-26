# Objective 2 CNN validation candidate

This directory records the first complete Objective 2
classification baseline.

## Experiment

- Model: CNN
- Training images: 30,000
- Validation images: 5,000
- Labels: 12
- Best epoch: 14
- Validation macro AUROC: 0.733899
- Validation macro AUPRC: 0.126425
- Checkpoint SHA-256: `3f5e37ad4995f0af8381396dc6b01947c2cceb858d49879f4c67d39d0f9da3b0`
- Source commit: `671385347af85dab92b1e66ed21d80b15c80ceb6`

The checkpoint was selected using validation macro AUROC.
The locked test cohort was not read, evaluated, or used for
threshold selection. This is a candidate result, not the final
Objective 2 model comparison.

The model checkpoint is stored on Hugging Face and attached to
the corresponding GitHub Release. No patient identifiers, image
identifiers, medical images, private manifests, or case-level
predictions are published.
