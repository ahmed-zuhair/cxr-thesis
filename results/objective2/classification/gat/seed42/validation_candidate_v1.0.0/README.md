# Objective 2 GAT validation candidate

- Architecture: graph attention network
- Training images: 30,000
- Validation images: 5,000
- Labels: 12
- Best epoch: 20
- Validation macro AUROC: 0.657986
- Validation macro AUPRC: 0.088991
- Checkpoint SHA-256: `f34c3db2038c136077011659daee2a1a7d799cc6f87652ddd94d8b5fced7c70d`
- Source commit: `228024ef1f250d5c6fd9ae6b7e15818f124c5cdb`

The checkpoint was selected using validation macro AUROC. The locked test cohort
was not read, evaluated, or used for threshold selection. This directory contains
no patient identifiers, image identifiers, medical images, private manifests, or
case-level predictions. The checkpoint is stored on Hugging Face and attached to
the corresponding GitHub Release.
