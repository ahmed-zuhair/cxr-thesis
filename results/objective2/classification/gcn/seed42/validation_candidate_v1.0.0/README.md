# Objective 2 GCN validation candidate

- Architecture: graph convolutional network
- Training images: 30,000
- Validation images: 5,000
- Labels: 12
- Best epoch: 20
- Validation macro AUROC: 0.652047
- Validation macro AUPRC: 0.087940
- Checkpoint SHA-256: `a6e88ebe8e709ffaef72eb7eac3536a3edc97bb143a8a5c41020d46c461bc4f5`
- Source commit: `228024ef1f250d5c6fd9ae6b7e15818f124c5cdb`

The checkpoint was selected using validation macro AUROC. The locked test cohort
was not read, evaluated, or used for threshold selection. This directory contains
no patient identifiers, image identifiers, medical images, private manifests, or
case-level predictions. The checkpoint is stored on Hugging Face and attached to
the corresponding GitHub Release.
