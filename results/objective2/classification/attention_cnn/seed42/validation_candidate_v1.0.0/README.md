# Objective 2 Attention-CNN validation candidate

This directory records the completed CBAM Attention-CNN
classification baseline.

## Experiment

- Model: CBAM Attention-CNN
- Training images: 30,000
- Validation images: 5,000
- Labels: 12
- Epochs completed: 11
- Best epoch: 6
- Validation macro AUROC: 0.727068
- Validation macro AUPRC: 0.119950
- Checkpoint SHA-256: `cf3a57d48e032e306d2416794850b9dae412b0a94deb8b9837a9fc34e2336a68`
- Source commit: `2bef7f11dfeb14baf0a129c7ee4bbe1b9c11d5b7`

The checkpoint was selected using validation macro AUROC.
Early stopping occurred after five epochs without improvement.
The locked test cohort was not read, evaluated, or used for
threshold selection.

This remains a validation-selected candidate until CNN,
Attention-CNN, ViT, GCN and GAT have all been trained and their
selection rules frozen.

The checkpoint is stored on Hugging Face and attached to the
corresponding GitHub Release. No patient identifiers, image
identifiers, medical images, private manifests, or case-level
predictions are published.
