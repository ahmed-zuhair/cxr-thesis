# Objective 2 DENSENET121 validation candidate

- Architecture: ImageNet-pretrained DenseNet-121
- Training images: 30,000
- Validation images: 5,000
- Labels: 12
- Best epoch: 19
- Validation macro AUROC: 0.804678
- Validation macro AUPRC: 0.209602
- Checkpoint SHA-256: `2b7fa0d2f3dee3c59c538be15dd0435c71ad26b411fc1312bd7e5fe99fbac55f`
- Source commit: `d909139111da711657fa96a668f8ec366becc19f`

The checkpoint was selected using validation macro AUROC. The locked test cohort
was not read, evaluated, or used for threshold selection. This directory contains
no patient identifiers, image identifiers, medical images, private manifests, or
case-level predictions. The checkpoint is stored on Hugging Face and attached to
the corresponding GitHub Release.
