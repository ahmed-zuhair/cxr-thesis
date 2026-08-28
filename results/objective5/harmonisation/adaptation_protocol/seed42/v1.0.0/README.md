# Objective 5 External-Domain Adaptation Protocol

This directory contains the protocol locked before any Objective 5
target-domain adaptation training.

The confirmed NIH-trained DenseNet-121 is evaluated as a frozen zero-shot
baseline on CheXpert and PadChest validation cohorts. One conservative
adaptation candidate is then trained independently for each target dataset.

An adapted candidate advances only if target-validation macro AUROC improves
by at least 0.005 over the corresponding frozen zero-shot baseline. Otherwise,
the zero-shot model is retained.

No locked-test data, labels, predictions, thresholds, or metrics were used to
construct this protocol or select adaptation hyperparameters.

No patient identifiers, image identifiers, medical images, private manifests,
or case-level predictions are included.
