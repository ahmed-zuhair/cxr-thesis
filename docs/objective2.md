# Objective 2 — Multilabel classification comparison

Objective 2 compares five patient-level model families on the same locked NIH
ChestXray14 cohort: a CNN, a CBAM attention-CNN, a compact vision transformer,
a graph convolutional network, and a graph attention network. Every model uses
the same 12 labels, training/validation/test split, training-only class weights,
validation macro AUROC selection rule, and validation-only per-label threshold
selection.

The image models consume a CXR plus encoded clinical context. The graph models
consume an automatically generated ROI-aware patch graph plus the same clinical
context. The implementation is dependency-light PyTorch and does not require
PyTorch Geometric.

The 5,000-case test cohort is label-blind during selection and must be evaluated
exactly once after all five checkpoints and thresholds are frozen. No manual
masking is required for Objective 2.

Use `scripts/train_objective2_classifier.py` to train exactly one model with
training and validation manifests. The script deliberately accepts no test
manifest, selects checkpoints by validation macro AUROC, freezes per-label
validation F1 thresholds, and writes `test_evaluated: false` into every
checkpoint.

Long training runs write an atomic `last.pt`, `last.sha256`, and
`history_progress.csv` after every completed epoch. If a Kaggle session is
interrupted but the output directory survives, repeat the identical command
with `--resume`. Resume is rejected when the model settings or the SHA-256 of
either training/validation manifest differs from the saved run. Optimizer,
scheduler, data-loader generator, and Python/NumPy/PyTorch RNG states are all
restored; the test cohort remains inaccessible.
