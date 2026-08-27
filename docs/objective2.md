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

`scripts/generate_objective2_graphs.py` applies the frozen, target-adapted
Objective 1 U-Net in GPU batches and converts each model-space probability map
directly into a compact seven-feature ROI patch graph. It deliberately saves no
predicted masks or preprocessed medical images. Generation is resumable, binds
every graph to the exact cohort and checkpoint SHA-256 values, and writes a
private per-case audit plus an aggregate summary. The same immutable graph root
is then supplied to both GCN and GAT so their comparison changes only the graph
message-passing architecture.

For a full Kaggle run, `scripts/generate_objective2_graph_shards.py` divides
the 35,000 frozen training and validation cases into deterministic private
archives. It verifies that the recovery repository is private and uploads each
completed shard before continuing. Repeating the same command restores verified
remote shards and resumes an interrupted local shard. The driver cannot accept
a locked-test manifest, does not save predicted masks, and does not copy source
medical images.

After a completely fresh Kaggle runtime,
`scripts/recover_objective2_graph_shards.py` reconstructs both frozen
train/validation manifests and all graph files directly from those private,
checksum-protected archives. It therefore does not need to rescan NIH images or
repeat segmentation inference.

The 5,000-case test cohort is label-blind during selection and must be evaluated
exactly once after all five checkpoints and thresholds are frozen. No manual
masking is required for Objective 2.

After every validation candidate has been published and its checkpoint hash is
frozen, `scripts/generate_objective2_locked_test_graph_shards.py` creates the
test graphs. It reads the test CSV with `usecols` that explicitly exclude every
`label_*` column, stores each completed graph shard in the private recovery
repository, and saves neither probability masks nor source images. Thus graph
construction cannot expose test labels to model selection.

`scripts/evaluate_objective2_locked_test.py` is the only program allowed to
load test label values. Before doing so it verifies all five checkpoint hashes,
model identities, label order and validation-selected thresholds. It evaluates
the frozen CNN, attention-CNN, compact ViT, GCN and GAT candidates, saves each
model's predictions atomically for interruption recovery, applies the unchanged
validation thresholds, and writes a final lock that prevents a second completed
evaluation. Confidence intervals and comparisons use the same paired bootstrap
case resamples for every model. `scripts/publish_objective2_locked_test.py`
publishes only the aggregate JSON, figure and lock record; case predictions,
identifiers, manifests, images and graph files remain private.

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

`scripts/train_objective2_with_private_recovery.py` adds private Hub recovery
around that atomic state. It verifies that the recovery repository is private,
uploads each stable completed epoch, restores the newest verified snapshot after
a fresh Kaggle runtime, and uploads the final validation-selected artifacts. It
does not accept a test manifest.
