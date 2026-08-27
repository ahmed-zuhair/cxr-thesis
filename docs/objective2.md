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

If a fresh Kaggle runtime loses the private test CSV,
`scripts/recover_objective2_locked_test_cohort.py` reconstructs it from the
verified full NIH manifest. Patient identities are selected from only the
`patient_id` and `split` columns using the frozen role seed. The recovered CSV
is written atomically only when its complete byte-level SHA-256 exactly matches
the original protected cohort hash; test-label statistics are neither computed
nor displayed.

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

## Validation-only enhanced baseline

The published five-model locked-test comparison is immutable. Enhancement work
is stored as a separate validation candidate and cannot overwrite or reinterpret
those results. The first enhanced candidate is an ImageNet-pretrained
DenseNet-121 at 320 px with three-channel ImageNet normalisation, epoch-varying
mild CXR augmentation, square-root transformed and clipped BCE positive
weights, gradient accumulation, gradient clipping and a cosine learning-rate
schedule. Its locked configuration is recorded in
`configs/objective2/nih_enhanced_densenet121.yaml`.

The enhanced path remains test-blind and uses the same epoch-atomic private Hub
recovery wrapper. A typical validation run is:

```bash
python scripts/train_objective2_with_private_recovery.py \
  --model densenet121 --pretrained \
  --train-manifest /private/train_cohort_private.csv \
  --val-manifest /private/val_cohort_private.csv \
  --output-dir /kaggle/working/outputs/objective2_densenet121_seed42 \
  --data-root / --hf-repo OWNER/PRIVATE_RECOVERY_REPO \
  --hf-path objective2/densenet121/seed42/validation_candidate_v1.0.0 \
  --expected-train-sha256 TRAIN_SHA256 --expected-val-sha256 VAL_SHA256 \
  --image-size 320 --batch-size 16 --accumulation-steps 4 \
  --learning-rate 0.0001 --backbone-learning-rate-multiplier 0.1 \
  --augmentation-profile cxr_mild --epoch-varying-augmentation \
  --loss bce --positive-weight-transform sqrt --max-positive-weight 10 \
  --scheduler cosine --gradient-clip-norm 1.0
```

No previously evaluated locked-test cohort is accepted by this workflow. A new
confirmation protocol must be frozen before any enhanced model is tested.
