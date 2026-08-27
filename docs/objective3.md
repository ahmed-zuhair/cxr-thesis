# Objective 3: Quantum-Enhanced Graph Classification

Objective 3 tests whether a genuine variational quantum bottleneck adds useful
classification signal to representations learned by the frozen Objective 2 GAT
encoder. The experiment compares the quantum head against an exactly
parameter-matched classical bottleneck under identical embeddings, residual
fusion, classifier, losses, seeds, and validation protocol.

The frozen circuit uses four qubits, angle embedding, two strongly entangling
layers, and four Pauli-Z expectation values. It therefore has 24 trainable
variational parameters. The classical control also has exactly 24 trainable
bottleneck parameters. Both heads receive the same 160-dimensional fused GAT
and clinical embedding.

Three training seeds (42, 43, and 44) will quantify optimization variability.
Architecture and thresholds are selected using training and validation only.
Neither completed Objective 2 evaluation cohort is reopened. Before final
Objective 3 evaluation, a new label-blind cohort of complete, previously unused
NIH test patients will be locked and publicly timestamped; it will then be
evaluated exactly once.

`scripts/smoke_objective3_quantum.py` verifies dependency compatibility,
forward/backward differentiation, finite gradients, output shapes, and exact
classical-versus-quantum bottleneck parameter matching without accessing any
medical data.

`scripts/extract_objective3_gat_embeddings.py` loads the frozen, test-blind
Objective 2 GAT checkpoint and extracts one 160-dimensional fused graph and
clinical embedding for every training and validation case. The embeddings are
saved in deterministic manifest order as private 1,000-case shards. Every
completed shard is checksum-protected and uploaded to a verified private Hub
repository, so a fresh Kaggle runtime restores completed work instead of
repeating it. Test manifests, test labels, medical images, and predicted masks
are not accessed or stored by this stage.

`scripts/train_objective3_with_private_recovery.py` trains one member of the
paired comparison from those frozen embeddings. Embedding standardization and
positive class weights are fitted on training data only. Shared projection and
classifier layers receive exactly the same seeded initialization for the
quantum and classical variants, while each bottleneck has exactly 24 trainable
parameters. The head contains 2,648 trainable parameters in total and executes
on CPU because the quantum simulator is CPU-bound. Every completed epoch is
checksum-protected and uploaded to the private recovery repository. The full
study runs both variants at seeds 42, 43, and 44; smoke outputs made with case
limits are explicitly marked as non-research results.

## Bounded v1.1 enhancement amendment

The original v1.0 protocol and its negative validation result remain immutable.
Before any additional training or test-cohort selection,
`scripts/lock_objective3_enhancement_protocol.py` records a v1.1 amendment that
freezes one and only one enhancement attempt. The amendment references the
original protocol by SHA-256 and refuses to run unless the original record is
still test-blind.

If Kaggle restarts after printing the successful v1.0 lock but before that
public JSON is uploaded, the same tool supports the explicit
`--recover-missing-original-from-recorded-lock` mode. The amendment then keeps
the recorded v1.0 SHA-256 and test-blind terminal record but states that the
missing ephemeral artifact cannot be byte-for-byte re-verified. This recovery
mode is rejected whenever the original file is available.

The v1.1 head repeats four-feature angle encoding before each of three shallow
entangling blocks, producing a 36-parameter quantum bottleneck. Its classical
control uses two four-feature linear transformations with exactly 36 trainable
parameters. Both variants use the same learned, initially 0.1-scaled 4-to-160
residual back-projection, classifier, initialization, embeddings, losses,
optimizer, and seeds. Each head has 3,253 trainable parameters in total.

The enhanced quantum head advances to the single final evaluation only when
its mean validation macro AUROC exceeds the matched control and it wins at
least two of three seeds. Macro AUPRC and macro F1 are reported but do not
select the model. If either primary rule fails, no further Objective 3
architecture tuning is permitted and the negative result is reported.

When both rules pass, `scripts/create_objective3_final_evaluation_cohort.py`
selects exactly 5,000 images at seed 4042 from complete official NIH test
patients after excluding every patient used by either earlier Objective 2
evaluation cohort. The irreversible identity selection reads only patient and
split columns. Labels are serialized only after selection and no label
statistics are calculated. The private cohort and lock are stored only in the
private recovery repository.

`scripts/publish_objective3_final_protocol.py` publishes the sanitized v1.1
validation aggregate, amendment, final-cohort hash, and protocol lock before
any final-cohort label evaluation. It rejects private manifests, identifiers,
images, and case-level predictions.

Use `--architecture v1_1_reupload_gated` with the smoke, direct-training, and
private-recovery scripts. Use a new output directory and private Hub path so
the v1.0 checkpoints are never overwritten.
