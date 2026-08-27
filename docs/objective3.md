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
