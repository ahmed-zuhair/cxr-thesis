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
