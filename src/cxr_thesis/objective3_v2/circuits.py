"""Parameterised quantum circuits for Objective 3 v2.0.

Populated by Part 3 (diagnostics) and Part 6 (the graph-structured ansatz).

Planned contents
----------------
``expressibility``
    KL divergence between the sampled fidelity distribution of an ansatz and the
    Haar-random distribution P(F) = (2**n - 1) * (1 - F)**(2**n - 2), following
    Sim, Johnson & Aspuru-Guzik (Adv. Quantum Technol., 2019).
``entangling_capability``
    Mean Meyer-Wallach Q over random parameter draws, same reference.
``gradient_variance``
    Var[d<Z_0>/d theta] against qubit count and depth, to detect the barren
    plateaus of McClean et al. (Nat. Commun. 9:4812, 2018).
``GraphStructuredCircuit``
    One qubit per graph node; per-node-type classical projection into angle
    encoding; entangling gates placed ONLY on pairs present in that graph's
    adjacency, with parameters shared across graphs. The ablation that matters
    is the same circuit entangled over all pairs, which isolates whether the
    topology carries usable information.

Keep pennylane pinned at 0.45.1 so results stay comparable with v1.1.
"""

from __future__ import annotations

__all__: list[str] = []
