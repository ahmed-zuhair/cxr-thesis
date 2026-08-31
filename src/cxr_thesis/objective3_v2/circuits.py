"""Circuit diagnostics for Objective 3 v2.0: was the v1.1 ansatz ever trainable?

Part 3 measures three standard properties of the circuits v1.0 and v1.1 actually
used, so the null result can be attributed to a mechanism rather than left
unexplained:

* **Expressibility** (Sim, Johnson & Aspuru-Guzik, Adv. Quantum Technol. 2:1900070,
  2019) — KL divergence between the ansatz's fidelity distribution under random
  parameters and the Haar-random distribution. Large KL means the circuit reaches
  only a small corner of state space.
* **Entangling capability** (same reference) — mean Meyer-Wallach Q. Zero means
  the circuit never leaves product states, so the qubits carry no joint
  information.
* **Gradient variance** (McClean et al., Nat. Commun. 9:4812, 2018) — Var of a
  cost gradient under random initialisation, against qubit count and depth.
  Exponential decay is a barren plateau: the model cannot train regardless of
  how much data it is given.

The circuits mirror ``cxr_thesis.objective3.models`` exactly:

* v1.0 ``QuantumBottleneck``: AngleEmbedding(RY) then StronglyEntanglingLayers
  with weights of shape (2, 4, 3) — 24 parameters.
* v1.1 ``QuantumReuploadingBottleneck``: three blocks, each re-applying
  AngleEmbedding(RY) then one StronglyEntanglingLayer — weights (3, 4, 3), 36
  parameters.

Statevectors are built in NumPy and vectorised over the sample axis; gradients
use the exact parameter-shift rule. ``tests/test_objective3_v2_circuits.py``
cross-checks both ansätze against PennyLane 0.45.1.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

GRAPH_STRUCTURED_ANSATZ_NAME = "graph_structured"


# --------------------------------------------------------------------------
# gate plumbing
# --------------------------------------------------------------------------


def _zero_state(samples: int, qubits: int) -> np.ndarray:
    state = np.zeros((samples, 2**qubits), dtype=np.complex128)
    state[:, 0] = 1.0
    return state


def _apply_batched_unitary(
    state: np.ndarray,
    matrices: np.ndarray,
    qubit: int,
    qubits: int,
) -> np.ndarray:
    """Apply a per-sample 2x2 unitary to ``qubit``; ``matrices`` is [n, 2, 2]."""

    samples = state.shape[0]
    left = 2**qubit
    right = 2 ** (qubits - qubit - 1)
    view = state.reshape(samples, left, 2, right)
    lower, upper = view[:, :, 0, :], view[:, :, 1, :]
    result = np.empty_like(view)
    result[:, :, 0, :] = (
        matrices[:, 0, 0][:, None, None] * lower
        + matrices[:, 0, 1][:, None, None] * upper
    )
    result[:, :, 1, :] = (
        matrices[:, 1, 0][:, None, None] * lower
        + matrices[:, 1, 1][:, None, None] * upper
    )
    return result.reshape(samples, 2**qubits)


def rot_matrices(
    phi: np.ndarray,
    theta: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """PennyLane ``qml.Rot`` = RZ(omega) RY(theta) RZ(phi), batched."""

    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    result = np.empty(phi.shape + (2, 2), dtype=np.complex128)
    result[..., 0, 0] = np.exp(-0.5j * (phi + omega)) * cosine
    result[..., 0, 1] = -np.exp(0.5j * (phi - omega)) * sine
    result[..., 1, 0] = np.exp(-0.5j * (phi - omega)) * sine
    result[..., 1, 1] = np.exp(0.5j * (phi + omega)) * cosine
    return result


def ry_matrices(theta: np.ndarray) -> np.ndarray:
    """Batched RY(theta)."""

    theta = np.asarray(theta, dtype=np.float64)
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    result = np.empty(theta.shape + (2, 2), dtype=np.complex128)
    result[..., 0, 0] = cosine
    result[..., 0, 1] = -sine
    result[..., 1, 0] = sine
    result[..., 1, 1] = cosine
    return result


def _cnot_permutation(qubits: int, control: int, target: int) -> np.ndarray:
    """Index permutation implementing CNOT on the amplitude vector."""

    indices = np.arange(2**qubits)
    control_bit = (indices >> (qubits - 1 - control)) & 1
    flipped = indices ^ (1 << (qubits - 1 - target))
    return np.where(control_bit == 1, flipped, indices)


def _entangler_ranges(layers: int, qubits: int) -> list[int]:
    """PennyLane's default StronglyEntanglingLayers ranges."""

    if qubits < 2:
        return [0] * layers
    return [(layer % (qubits - 1)) + 1 for layer in range(layers)]


def strongly_entangling(
    state: np.ndarray,
    weights: np.ndarray,
    qubits: int,
) -> np.ndarray:
    """Apply StronglyEntanglingLayers; ``weights`` is [n, layers, qubits, 3]."""

    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 4 or array.shape[2] != qubits or array.shape[3] != 3:
        raise ValueError("Weights must have shape [samples, layers, qubits, 3]")
    layers = array.shape[1]
    for layer, span in enumerate(_entangler_ranges(layers, qubits)):
        for qubit in range(qubits):
            state = _apply_batched_unitary(
                state,
                rot_matrices(
                    array[:, layer, qubit, 0],
                    array[:, layer, qubit, 1],
                    array[:, layer, qubit, 2],
                ),
                qubit,
                qubits,
            )
        if qubits > 1 and span > 0:
            for qubit in range(qubits):
                target = (qubit + span) % qubits
                if target == qubit:
                    continue
                state = state[:, _cnot_permutation(qubits, qubit, target)]
    return state


def _angle_embed(state: np.ndarray, inputs: np.ndarray, qubits: int) -> np.ndarray:
    """AngleEmbedding with RY rotations."""

    for qubit in range(qubits):
        state = _apply_batched_unitary(
            state, ry_matrices(inputs[:, qubit]), qubit, qubits
        )
    return state


# --------------------------------------------------------------------------
# the two ansätze actually used in v1.0 and v1.1
# --------------------------------------------------------------------------


def v1_0_states(inputs: np.ndarray, weights: np.ndarray, qubits: int = 4) -> np.ndarray:
    """v1.0 QuantumBottleneck: embed once, then all entangling layers."""

    state = _zero_state(weights.shape[0], qubits)
    state = _angle_embed(state, inputs, qubits)
    return strongly_entangling(state, weights, qubits)


def v1_1_states(inputs: np.ndarray, weights: np.ndarray, qubits: int = 4) -> np.ndarray:
    """v1.1 QuantumReuploadingBottleneck: re-embed before every single layer."""

    state = _zero_state(weights.shape[0], qubits)
    for block in range(weights.shape[1]):
        state = _angle_embed(state, inputs, qubits)
        state = strongly_entangling(
            state, weights[:, block : block + 1], qubits
        )
    return state


ANSATZE: dict[str, tuple[Callable[..., np.ndarray], int, int]] = {
    # name: (builder, default layers, parameter count at 4 qubits)
    "v1_0_bottleneck": (v1_0_states, 2, 24),
    "v1_1_reupload": (v1_1_states, 3, 36),
}

# Kept as a tuple for compatibility with the Part 3 API and tests. The graph
# ansatz is registered by Part 6 through ``register_graph_structured_ansatz``.
ANSATZ_NAMES = ("v1_0_bottleneck", "v1_1_reupload")


def register_graph_structured_ansatz(
    builder: Callable[..., np.ndarray],
    *,
    default_layers: int,
    parameter_count_at_4_qubits: int,
) -> None:
    """Register the Part 6 graph circuit without changing this diagnostic API.

    ``builder`` must accept ``(inputs, weights, qubits)`` and return a batched
    statevector. Its weights use the common ``[samples, layers, qubits, 3]``
    layout, which lets all three preregistered diagnostics run unchanged.
    """

    if not callable(builder):
        raise TypeError("The graph-structured ansatz builder must be callable")
    if default_layers < 1 or parameter_count_at_4_qubits < 1:
        raise ValueError("Graph ansatz layers and parameter count must be positive")
    ANSATZE[GRAPH_STRUCTURED_ANSATZ_NAME] = (
        builder,
        int(default_layers),
        int(parameter_count_at_4_qubits),
    )


def registered_ansatz_names() -> tuple[str, ...]:
    """Return implemented ansätze in fixed reporting order."""

    ordered = list(ANSATZ_NAMES)
    if GRAPH_STRUCTURED_ANSATZ_NAME in ANSATZE:
        ordered.append(GRAPH_STRUCTURED_ANSATZ_NAME)
    return tuple(ordered)


# --------------------------------------------------------------------------
# expressibility
# --------------------------------------------------------------------------


def haar_bin_probabilities(edges: np.ndarray, qubits: int) -> np.ndarray:
    """Exact Haar-random fidelity mass per bin.

    The Haar fidelity density is ``(N-1)(1-F)**(N-2)`` with ``N = 2**qubits``,
    whose CDF is ``1 - (1-F)**(N-1)``. Integrating the CDF beats sampling the
    density, which would add avoidable noise to the KL divergence.
    """

    dimension = 2**qubits
    cdf = 1.0 - np.power(1.0 - np.clip(edges, 0.0, 1.0), dimension - 1)
    return np.diff(cdf)


def fidelity_kl_divergence(
    fidelities: np.ndarray,
    qubits: int,
    bins: int = 75,
) -> float:
    """KL(empirical || Haar) over the fidelity distribution. Lower is more expressible."""

    values = np.clip(np.asarray(fidelities, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    empirical = counts / max(counts.sum(), 1)
    haar = haar_bin_probabilities(edges, qubits)
    floor = 1e-12
    empirical = np.clip(empirical, floor, None)
    haar = np.clip(haar, floor, None)
    empirical = empirical / empirical.sum()
    haar = haar / haar.sum()
    return float(np.sum(empirical * np.log(empirical / haar)))


def fidelity_histogram(
    fidelities: np.ndarray,
    qubits: int,
    bins: int = 75,
) -> dict[str, list[float]]:
    """Return aggregate empirical and exact Haar bin probabilities."""

    if bins < 2:
        raise ValueError("At least two fidelity bins are required")
    values = np.clip(np.asarray(fidelities, dtype=np.float64), 0.0, 1.0)
    if values.size < 1:
        raise ValueError("At least one fidelity is required")
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    empirical = counts.astype(np.float64) / counts.sum()
    haar = haar_bin_probabilities(edges, qubits)
    return {
        "bin_edges": edges.tolist(),
        "empirical_probability": empirical.tolist(),
        "haar_probability": haar.tolist(),
    }


def expressibility(
    ansatz: str,
    qubits: int = 4,
    layers: int | None = None,
    samples: int = 5000,
    seed: int = 42,
    fixed_input: bool = True,
    bins: int = 75,
) -> dict[str, Any]:
    """Sample parameter pairs and score the ansatz against Haar randomness."""

    if ansatz not in ANSATZE:
        raise ValueError(f"Unknown ansatz {ansatz!r}; expected {ANSATZ_NAMES}")
    builder, default_layers, _ = ANSATZE[ansatz]
    depth = default_layers if layers is None else layers
    generator = np.random.default_rng(seed)

    shape = (samples, depth, qubits, 3)
    left = generator.uniform(-np.pi, np.pi, size=shape)
    right = generator.uniform(-np.pi, np.pi, size=shape)
    if fixed_input:
        inputs = np.zeros((samples, qubits))
    else:
        inputs = generator.uniform(-np.pi, np.pi, size=(samples, qubits))

    first = builder(inputs, left, qubits)
    second = builder(inputs, right, qubits)
    fidelities = np.abs(np.sum(np.conj(first) * second, axis=1)) ** 2
    histogram = fidelity_histogram(fidelities, qubits, bins=bins)
    return {
        "ansatz": ansatz,
        "qubits": int(qubits),
        "layers": int(depth),
        "samples": int(samples),
        "seed": int(seed),
        "fixed_input": bool(fixed_input),
        "fidelity_histogram": histogram,
        "kl_divergence_from_haar": fidelity_kl_divergence(fidelities, qubits),
        "mean_fidelity": float(np.mean(fidelities)),
        "haar_mean_fidelity": float(1.0 / (2**qubits)),
    }


# --------------------------------------------------------------------------
# entangling capability
# --------------------------------------------------------------------------


def meyer_wallach(states: np.ndarray, qubits: int) -> np.ndarray:
    """Meyer-Wallach Q per state: ``2 (1 - mean_k Tr[rho_k^2])``.

    Zero for product states, one for maximally entangled ones.
    """

    array = np.asarray(states, dtype=np.complex128)
    samples = array.shape[0]
    purity_total = np.zeros(samples, dtype=np.float64)
    for qubit in range(qubits):
        left = 2**qubit
        right = 2 ** (qubits - qubit - 1)
        view = array.reshape(samples, left, 2, right)
        lower = view[:, :, 0, :].reshape(samples, -1)
        upper = view[:, :, 1, :].reshape(samples, -1)
        rho_00 = np.sum(np.abs(lower) ** 2, axis=1)
        rho_11 = np.sum(np.abs(upper) ** 2, axis=1)
        rho_01 = np.sum(np.conj(lower) * upper, axis=1)
        purity_total += rho_00**2 + rho_11**2 + 2.0 * np.abs(rho_01) ** 2
    return 2.0 * (1.0 - purity_total / qubits)


def entangling_capability(
    ansatz: str,
    qubits: int = 4,
    layers: int | None = None,
    samples: int = 5000,
    seed: int = 42,
    fixed_input: bool = True,
) -> dict[str, Any]:
    """Mean Meyer-Wallach Q over random parameter draws."""

    if ansatz not in ANSATZE:
        raise ValueError(f"Unknown ansatz {ansatz!r}; expected {ANSATZ_NAMES}")
    builder, default_layers, _ = ANSATZE[ansatz]
    depth = default_layers if layers is None else layers
    generator = np.random.default_rng(seed)
    weights = generator.uniform(-np.pi, np.pi, size=(samples, depth, qubits, 3))
    inputs = (
        np.zeros((samples, qubits))
        if fixed_input
        else generator.uniform(-np.pi, np.pi, size=(samples, qubits))
    )
    measure = meyer_wallach(builder(inputs, weights, qubits), qubits)
    measure = np.clip(measure, 0.0, 1.0)
    return {
        "ansatz": ansatz,
        "qubits": int(qubits),
        "layers": int(depth),
        "samples": int(samples),
        "seed": int(seed),
        "fixed_input": bool(fixed_input),
        "meyer_wallach_mean": float(np.mean(measure)),
        "meyer_wallach_std": float(np.std(measure, ddof=1)),
    }


# --------------------------------------------------------------------------
# barren plateaus
# --------------------------------------------------------------------------


def _expectation_z0(states: np.ndarray, qubits: int) -> np.ndarray:
    """<Z> on qubit 0, the standard barren-plateau cost."""

    samples = states.shape[0]
    view = states.reshape(samples, 2, 2 ** (qubits - 1))
    return np.sum(np.abs(view[:, 0, :]) ** 2 - np.abs(view[:, 1, :]) ** 2, axis=1)


def gradient_variance(
    ansatz: str,
    qubits: int,
    layers: int,
    samples: int = 200,
    seed: int = 42,
    parameter: tuple[int, int, int] = (0, 0, 1),
) -> dict[str, Any]:
    """Variance of d<Z_0>/d(one parameter) under random initialisation.

    Uses the exact parameter-shift rule: every parameter here drives a single
    rotation whose generator has eigenvalues +/-1/2, so the derivative is
    ``[f(theta + pi/2) - f(theta - pi/2)] / 2``.

    The parameter choice is not free. ``Rot(phi, theta, omega)`` expands to
    ``RZ(omega) RY(theta) RZ(phi)``, and the inputs here are zeros, so the state
    entering the first layer is exactly ``|0...0>``. A leading ``RZ`` on ``|0>``
    contributes only a global phase, which makes the gradient with respect to
    ``phi`` at ``(0, wire, 0)`` identically zero and any variance computed from
    it pure round-off. The default therefore targets the ``RY`` component, and
    ``degenerate`` flags the case anyway so a silently meaningless number cannot
    reach a figure.
    """

    if ansatz not in ANSATZE:
        raise ValueError(f"Unknown ansatz {ansatz!r}; expected {ANSATZ_NAMES}")
    if samples < 2:
        raise ValueError("At least two gradient samples are required")
    builder, _, _ = ANSATZE[ansatz]
    layer, wire, component = parameter
    if layer >= layers or wire >= qubits or component >= 3:
        raise ValueError("Parameter index is outside the weight tensor")

    generator = np.random.default_rng(seed)
    weights = generator.uniform(-np.pi, np.pi, size=(samples, layers, qubits, 3))
    inputs = np.zeros((samples, qubits))

    plus = weights.copy()
    plus[:, layer, wire, component] += np.pi / 2.0
    minus = weights.copy()
    minus[:, layer, wire, component] -= np.pi / 2.0

    forward = _expectation_z0(builder(inputs, plus, qubits), qubits)
    backward = _expectation_z0(builder(inputs, minus, qubits), qubits)
    gradients = (forward - backward) / 2.0
    return {
        "ansatz": ansatz,
        "qubits": int(qubits),
        "layers": int(layers),
        "samples": int(samples),
        "seed": int(seed),
        "fixed_parameter": [int(layer), int(wire), int(component)],
        "gradient_variance": float(np.var(gradients, ddof=1)),
        "gradient_mean": float(np.mean(gradients)),
        "gradient_abs_mean": float(np.mean(np.abs(gradients))),
        "degenerate": bool(np.mean(np.abs(gradients)) < 1e-12),
    }


def barren_plateau_slope(qubit_counts: list[int], variances: list[float]) -> dict[str, Any]:
    """Fit log(Var) against qubit count; a steep negative slope is the signature.

    Variance decaying like ``2**(-k n)`` gives a slope of ``-k log 2`` per qubit,
    so ``decay_base_per_qubit`` below about 0.5 means the gradient roughly halves
    with each added qubit and training becomes hopeless at scale.
    """

    counts = np.asarray(qubit_counts, dtype=np.float64)
    values = np.asarray(variances, dtype=np.float64)
    usable = values > 0
    if usable.sum() < 2:
        return {"slope": None, "decay_base_per_qubit": None, "barren": None}
    slope, intercept = np.polyfit(counts[usable], np.log(values[usable]), 1)
    return {
        "slope_log_variance_per_qubit": float(slope),
        "intercept": float(intercept),
        "decay_base_per_qubit": float(np.exp(slope)),
        "barren": bool(slope < -0.3),
    }


# --------------------------------------------------------------------------
# encoding diagnostics — the follow-up Part 2 asks for
# --------------------------------------------------------------------------


def encoding_expressibility(
    encoding: str,
    qubits: int = 4,
    samples: int = 5000,
    seed: int = 42,
    **options: Any,
) -> dict[str, Any]:
    """Expressibility of a *feature map* over random data rather than parameters.

    Part 2 found the geometric difference depends strongly on the encoding, with
    angle encoding far below the advantage threshold and IQP above it. This
    measures how much of state space each encoding actually reaches, which is
    the mechanism that would explain that gap.
    """

    from .kernels import statevectors

    generator = np.random.default_rng(seed)
    left = generator.uniform(-np.pi, np.pi, size=(samples, qubits))
    right = generator.uniform(-np.pi, np.pi, size=(samples, qubits))
    first = statevectors(left, qubits, encoding, **options)
    second = statevectors(right, qubits, encoding, **options)
    fidelities = np.abs(np.sum(np.conj(first) * second, axis=1)) ** 2
    entanglement = meyer_wallach(first, qubits)
    return {
        "encoding": encoding,
        "qubits": int(qubits),
        "samples": int(samples),
        "kl_divergence_from_haar": fidelity_kl_divergence(fidelities, qubits),
        "mean_fidelity": float(np.mean(fidelities)),
        "haar_mean_fidelity": float(1.0 / (2**qubits)),
        "meyer_wallach_mean": float(np.mean(entanglement)),
    }
