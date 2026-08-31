"""Quantum and classical kernels, and the geometric difference between them.

Part 2 of the Objective 3 v2.0 study asks a question that needs no training: on
*this* data, could a quantum model outperform the best classical one at all?

Huang et al. (Nat. Commun. 12:2631, 2021) answer it with the geometric
difference between the classical and quantum kernel matrices,

    g(K_C || K_Q) = sqrt( || sqrt(K_Q) K_C^-1 sqrt(K_Q) ||_inf )

evaluated on kernels normalised so that ``trace(K) = N``. The logic runs one
way only: a small ``g`` relative to ``sqrt(N)`` rules a quantum advantage out,
while a large ``g`` is necessary but not sufficient for one. That asymmetry is
what makes the measurement useful here — it can explain the v1.1 null rather
than merely restate it.

Statevectors are built directly in NumPy rather than by simulating one circuit
per sample: the feature maps used here have closed forms, and the fidelity Gram
matrix is then a single matrix product. ``tests/test_objective3_v2_kernels.py``
cross-checks every construction against PennyLane 0.45.1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import linalg

ENCODINGS = ("angle", "iqp")


# --------------------------------------------------------------------------
# statevector feature maps
# --------------------------------------------------------------------------


def _as_batch(features: np.ndarray, qubits: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Features must be a two-dimensional [samples, qubits] array")
    if values.shape[1] != qubits:
        raise ValueError(
            f"Expected {qubits} features per sample, received {values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    return values


def _apply_single_qubit(state: np.ndarray, gate: np.ndarray, qubit: int, qubits: int) -> np.ndarray:
    """Apply a 2x2 gate to ``qubit`` for a batch of states of shape [n, 2**q]."""

    samples = state.shape[0]
    left = 2**qubit
    right = 2 ** (qubits - qubit - 1)
    view = state.reshape(samples, left, 2, right)
    return np.einsum("ab,nibj->niaj", gate, view).reshape(samples, 2**qubits)


def _apply_batched_ry(state: np.ndarray, angles: np.ndarray, qubit: int, qubits: int) -> np.ndarray:
    """Apply RY(theta) with a per-sample angle to ``qubit``."""

    samples = state.shape[0]
    left = 2**qubit
    right = 2 ** (qubits - qubit - 1)
    half = angles / 2.0
    cosine = np.cos(half)[:, None, None]
    sine = np.sin(half)[:, None, None]
    view = state.reshape(samples, left, 2, right)
    lower, upper = view[:, :, 0, :], view[:, :, 1, :]
    result = np.empty_like(view)
    result[:, :, 0, :] = cosine * lower - sine * upper
    result[:, :, 1, :] = sine * lower + cosine * upper
    return result.reshape(samples, 2**qubits)


def _basis_signs(qubits: int) -> np.ndarray:
    """Return the [2**q, q] matrix of Pauli-Z eigenvalues (+1 for |0>, -1 for |1>)."""

    indices = np.arange(2**qubits)
    bits = ((indices[:, None] >> np.arange(qubits - 1, -1, -1)[None, :]) & 1).astype(
        np.float64
    )
    return 1.0 - 2.0 * bits


def angle_statevectors(
    features: np.ndarray,
    qubits: int,
    layers: int = 2,
) -> np.ndarray:
    """Angle (RY) embedding interleaved with a fixed CZ ring.

    The entangling ring matters: without it the state stays a product state and
    the fidelity kernel factorises into a product of single-qubit overlaps, which
    no quantum model could exploit.
    """

    values = _as_batch(features, qubits)
    samples = values.shape[0]
    state = np.zeros((samples, 2**qubits), dtype=np.complex128)
    state[:, 0] = 1.0
    signs = _basis_signs(qubits)
    for _ in range(max(1, layers)):
        for qubit in range(qubits):
            state = _apply_batched_ry(state, values[:, qubit], qubit, qubits)
        if qubits > 1:
            for qubit in range(qubits):
                partner = (qubit + 1) % qubits
                if qubits == 2 and qubit == 1:
                    continue
                both_one = (signs[:, qubit] < 0) & (signs[:, partner] < 0)
                state = state * np.where(both_one, -1.0, 1.0)[None, :]
    return state


def iqp_statevectors(
    features: np.ndarray,
    qubits: int,
    repeats: int = 2,
) -> np.ndarray:
    """IQP-style feature map of Havlicek et al. (Nature 567:209, 2019).

    Each repeat applies Hadamards followed by the diagonal unitary
    ``exp(i [sum_k x_k Z_k + sum_{k<l} x_k x_l Z_k Z_l])``. The quadratic term is
    evaluated in closed form from the linear one, since ``Z`` eigenvalues square
    to one:  ``sum_{k<l} x_k x_l z_k z_l = ((sum_k x_k z_k)^2 - sum_k x_k^2) / 2``.
    """

    values = _as_batch(features, qubits)
    samples = values.shape[0]
    signs = _basis_signs(qubits)
    linear = values @ signs.T
    square_sum = np.sum(values**2, axis=1)[:, None]
    phase = np.exp(1j * (linear + 0.5 * (linear**2 - square_sum)))
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)

    state = np.zeros((samples, 2**qubits), dtype=np.complex128)
    state[:, 0] = 1.0
    for _ in range(max(1, repeats)):
        for qubit in range(qubits):
            state = _apply_single_qubit(state, hadamard, qubit, qubits)
        state = state * phase
    return state


def statevectors(
    features: np.ndarray,
    qubits: int,
    encoding: str = "angle",
    **options: Any,
) -> np.ndarray:
    """Dispatch to the requested feature map."""

    if encoding == "angle":
        return angle_statevectors(features, qubits, **options)
    if encoding == "iqp":
        return iqp_statevectors(features, qubits, **options)
    raise ValueError(f"Unknown encoding {encoding!r}; expected one of {ENCODINGS}")


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


def normalise_trace(kernel: np.ndarray) -> np.ndarray:
    """Rescale so ``trace(K) = N``, the convention the geometric difference assumes."""

    matrix = np.asarray(kernel, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A kernel must be a square matrix")
    trace = float(np.trace(matrix))
    if not np.isfinite(trace) or abs(trace) < 1e-12:
        raise ValueError("Kernel trace is zero or non-finite; cannot normalise")
    return matrix * (matrix.shape[0] / trace)


def reduce_to_qubits(values: np.ndarray, qubits: int, seed: int = 42) -> np.ndarray:
    """PCA-reduce embeddings to one feature per qubit and scale into [-pi, pi].

    The 160-dimensional embeddings cannot be angle-encoded onto a handful of
    qubits without compression, and the compression is part of what is being
    measured, so it is reported alongside the result.
    """

    from sklearn.decomposition import PCA

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional")
    if array.shape[1] < qubits:
        raise ValueError("Fewer embedding dimensions than qubits")
    reducer = PCA(n_components=qubits, random_state=seed)
    reduced = reducer.fit_transform(array)
    spread = np.max(np.abs(reduced), axis=0)
    spread[spread < 1e-12] = 1.0
    return np.pi * reduced / spread


def classical_kernels(values: np.ndarray) -> dict[str, np.ndarray]:
    """Candidate classical kernels, each trace-normalised to N.

    The comparison must be against the *best* classical kernel, so a family is
    returned rather than a single choice.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional")
    dimension = array.shape[1]
    squared = np.sum(array**2, axis=1)
    distances = np.maximum(
        squared[:, None] + squared[None, :] - 2.0 * (array @ array.T), 0.0
    )
    kernels = {"linear": normalise_trace(array @ array.T)}
    for scale in (0.1, 1.0, 10.0):
        gamma = scale / dimension
        kernels[f"rbf_gamma_{scale:g}_over_d"] = normalise_trace(
            np.exp(-gamma * distances)
        )
    return kernels


def fidelity_kernel(states: np.ndarray) -> np.ndarray:
    """Quantum fidelity kernel ``K[i,j] = |<psi_i|psi_j>|**2``, trace-normalised."""

    array = np.asarray(states, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError("States must be [samples, amplitudes]")
    overlap = array.conj() @ array.T
    return normalise_trace(np.abs(overlap) ** 2)


def bloch_features(states: np.ndarray, qubits: int) -> np.ndarray:
    """Per-qubit Bloch vectors ``(<X>, <Y>, <Z>)`` for each state, shape [n, 3q]."""

    array = np.asarray(states, dtype=np.complex128)
    samples = array.shape[0]
    if array.shape[1] != 2**qubits:
        raise ValueError("State dimension does not match the qubit count")
    out = np.empty((samples, 3 * qubits), dtype=np.float64)
    for qubit in range(qubits):
        left = 2**qubit
        right = 2 ** (qubits - qubit - 1)
        view = array.reshape(samples, left, 2, right)
        lower = view[:, :, 0, :].reshape(samples, -1)
        upper = view[:, :, 1, :].reshape(samples, -1)
        cross = np.sum(np.conj(lower) * upper, axis=1)
        out[:, 3 * qubit + 0] = 2.0 * np.real(cross)
        out[:, 3 * qubit + 1] = -2.0 * np.imag(cross)
        out[:, 3 * qubit + 2] = np.sum(np.abs(lower) ** 2 - np.abs(upper) ** 2, axis=1)
    return out


def projected_quantum_kernel(
    states: np.ndarray,
    qubits: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """Projected quantum kernel of Huang et al. (2021), trace-normalised.

    ``K[i,j] = exp(-gamma * sum_k || rho_i^(k) - rho_j^(k) ||_F^2)`` over the
    reduced single-qubit density matrices. Writing ``rho = (I + r . sigma) / 2``
    and using ``Tr(sigma_a sigma_b) = 2 delta_ab`` gives
    ``|| rho_i - rho_j ||_F^2 = |r_i - r_j|^2 / 2``, so the kernel is an RBF on
    the stacked Bloch vectors — exact, and far cheaper than forming the density
    matrices explicitly.
    """

    features = bloch_features(states, qubits)
    squared = np.sum(features**2, axis=1)
    distances = np.maximum(
        squared[:, None] + squared[None, :] - 2.0 * (features @ features.T), 0.0
    )
    return normalise_trace(np.exp(-gamma * distances / 2.0))


# --------------------------------------------------------------------------
# geometric difference
# --------------------------------------------------------------------------


def _psd_sqrt(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Symmetric square root, clipping the negative eigenvalues of round-off."""

    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, floor, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


@dataclass(frozen=True)
class GeometricDifference:
    """One geometric-difference measurement at a single regularisation."""

    samples: int
    regularisation: float
    geometric_difference: float
    sqrt_samples: float
    ratio: float
    advantage_possible: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def geometric_difference(
    classical: np.ndarray,
    quantum: np.ndarray,
    regularisation: float = 1e-6,
    advantage_threshold: float = 0.5,
    quantum_sqrt: np.ndarray | None = None,
) -> GeometricDifference:
    """``g(K_C || K_Q) = sqrt(|| sqrt(K_Q) (K_C + lam I)^-1 sqrt(K_Q) ||_inf)``.

    ``regularisation`` is Tikhonov damping on the classical inverse; the
    unregularised definition is the ``lam -> 0`` limit. Because kernel spectra
    here decay steeply, a single ``lam`` is not reportable — sweep it with
    :func:`geometric_difference_sweep` and report the curve.

    ``ratio`` is ``g / sqrt(N)``. Below ``advantage_threshold`` no quantum
    advantage is achievable on this data; above it, one is possible but not
    implied.

    ``quantum_sqrt`` lets a caller reuse ``sqrt(K_Q)`` across many classical
    kernels and regularisations; recomputing it is the dominant cost.
    """

    left = np.asarray(classical, dtype=np.float64)
    right = np.asarray(quantum, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("Both kernels must be square and the same shape")
    if regularisation <= 0:
        raise ValueError("Regularisation must be positive")
    samples = left.shape[0]

    root = _psd_sqrt(right) if quantum_sqrt is None else np.asarray(
        quantum_sqrt, dtype=np.float64
    )
    if root.shape != right.shape:
        raise ValueError("Supplied quantum square root has the wrong shape")
    damped = (left + left.T) / 2.0 + regularisation * np.eye(samples)
    solved = linalg.solve(damped, root, assume_a="pos")
    product = root @ solved
    product = (product + product.T) / 2.0
    spectral = float(np.max(linalg.eigvalsh(product)))
    spectral = max(spectral, 0.0)
    value = float(np.sqrt(spectral))
    ratio = value / np.sqrt(samples)
    return GeometricDifference(
        samples=int(samples),
        regularisation=float(regularisation),
        geometric_difference=value,
        sqrt_samples=float(np.sqrt(samples)),
        ratio=float(ratio),
        advantage_possible=bool(ratio >= advantage_threshold),
    )


def quantum_sqrt(quantum: np.ndarray) -> np.ndarray:
    """Precompute ``sqrt(K_Q)`` for reuse across classical kernels and lambdas."""

    return _psd_sqrt(np.asarray(quantum, dtype=np.float64))


def geometric_difference_sweep(
    classical: np.ndarray,
    quantum: np.ndarray,
    regularisations: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2),
    advantage_threshold: float = 0.5,
    precomputed_sqrt: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Geometric difference across a range of regularisation strengths."""

    root = quantum_sqrt(quantum) if precomputed_sqrt is None else precomputed_sqrt
    return [
        geometric_difference(
            classical, quantum, value, advantage_threshold, quantum_sqrt=root
        ).as_dict()
        for value in regularisations
    ]


def concentration_diagnostic(kernel: np.ndarray) -> dict[str, float]:
    """Measure how close a kernel has drifted towards the identity matrix.

    Fidelity kernels concentrate exponentially with qubit count: off-diagonal
    entries collapse towards zero, every point becomes near-orthogonal to every
    other, and the kernel stops carrying usable similarity information. A
    geometric difference computed on a concentrated kernel is not evidence of a
    useful separation, so this diagnostic must be reported beside it.

    ``effective_rank`` is the exponential of the von Neumann entropy of the
    normalised spectrum: it equals N for the identity and falls as structure
    appears.
    """

    matrix = np.asarray(kernel, dtype=np.float64)
    samples = matrix.shape[0]
    mask = ~np.eye(samples, dtype=bool)
    off_diagonal = matrix[mask]
    eigenvalues = np.clip(linalg.eigvalsh((matrix + matrix.T) / 2.0), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0:
        effective_rank = 0.0
    else:
        weights = eigenvalues / total
        weights = weights[weights > 1e-15]
        effective_rank = float(np.exp(-np.sum(weights * np.log(weights))))
    return {
        "off_diagonal_mean": float(np.mean(off_diagonal)),
        "off_diagonal_std": float(np.std(off_diagonal)),
        "diagonal_mean": float(np.mean(np.diag(matrix))),
        "effective_rank": effective_rank,
        "effective_rank_fraction": float(effective_rank / samples),
        "concentrated": bool(float(np.mean(off_diagonal)) < 0.01),
    }


def interpret(ratios: list[float], threshold: float = 0.5) -> str:
    """One paragraph stating what the measurement licenses, and what it does not."""

    if not ratios:
        raise ValueError("At least one ratio is required")
    largest = max(ratios)
    if largest < threshold:
        return (
            f"Across every configuration tested, the geometric difference stayed "
            f"below {threshold:g} times sqrt(N) (largest observed ratio "
            f"{largest:.4f}). On this data and with these feature maps, no "
            "separation between the quantum and the best classical kernel is "
            "achievable, so no quantum model of this family could have "
            "outperformed its classical counterpart. The v1.1 null is therefore "
            "the predicted outcome rather than an unexplained negative result."
        )
    return (
        f"At least one configuration reached a geometric difference of "
        f"{largest:.4f} times sqrt(N), above the {threshold:g} threshold. A "
        "separation is therefore not ruled out by the data geometry. This is a "
        "necessary but not a sufficient condition: a large geometric difference "
        "permits a quantum advantage without implying one, and the empirical "
        "comparison still decides the question."
    )
