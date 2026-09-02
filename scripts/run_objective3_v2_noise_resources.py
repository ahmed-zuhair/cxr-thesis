#!/usr/bin/env python3
"""Part 7: what would it take to run this on real hardware, and would it survive?

Two analyses on the Part 6 circuit, both cheap and both things an examiner asks:

* **Noise sensitivity.** Re-evaluate a noiseless-trained circuit under
  depolarising and readout noise and finite shots. The model is NOT retrained
  under noise; this measures how far a simulator result would degrade if run on
  a device, not how well a noise-aware model could do.
* **Resource estimation.** Qubits, depth, gate counts, and the shots needed for a
  given precision, extrapolated to a wall-clock estimate under stated hardware
  assumptions. Every assumption is recorded in the output rather than buried.

No data, no labels, no training. The locked test cohort is never opened.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective3_v2 import STUDY, VERSION
from cxr_thesis.objective3_v2.guards import assert_no_locked_test
from cxr_thesis.objective3_v2.io_utils import sha256_file, write_results
from cxr_thesis.objective3_v2.seeds import seed_everything

PART = "part7_noise_and_resources"
DEPOLARISING = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
READOUT = (0.0, 1e-3, 1e-2, 5e-2)
SHOTS = (100, 1000, 10000, None)

# Stated openly so the estimate can be checked or rescaled, rather than
# presented as though it were measured. These are order-of-magnitude figures
# typical of current superconducting hardware.
HARDWARE = {
    "single_qubit_gate_seconds": 25e-9,
    "two_qubit_gate_seconds": 300e-9,
    "readout_seconds": 1.0e-6,
    "reset_seconds": 250e-9,
    "classical_latency_per_circuit_seconds": 1.0e-3,
    "source": "order-of-magnitude values for current superconducting devices",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--supernodes", type=int, default=4, choices=(2, 4))
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--cases", type=int, default=256)
    parser.add_argument("--validation-cases", type=int, default=5000)
    parser.add_argument("--precision", type=float, default=0.01)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def circuit_expectations(
    angles: np.ndarray,
    adjacency: np.ndarray,
    rotations: np.ndarray,
    couplings: np.ndarray,
    depolarising: float = 0.0,
    readout: float = 0.0,
    shots: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Pauli-Z expectations for the Part 6 circuit under an optional noise model."""

    import pennylane as qml

    supernodes = angles.shape[1]
    layers = rotations.shape[0]
    pairs = [(i, j) for i in range(supernodes) for j in range(i + 1, supernodes)]
    wires = tuple(range(supernodes))
    if depolarising > 0 or readout > 0 or shots is not None:
        # Shot sampling is stochastic. Without an explicit device seed the
        # finite-shot rows change between runs even though the results JSON
        # records a seed, which makes a published number irreproducible.
        try:
            device = qml.device(
                "default.mixed", wires=supernodes, shots=shots, seed=seed
            )
        except TypeError:  # older PennyLane devices take no seed argument
            np.random.seed(seed)
            device = qml.device("default.mixed", wires=supernodes, shots=shots)
    else:
        device = qml.device("default.qubit", wires=supernodes)

    @qml.qnode(device)
    def circuit(sample_angles, weights):
        for layer in range(layers):
            qml.AngleEmbedding(sample_angles, wires=wires, rotation="Y")
            for wire in wires:
                qml.Rot(*rotations[layer, wire], wires=wire)
            for index, (left, right) in enumerate(pairs):
                qml.CRZ(
                    float(couplings[layer, index] * weights[left, right]),
                    wires=[left, right],
                )
            if depolarising > 0:
                for wire in wires:
                    qml.DepolarizingChannel(depolarising, wires=wire)
        if readout > 0:
            for wire in wires:
                qml.BitFlip(readout, wires=wire)
        return [qml.expval(qml.PauliZ(wire)) for wire in wires]

    return np.array(
        [
            np.asarray(circuit(angles[index], adjacency[index]), dtype=float)
            for index in range(angles.shape[0])
        ]
    )


def noise_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Degradation of the circuit output under noise, relative to the ideal run."""

    generator = np.random.default_rng(args.seed)
    supernodes = args.supernodes
    angles = generator.uniform(-np.pi, np.pi, size=(args.cases, supernodes))
    adjacency = np.zeros((args.cases, supernodes, supernodes))
    upper = generator.uniform(size=(args.cases, supernodes, supernodes))
    for index in range(args.cases):
        triangle = np.triu(upper[index], 1)
        adjacency[index] = triangle + triangle.T
    rotations = generator.uniform(-np.pi, np.pi, size=(args.layers, supernodes, 3))
    pairs = supernodes * (supernodes - 1) // 2
    couplings = generator.uniform(-np.pi, np.pi, size=(args.layers, pairs))

    ideal = circuit_expectations(angles, adjacency, rotations, couplings)
    rows = []
    for probability in DEPOLARISING:
        noisy = circuit_expectations(
            angles, adjacency, rotations, couplings, depolarising=probability
        )
        rows.append(
            {
                "channel": "depolarising",
                "probability": float(probability),
                "shots": None,
                "mean_absolute_deviation": float(np.mean(np.abs(noisy - ideal))),
                "max_absolute_deviation": float(np.max(np.abs(noisy - ideal))),
                "correlation_with_ideal": float(
                    np.corrcoef(noisy.ravel(), ideal.ravel())[0, 1]
                ),
            }
        )
    for probability in READOUT:
        noisy = circuit_expectations(
            angles, adjacency, rotations, couplings, readout=probability
        )
        rows.append(
            {
                "channel": "readout_bitflip",
                "probability": float(probability),
                "shots": None,
                "mean_absolute_deviation": float(np.mean(np.abs(noisy - ideal))),
                "max_absolute_deviation": float(np.max(np.abs(noisy - ideal))),
                "correlation_with_ideal": float(
                    np.corrcoef(noisy.ravel(), ideal.ravel())[0, 1]
                ),
            }
        )
    for count in SHOTS:
        sampled = circuit_expectations(
            angles[:64], adjacency[:64], rotations, couplings,
            shots=count, seed=args.seed,
        )
        reference = ideal[:64]
        rows.append(
            {
                "channel": "finite_shots",
                "probability": 0.0,
                "shots": count,
                "mean_absolute_deviation": float(
                    np.mean(np.abs(sampled - reference))
                ),
                "max_absolute_deviation": float(np.max(np.abs(sampled - reference))),
                "correlation_with_ideal": float(
                    np.corrcoef(sampled.ravel(), reference.ravel())[0, 1]
                ),
            }
        )
    return {
        "rows": rows,
        "retrained_under_noise": False,
        "note": (
            "The circuit was trained noiselessly and evaluated under noise. This "
            "bounds how far a simulator result would degrade on a device; it does "
            "not describe what a noise-aware model could achieve."
        ),
    }


def resource_table(args: argparse.Namespace, noise: dict[str, Any]) -> dict[str, Any]:
    """Gate counts, shots for a target precision, and an extrapolated wall clock."""

    supernodes, layers = args.supernodes, args.layers
    pairs = supernodes * (supernodes - 1) // 2
    single_gates = layers * (supernodes + supernodes)  # embedding + rotations
    two_qubit_gates = layers * pairs
    depth = layers * (2 + pairs)

    # Shot noise on a Pauli-Z expectation has standard error <= 1/sqrt(shots),
    # so the shots needed for a target precision follow directly.
    shots_needed = int(np.ceil(1.0 / (args.precision**2)))

    seconds_per_circuit = (
        single_gates * HARDWARE["single_qubit_gate_seconds"]
        + two_qubit_gates * HARDWARE["two_qubit_gate_seconds"]
        + supernodes * HARDWARE["readout_seconds"]
        + supernodes * HARDWARE["reset_seconds"]
    )
    per_case = shots_needed * seconds_per_circuit + HARDWARE[
        "classical_latency_per_circuit_seconds"
    ]
    validation_hours = per_case * args.validation_cases / 3600.0

    started = time.perf_counter()
    generator = np.random.default_rng(0)
    angles = generator.uniform(-np.pi, np.pi, size=(32, supernodes))
    adjacency = np.zeros((32, supernodes, supernodes))
    rotations = generator.uniform(-np.pi, np.pi, size=(layers, supernodes, 3))
    couplings = generator.uniform(-np.pi, np.pi, size=(layers, pairs))
    circuit_expectations(angles, adjacency, rotations, couplings)
    simulator_seconds_per_case = (time.perf_counter() - started) / 32

    return {
        "qubits": supernodes,
        "layers": layers,
        "circuit_depth": depth,
        "single_qubit_gates": single_gates,
        "two_qubit_gates": two_qubit_gates,
        "target_precision": args.precision,
        "shots_for_target_precision": shots_needed,
        "seconds_per_circuit_execution": seconds_per_circuit,
        "seconds_per_case_on_hardware": per_case,
        "hours_for_one_validation_pass_on_hardware": validation_hours,
        "simulator_seconds_per_case_measured": simulator_seconds_per_case,
        "simulator_hours_for_one_validation_pass": (
            simulator_seconds_per_case * args.validation_cases / 3600.0
        ),
        "validation_cases": args.validation_cases,
        "assumptions": HARDWARE,
        "caveat": (
            "Wall clock on hardware is estimated from stated gate times, not "
            "measured. Queueing, transpilation overhead and calibration drift "
            "are excluded and would dominate in practice."
        ),
    }


def write_figures(results: dict[str, Any], output: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    rows = results["noise"]["rows"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for channel in ("depolarising", "readout_bitflip"):
        selected = [r for r in rows if r["channel"] == channel and r["probability"] > 0]
        if selected:
            axes[0].semilogx(
                [r["probability"] for r in selected],
                [r["mean_absolute_deviation"] for r in selected],
                marker="o",
                label=channel,
            )
    axes[0].set_xlabel("error probability")
    axes[0].set_ylabel("mean |noisy - ideal| expectation")
    axes[0].set_title("Noise sensitivity")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    shots = [r for r in rows if r["channel"] == "finite_shots" and r["shots"]]
    axes[1].loglog(
        [r["shots"] for r in shots],
        [r["mean_absolute_deviation"] for r in shots],
        marker="o",
        label="measured",
    )
    counts = np.array([r["shots"] for r in shots], dtype=float)
    axes[1].loglog(counts, 1.0 / np.sqrt(counts), "--", label=r"$1/\sqrt{shots}$")
    axes[1].set_xlabel("shots")
    axes[1].set_ylabel("mean |sampled - exact|")
    axes[1].set_title("Shot noise")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    path = output / "noise_and_shots.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return {"noise_and_shots": sha256_file(path)}


def main() -> None:
    args = parse_args()
    output = assert_no_locked_test(args.output_dir)
    if args.smoke:
        args.cases, args.validation_cases = 16, 100
    seed_record = seed_everything(args.seed)

    print("Sweeping noise channels...", flush=True)
    noise = noise_sweep(args)
    print("Estimating resources...", flush=True)
    resources = resource_table(args, noise)
    results = {"noise": noise, "resources": resources, "data_used": False}

    artifact_hashes = write_figures(results, output) if args.figures else {}
    path, digest = write_results(
        output / "results.json",
        study=STUDY,
        part=PART,
        config={
            "version": VERSION,
            "supernodes": args.supernodes,
            "layers": args.layers,
            "cases": args.cases,
            "precision": args.precision,
            "smoke": bool(args.smoke),
            "seeding": seed_record,
            "labels_used": False,
            "models_trained": False,
        },
        results=results,
        artifact_hashes=artifact_hashes,
        seed=args.seed,
        locked_test_accessed=False,
    )

    print("")
    print(f"{'channel':>16} {'p / shots':>12} {'mean |dev|':>11} {'corr':>7}")
    for row in noise["rows"]:
        marker = row["shots"] if row["channel"] == "finite_shots" else row["probability"]
        print(
            f"{row['channel']:>16} {str(marker):>12} "
            f"{row['mean_absolute_deviation']:>11.4f} "
            f"{row['correlation_with_ideal']:>7.4f}"
        )
    print("")
    print("--- RESOURCES ---")
    for key in (
        "qubits", "circuit_depth", "single_qubit_gates", "two_qubit_gates",
        "shots_for_target_precision", "hours_for_one_validation_pass_on_hardware",
        "simulator_hours_for_one_validation_pass",
    ):
        print(f"  {key:>44}: {resources[key]}")
    print(f"\n  {resources['caveat']}")
    print("")
    print(f"Results: {path}")
    print(f"Results SHA-256: {digest}")
    print("Data used: False | Labels used: False | Locked test accessed: False")


if __name__ == "__main__":
    main()
