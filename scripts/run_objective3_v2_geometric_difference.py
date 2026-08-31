#!/usr/bin/env python3
"""Part 2: can a quantum advantage exist on this data at all?

Computes the geometric difference between the best classical kernel and the
quantum kernels on the frozen Objective 2 GAT embeddings, following Huang et al.
(Nat. Commun. 12:2631, 2021). No model is trained and no label is used.

The headline number is ``min_ratio`` per configuration: the geometric difference
against the *closest* classical kernel in the family, divided by sqrt(N). The
minimum is the conservative choice, because a quantum kernel that some classical
kernel already reproduces cannot yield an advantage over the family.

Training and validation embeddings only. The locked test cohort is never opened.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective3_v2 import STUDY, VERSION
from cxr_thesis.objective3_v2.guards import assert_no_locked_test, require_existing
from cxr_thesis.objective3_v2.io_utils import read_json, sha256_file, write_results
from cxr_thesis.objective3_v2.kernels import (
    ENCODINGS,
    classical_kernels,
    concentration_diagnostic,
    fidelity_kernel,
    geometric_difference_sweep,
    interpret,
    quantum_sqrt,
    projected_quantum_kernel,
    reduce_to_qubits,
    statevectors,
)
from cxr_thesis.objective3_v2.seeds import seed_everything

DEFAULT_QUBITS = (4, 6, 8)
DEFAULT_REGULARISATIONS = (1e-8, 1e-6, 1e-4, 1e-2)
ADVANTAGE_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--recovery-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=list(DEFAULT_QUBITS),
        help="Qubit counts to sweep",
    )
    parser.add_argument("--pqk-gamma", type=float, default=1.0)
    parser.add_argument("--angle-layers", type=int, default=2)
    parser.add_argument("--iqp-repeats", type=int, default=2)
    parser.add_argument("--figure", action="store_true", help="Write the sweep figure")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on small synthetic embeddings, no data required",
    )
    return parser.parse_args()


def load_embeddings(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, str]]:
    """Load train+validation embedding shards, or synthesise them for the smoke run."""

    if args.smoke:
        generator = np.random.default_rng(args.seed)
        latent = generator.normal(size=(args.samples, 12))
        mixing = generator.normal(size=(12, 160))
        noise = 0.3 * generator.normal(size=(args.samples, 160))
        return (latent @ mixing + noise).astype(np.float32), {}

    if args.embedding_root is None or args.recovery_index is None:
        raise ValueError(
            "--embedding-root and --recovery-index are required unless --smoke is set"
        )
    root = assert_no_locked_test(args.embedding_root)
    index_path = assert_no_locked_test(args.recovery_index)
    require_existing([root, index_path])

    index = read_json(index_path)
    records = sorted(index.get("shards", []), key=lambda item: int(item["start"]))
    if not records:
        raise ValueError("The embedding recovery index lists no shards")

    from cxr_thesis.objective3.embeddings import load_embedding_shard

    hashes: dict[str, str] = {"recovery_index": sha256_file(index_path)}
    blocks: list[np.ndarray] = []
    expected_start = 0
    for record in records:
        start, stop = int(record["start"]), int(record["stop"])
        if start != expected_start or stop <= start:
            raise ValueError("Embedding shards are not contiguous")
        shard_path = assert_no_locked_test(root / f"{record['shard']}.npz")
        embeddings, _ = load_embedding_shard(shard_path)
        if len(embeddings) != stop - start:
            raise ValueError(f"Shard {record['shard']} case count disagrees with index")
        blocks.append(np.asarray(embeddings, dtype=np.float32))
        expected_start = stop
    values = np.concatenate(blocks, axis=0)
    hashes["embedding_shards"] = str(len(records))
    return values, hashes


def subsample(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Draw a reproducible subsample; kernels are O(N^2) so N is capped."""

    if count >= values.shape[0]:
        return values
    generator = np.random.default_rng(seed)
    chosen = generator.choice(values.shape[0], size=count, replace=False)
    return values[np.sort(chosen)]


def evaluate(values: np.ndarray, args: argparse.Namespace) -> dict[str, object]:
    """Sweep qubits x encoding x quantum-kernel-type x classical-kernel x lambda."""

    classical = classical_kernels(values)
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, dict[str, float]] = {}

    for qubits in args.qubits:
        reduced = reduce_to_qubits(values, qubits, seed=args.seed)
        for encoding in ENCODINGS:
            options = (
                {"layers": args.angle_layers}
                if encoding == "angle"
                else {"repeats": args.iqp_repeats}
            )
            states = statevectors(reduced, qubits, encoding, **options)
            quantum_kernels = {
                "fidelity": fidelity_kernel(states),
                "projected": projected_quantum_kernel(
                    states, qubits, gamma=args.pqk_gamma
                ),
            }
            for quantum_name, quantum in quantum_kernels.items():
                diagnostics[f"{qubits}q_{encoding}_{quantum_name}"] = (
                    concentration_diagnostic(quantum)
                )
                root = quantum_sqrt(quantum)
                for classical_name, reference in classical.items():
                    for entry in geometric_difference_sweep(
                        reference,
                        quantum,
                        regularisations=DEFAULT_REGULARISATIONS,
                        advantage_threshold=ADVANTAGE_THRESHOLD,
                        precomputed_sqrt=root,
                    ):
                        rows.append(
                            {
                                "qubits": int(qubits),
                                "encoding": encoding,
                                "quantum_kernel": quantum_name,
                                "classical_kernel": classical_name,
                                **entry,
                            }
                        )
                print(
                    f"  qubits={qubits} encoding={encoding:5s} "
                    f"quantum={quantum_name:9s} done",
                    flush=True,
                )

    summary: list[dict[str, object]] = []
    for qubits in args.qubits:
        for encoding in ENCODINGS:
            for quantum_name in ("fidelity", "projected"):
                for regularisation in DEFAULT_REGULARISATIONS:
                    matching = [
                        row
                        for row in rows
                        if row["qubits"] == qubits
                        and row["encoding"] == encoding
                        and row["quantum_kernel"] == quantum_name
                        and row["regularisation"] == regularisation
                    ]
                    if not matching:
                        continue
                    best = min(matching, key=lambda row: row["ratio"])
                    summary.append(
                        {
                            "qubits": int(qubits),
                            "encoding": encoding,
                            "quantum_kernel": quantum_name,
                            "regularisation": float(regularisation),
                            "closest_classical_kernel": best["classical_kernel"],
                            "min_ratio": float(best["ratio"]),
                            "concentrated": bool(
                                diagnostics[
                                    f"{qubits}q_{encoding}_{quantum_name}"
                                ]["concentrated"]
                            ),
                            "geometric_difference": float(
                                best["geometric_difference"]
                            ),
                            "advantage_possible": bool(best["advantage_possible"]),
                        }
                    )

    ratios = [float(row["min_ratio"]) for row in summary]
    return {
        "samples": int(values.shape[0]),
        "sqrt_samples": float(np.sqrt(values.shape[0])),
        "advantage_threshold": ADVANTAGE_THRESHOLD,
        "largest_min_ratio": max(ratios) if ratios else None,
        "any_advantage_possible": bool(any(row["advantage_possible"] for row in summary)),
        "interpretation": interpret(ratios, ADVANTAGE_THRESHOLD),
        "concentration_diagnostics": diagnostics,
        "any_kernel_concentrated": bool(
            any(entry["concentrated"] for entry in diagnostics.values())
        ),
        "summary": summary,
        "full_sweep": rows,
    }


def write_figure(results: dict[str, object], path: Path) -> str:
    """Plot min ratio against qubit count, with the advantage threshold marked."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tightest = min(DEFAULT_REGULARISATIONS)
    rows = [
        row
        for row in results["summary"]
        if row["regularisation"] == tightest
    ]
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for encoding in ENCODINGS:
        for quantum_name in ("fidelity", "projected"):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["encoding"] == encoding
                    and row["quantum_kernel"] == quantum_name
                ),
                key=lambda row: row["qubits"],
            )
            if not selected:
                continue
            axis.plot(
                [row["qubits"] for row in selected],
                [row["min_ratio"] for row in selected],
                marker="o",
                label=f"{encoding} / {quantum_name}",
            )
    axis.axhline(
        ADVANTAGE_THRESHOLD,
        linestyle="--",
        color="black",
        linewidth=1.0,
        label=f"advantage threshold ({ADVANTAGE_THRESHOLD:g})",
    )
    axis.set_xlabel("qubits")
    axis.set_ylabel(r"$g(K_C \| K_Q) / \sqrt{N}$")
    axis.set_title(
        f"Geometric difference vs best classical kernel (N = {results['samples']})"
    )
    axis.set_ylim(bottom=0.0)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return sha256_file(path)


def main() -> None:
    args = parse_args()
    output = assert_no_locked_test(args.output_dir)
    seed_record = seed_everything(args.seed)

    values, input_hashes = load_embeddings(args)
    values = subsample(values, args.samples, args.seed)
    print(
        f"Embeddings: {values.shape[0]} cases x {values.shape[1]} dimensions"
        f"{' (SYNTHETIC SMOKE)' if args.smoke else ''}",
        flush=True,
    )

    results = evaluate(values, args)

    artifact_hashes = dict(input_hashes)
    if args.figure:
        artifact_hashes["figure"] = write_figure(
            results, output / "geometric_difference.png"
        )

    path, digest = write_results(
        output / "results.json",
        study=STUDY,
        part="part2_geometric_difference",
        config={
            "version": VERSION,
            "samples": int(args.samples),
            "qubits": [int(value) for value in args.qubits],
            "encodings": list(ENCODINGS),
            "regularisations": list(DEFAULT_REGULARISATIONS),
            "pqk_gamma": float(args.pqk_gamma),
            "angle_layers": int(args.angle_layers),
            "iqp_repeats": int(args.iqp_repeats),
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

    print("\n--- GEOMETRIC DIFFERENCE SUMMARY (tightest regularisation) ---")
    tightest = min(DEFAULT_REGULARISATIONS)
    print(f"{'qubits':>6} {'encoding':>8} {'kernel':>10} {'closest classical':>22} {'g/sqrt(N)':>10} {'conc?':>6}")
    for row in results["summary"]:
        if row["regularisation"] != tightest:
            continue
        print(
            f"{row['qubits']:>6} {row['encoding']:>8} {row['quantum_kernel']:>10} "
            f"{row['closest_classical_kernel']:>22} {row['min_ratio']:>10.4f} "
            f"{'YES' if row['concentrated'] else '-':>6}"
        )
    print(f"\nLargest min ratio: {results['largest_min_ratio']:.4f}")
    print(f"Any advantage possible: {results['any_advantage_possible']}")
    print("")
    print("--- KERNEL CONCENTRATION (off-diagonal mean, eff. rank fraction) ---")
    for name, entry in results["concentration_diagnostics"].items():
        flag = "  <-- CONCENTRATED, ratio not meaningful" if entry["concentrated"] else ""
        print(
            f"  {name:>22}  off-diag {entry['off_diagonal_mean']:.4f}  "
            f"eff.rank {entry['effective_rank_fraction']:.3f}{flag}"
        )
    print(f"\n{results['interpretation']}\n")
    print(f"Results: {path}")
    print(f"Results SHA-256: {digest}")
    print("Labels used: False | Models trained: False | Locked test accessed: False")


if __name__ == "__main__":
    main()
