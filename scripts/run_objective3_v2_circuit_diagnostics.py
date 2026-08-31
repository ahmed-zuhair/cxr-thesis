#!/usr/bin/env python3
"""Part 3: circuit expressibility, entanglement, and barren-plateau diagnostics.

This job uses synthetic circuit parameters only. It never opens a cohort,
manifest, label file, report, image, or prediction. Work is committed to hashed
aggregate shards so an interrupted Kaggle session can resume safely.

Smoke check (tiny synthetic run, normally well under 60 seconds):

    python scripts/run_objective3_v2_circuit_diagnostics.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective3_v2 import STUDY, VERSION
from cxr_thesis.objective3_v2.circuits import (
    ANSATZE,
    GRAPH_STRUCTURED_ANSATZ_NAME,
    barren_plateau_slope,
    entangling_capability,
    expressibility,
    gradient_variance,
    registered_ansatz_names,
)
from cxr_thesis.objective3_v2.guards import assert_no_locked_test
from cxr_thesis.objective3_v2.io_utils import (
    ShardLedger,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
    write_results,
)
from cxr_thesis.objective3_v2.seeds import seed_everything

PART = "part3_circuit_diagnostics"
PLATEAU_QUBITS = (2, 4, 6, 8, 10)
PLATEAU_DEPTHS = (1, 2, 4, 8, 16)
DEPLOYED_QUBITS = 4
FIDELITY_BINS = 75
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "results" / "objective3_v2" / "circuit_diagnostics" / VERSION
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Default: {DEFAULT_OUTPUT} (or its smoke/ child with --smoke)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expressibility-samples", type=int, default=5000)
    parser.add_argument("--entangling-samples", type=int, default=5000)
    parser.add_argument("--gradient-samples", type=int, default=200)
    parser.add_argument("--fidelity-bins", type=int, default=FIDELITY_BINS)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip plots for a diagnostic-only development run",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny synthetic sweep that completes in under 60 seconds",
    )
    return parser.parse_args()


def _report_number(value: float) -> float:
    """Keep report values to four decimal places without erasing tiny variances."""

    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Non-finite numbers are not permitted in public results")
    if number != 0.0 and abs(number) < 0.0001:
        return float(f"{number:.4e}")
    return round(number, 4)


def _round_report_tree(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return _report_number(float(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _round_report_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_report_tree(item) for item in value]
    return value


def _fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(canonical)


def _load_or_compute_shard(
    *,
    ledger: ShardLedger,
    shard_dir: Path,
    shard_name: str,
    run_fingerprint: str,
    compute: Callable[[], dict[str, Any]],
    counters: dict[str, int],
) -> tuple[dict[str, Any], Path, str]:
    """Load an intact aggregate shard, or atomically compute and hash it."""

    shard_path = shard_dir / f"{run_fingerprint[:12]}__{shard_name}.json"
    if ledger.is_complete(shard_name, shard_path):
        payload = read_json(shard_path)
        if payload.get("run_fingerprint") == run_fingerprint:
            counters["resumed"] += 1
            return payload["result"], shard_path, sha256_file(shard_path)

    result = _round_report_tree(compute())
    write_json_atomic(
        shard_path,
        {
            "study": STUDY,
            "part": PART,
            "shard": shard_name,
            "run_fingerprint": run_fingerprint,
            "aggregate_only": True,
            "result": result,
        },
    )
    record = ledger.mark_complete(
        shard_name,
        shard_path,
        run_fingerprint=run_fingerprint,
        aggregate_only=True,
    )
    counters["computed"] += 1
    return result, shard_path, str(record["sha256"])


def _fit_plateaus(
    names: tuple[str, ...],
    rows: list[dict[str, Any]],
    depths: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    fits: dict[str, dict[str, Any]] = {}
    for name in names:
        fits[name] = {}
        for depth in depths:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["ansatz"] == name and row["layers"] == depth
                ),
                key=lambda row: row["qubits"],
            )
            fit = barren_plateau_slope(
                [int(row["qubits"]) for row in selected],
                [float(row["gradient_variance"]) for row in selected],
            )
            fit["layers"] = int(depth)
            fits[name][str(depth)] = _round_report_tree(fit)
    return fits


def interpret(results: dict[str, Any]) -> str:
    """Give a cautious plain-English interpretation without inventing cutoffs."""

    lines: list[str] = []
    express_rows = results["expressibility"]
    if express_rows:
        ordered = sorted(express_rows, key=lambda row: row["kl_divergence_from_haar"])
        best = ordered[0]
        lines.append(
            f"At {DEPLOYED_QUBITS} qubits, {best['ansatz']} was closest to Haar "
            f"with KL {best['kl_divergence_from_haar']:.4f}; lower KL means broader "
            "state-space coverage. There is no universal KL pass threshold, so "
            "this is a relative expressibility diagnosis rather than a claim of "
            "sufficient expressibility."
        )

    entangling_rows = results["entangling_capability"]
    if entangling_rows:
        strongest = max(entangling_rows, key=lambda row: row["meyer_wallach_mean"])
        lines.append(
            f"The largest mean Meyer-Wallach Q was {strongest['meyer_wallach_mean']:.4f} "
            f"for {strongest['ansatz']}; Q near zero indicates product-like states "
            "and larger Q indicates genuine multi-qubit entanglement."
        )

    for row in results["deployed_gradient"]:
        variance = float(row["gradient_variance"])
        if variance <= 1e-12:
            judgement = "numerically vanished, which is incompatible with useful training"
        else:
            judgement = (
                "was non-zero at the deployed width; trainability must still be judged "
                "with the observed optimisation trace because no universal variance cutoff exists"
            )
        lines.append(
            f"For {row['ansatz']} at its deployed {row['layers']}-layer, "
            f"{DEPLOYED_QUBITS}-qubit setting, gradient variance was {variance:.4e} and {judgement}."
        )

    if results["graph_structured_ansatz"]["status"] != "complete":
        lines.append(
            "The graph-structured ansatz is explicitly marked pending; Part 6 can "
            "register its statevector builder through register_graph_structured_ansatz, "
            "after which this same runner will apply all three diagnostics."
        )
    return " ".join(lines)


def write_figures(results: dict[str, Any], output: Path) -> dict[str, str]:
    """Write the three preregistered aggregate figures and return their hashes."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    rows = results["expressibility"]
    figure, axes = plt.subplots(
        len(rows), 1, figsize=(8.0, max(3.5, 3.0 * len(rows))), squeeze=False
    )
    for axis, row in zip(axes[:, 0], rows):
        histogram = row["fidelity_histogram"]
        edges = np.asarray(histogram["bin_edges"], dtype=float)
        empirical = np.asarray(histogram["empirical_probability"], dtype=float)
        haar = np.asarray(histogram["haar_probability"], dtype=float)
        widths = np.diff(edges)
        centers = edges[:-1] + widths / 2.0
        axis.bar(
            centers,
            empirical,
            width=widths,
            alpha=0.65,
            label="ansatz samples",
            color="tab:blue",
        )
        axis.step(edges[:-1], haar, where="post", color="black", label="Haar")
        axis.set_yscale("log")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylabel("probability/bin")
        axis.set_title(
            f"{row['ansatz']} — KL={row['kl_divergence_from_haar']:.4f}"
        )
        axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel("state fidelity")
    figure.suptitle(f"Expressibility at {DEPLOYED_QUBITS} qubits")
    figure.tight_layout()
    path = output / "expressibility_histogram.png"
    figure.savefig(path, dpi=200, metadata={"Software": "cxr-thesis Objective 3 v2.0"})
    plt.close(figure)
    hashes[path.name] = sha256_file(path)

    rows = results["entangling_capability"]
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    positions = np.arange(len(rows))
    axis.bar(
        positions,
        [row["meyer_wallach_mean"] for row in rows],
        yerr=[row["meyer_wallach_std"] for row in rows],
        capsize=4,
        color="tab:orange",
    )
    axis.set_xticks(positions, [row["ansatz"] for row in rows], rotation=15)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("mean Meyer-Wallach Q")
    axis.set_title(f"Entangling capability at {DEPLOYED_QUBITS} qubits")
    figure.tight_layout()
    path = output / "entangling_capability.png"
    figure.savefig(path, dpi=200, metadata={"Software": "cxr-thesis Objective 3 v2.0"})
    plt.close(figure)
    hashes[path.name] = sha256_file(path)

    names = tuple(row["ansatz"] for row in results["expressibility"])
    figure, axes = plt.subplots(
        1, len(names), figsize=(max(7.0, 6.0 * len(names)), 4.8), squeeze=False
    )
    for axis, name in zip(axes[0], names):
        for depth in results["depth_sweep"]:
            selected = sorted(
                (
                    row
                    for row in results["gradient_variance"]
                    if row["ansatz"] == name and row["layers"] == depth
                ),
                key=lambda row: row["qubits"],
            )
            positive = [row for row in selected if row["gradient_variance"] > 0]
            if len(positive) < 2:
                continue
            axis.semilogy(
                [row["qubits"] for row in positive],
                [row["gradient_variance"] for row in positive],
                marker="o",
                label=f"depth {depth}",
            )
        axis.set_xlabel("qubits")
        axis.set_ylabel("gradient variance")
        axis.set_title(name)
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8)
    figure.suptitle("Barren-plateau diagnostic: log variance vs qubit count")
    figure.tight_layout()
    path = output / "barren_plateau_log_variance.png"
    figure.savefig(path, dpi=200, metadata={"Software": "cxr-thesis Objective 3 v2.0"})
    plt.close(figure)
    hashes[path.name] = sha256_file(path)
    return hashes


def main() -> None:
    args = parse_args()
    if args.expressibility_samples < 1 or args.entangling_samples < 2:
        raise ValueError("Expressibility needs >=1 sample and entangling capability needs >=2")
    if args.gradient_samples < 2 or args.fidelity_bins < 2:
        raise ValueError("Gradient samples and fidelity bins must both be >=2")

    requested_output = args.output_dir
    if requested_output is None:
        requested_output = DEFAULT_OUTPUT / "smoke" if args.smoke else DEFAULT_OUTPUT
    output = assert_no_locked_test(requested_output)
    output.mkdir(parents=True, exist_ok=True)
    seed_record = seed_everything(args.seed)

    qubit_sweep = (2, 4) if args.smoke else PLATEAU_QUBITS
    depth_sweep = (1, 2) if args.smoke else PLATEAU_DEPTHS
    express_samples = min(args.expressibility_samples, 128) if args.smoke else args.expressibility_samples
    entangle_samples = min(args.entangling_samples, 128) if args.smoke else args.entangling_samples
    gradient_samples = min(args.gradient_samples, 32) if args.smoke else args.gradient_samples
    names = registered_ansatz_names()

    run_config: dict[str, Any] = {
        "version": VERSION,
        "seed": int(args.seed),
        "ansatze": list(names),
        "qubit_sweep": list(qubit_sweep),
        "depth_sweep": list(depth_sweep),
        "deployed_qubits": DEPLOYED_QUBITS,
        "expressibility_samples": int(express_samples),
        "entangling_samples": int(entangle_samples),
        "gradient_samples": int(gradient_samples),
        "fidelity_bins": int(args.fidelity_bins),
        "fixed_gradient_parameter": [0, 0, 1],
        "fixed_input": True,
        "smoke": bool(args.smoke),
    }
    run_fingerprint = _fingerprint(run_config)
    run_config["run_fingerprint"] = run_fingerprint

    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    ledger = ShardLedger(shard_dir / "index.json", study=STUDY, part=PART)
    counters = {"computed": 0, "resumed": 0}
    shard_hashes: dict[str, str] = {}

    express_rows: list[dict[str, Any]] = []
    entangle_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    deployed_gradient_rows: list[dict[str, Any]] = []

    print("--- expressibility and entangling capability ---", flush=True)
    for name in names:
        row, path, digest = _load_or_compute_shard(
            ledger=ledger,
            shard_dir=shard_dir,
            shard_name=f"expressibility__{name}",
            run_fingerprint=run_fingerprint,
            compute=lambda name=name: expressibility(
                name,
                DEPLOYED_QUBITS,
                samples=express_samples,
                seed=args.seed,
                bins=args.fidelity_bins,
            ),
            counters=counters,
        )
        express_rows.append(row)
        shard_hashes[f"shards/{path.name}"] = digest

        row, path, digest = _load_or_compute_shard(
            ledger=ledger,
            shard_dir=shard_dir,
            shard_name=f"entangling__{name}",
            run_fingerprint=run_fingerprint,
            compute=lambda name=name: entangling_capability(
                name, DEPLOYED_QUBITS, samples=entangle_samples, seed=args.seed
            ),
            counters=counters,
        )
        entangle_rows.append(row)
        shard_hashes[f"shards/{path.name}"] = digest
        print(
            f"  {name:>18}  KL {express_rows[-1]['kl_divergence_from_haar']:.4f}  "
            f"Q {entangle_rows[-1]['meyer_wallach_mean']:.4f}",
            flush=True,
        )

    print("--- gradient variance sweep ---", flush=True)
    for name in names:
        for depth in depth_sweep:
            for qubits in qubit_sweep:
                shard_name = f"gradient__{name}__q{qubits}__d{depth}"
                row, path, digest = _load_or_compute_shard(
                    ledger=ledger,
                    shard_dir=shard_dir,
                    shard_name=shard_name,
                    run_fingerprint=run_fingerprint,
                    compute=lambda name=name, qubits=qubits, depth=depth: gradient_variance(
                        name,
                        qubits=qubits,
                        layers=depth,
                        samples=gradient_samples,
                        seed=args.seed,
                    ),
                    counters=counters,
                )
                gradient_rows.append(row)
                shard_hashes[f"shards/{path.name}"] = digest
        print(f"  {name} sweep done", flush=True)

    for name in names:
        default_layers = int(ANSATZE[name][1])
        existing = next(
            (
                row
                for row in gradient_rows
                if row["ansatz"] == name
                and row["qubits"] == DEPLOYED_QUBITS
                and row["layers"] == default_layers
            ),
            None,
        )
        if existing is None:
            shard_name = f"gradient_deployed__{name}__q{DEPLOYED_QUBITS}__d{default_layers}"
            existing, path, digest = _load_or_compute_shard(
                ledger=ledger,
                shard_dir=shard_dir,
                shard_name=shard_name,
                run_fingerprint=run_fingerprint,
                compute=lambda name=name, default_layers=default_layers: gradient_variance(
                    name,
                    qubits=DEPLOYED_QUBITS,
                    layers=default_layers,
                    samples=gradient_samples,
                    seed=args.seed,
                ),
                counters=counters,
            )
            shard_hashes[f"shards/{path.name}"] = digest
        deployed_gradient_rows.append(existing)

    fits = _fit_plateaus(names, gradient_rows, depth_sweep)
    graph_status = {
        "ansatz": GRAPH_STRUCTURED_ANSATZ_NAME,
        "status": "complete" if GRAPH_STRUCTURED_ANSATZ_NAME in names else "pending_part6",
        "registration_hook": "register_graph_structured_ansatz",
    }
    results: dict[str, Any] = {
        "aggregate_only": True,
        "data_used": False,
        "labels_used": False,
        "models_trained": False,
        "deployed_qubits": DEPLOYED_QUBITS,
        "depth_sweep": list(depth_sweep),
        "expressibility": express_rows,
        "entangling_capability": entangle_rows,
        "gradient_variance": gradient_rows,
        "deployed_gradient": deployed_gradient_rows,
        "barren_plateau_fit": fits,
        "graph_structured_ansatz": graph_status,
        "shards_computed": counters["computed"],
        "shards_resumed": counters["resumed"],
    }
    results["interpretation"] = interpret(results)
    results = _round_report_tree(results)

    artifact_hashes = dict(sorted(shard_hashes.items()))
    artifact_hashes["shards/index.json"] = sha256_file(ledger.index_path)
    if not args.no_figures:
        artifact_hashes.update(write_figures(results, output))

    path, digest = write_results(
        output / "results.json",
        study=STUDY,
        part=PART,
        seed=args.seed,
        config={
            **run_config,
            "seeding": seed_record,
            "figures_written": not args.no_figures,
        },
        results=results,
        artifact_hashes=dict(sorted(artifact_hashes.items())),
        locked_test_accessed=False,
    )

    print()
    print(results["interpretation"])
    print()
    print(f"Shards computed: {counters['computed']} | resumed: {counters['resumed']}")
    print(f"Results: {path}")
    print(f"Results SHA-256: {digest}")
    print("Data used: False | Labels used: False | Locked-test access: False")


if __name__ == "__main__":
    main()
