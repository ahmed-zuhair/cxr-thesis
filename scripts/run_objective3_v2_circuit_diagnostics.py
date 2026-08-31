#!/usr/bin/env python3
"""Part 3: were the v1.0 and v1.1 circuits ever able to learn?

Measures expressibility, entangling capability, and gradient variance for the
two ansätze actually used, plus the two feature maps compared in Part 2. No data,
no labels, no training; every number comes from the circuit structure alone.

The point is attribution. A null result with a diagnosed mechanism is a finding;
a null result without one is just a failed experiment.
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
from cxr_thesis.objective3_v2.circuits import (
    ANSATZ_NAMES,
    barren_plateau_slope,
    encoding_expressibility,
    entangling_capability,
    expressibility,
    gradient_variance,
)
from cxr_thesis.objective3_v2.guards import assert_no_locked_test
from cxr_thesis.objective3_v2.io_utils import sha256_file, write_results
from cxr_thesis.objective3_v2.kernels import ENCODINGS
from cxr_thesis.objective3_v2.seeds import seed_everything

PLATEAU_QUBITS = (2, 4, 6, 8, 10)
PLATEAU_DEPTHS = (1, 2, 4, 8, 16)
DEPLOYED_QUBITS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expressibility-samples", type=int, default=5000)
    parser.add_argument("--entangling-samples", type=int, default=5000)
    parser.add_argument("--gradient-samples", type=int, default=200)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Small sample counts and a reduced sweep, for a fast check",
    )
    return parser.parse_args()


def interpret(results: dict[str, object]) -> str:
    """State what the diagnostics do and do not explain."""

    lines: list[str] = []
    deployed = {row["ansatz"]: row for row in results["expressibility"]}
    v11 = deployed.get("v1_1_reupload")
    v10 = deployed.get("v1_0_bottleneck")
    if v10 and v11:
        lines.append(
            f"At the deployed scale of {DEPLOYED_QUBITS} qubits the v1.1 "
            f"re-uploading ansatz scored KL {v11['kl_divergence_from_haar']:.4f} "
            f"against Haar, versus {v10['kl_divergence_from_haar']:.4f} for v1.0 "
            "(lower is more expressible)."
        )
    entangling = {row["ansatz"]: row for row in results["entangling_capability"]}
    if "v1_1_reupload" in entangling:
        measure = entangling["v1_1_reupload"]["meyer_wallach_mean"]
        lines.append(
            f"Its mean Meyer-Wallach entangling capability was {measure:.4f}"
            + (
                ", so the circuit barely leaves product states and the qubits "
                "carry almost no joint information."
                if measure < 0.2
                else ", so the circuit does generate genuine entanglement."
            )
        )
    fit = results["barren_plateau_fit"].get("v1_1_reupload", {})
    if fit.get("barren") is True:
        lines.append(
            "Gradient variance decays exponentially with qubit count "
            f"(factor {fit['decay_base_per_qubit']:.3f} per qubit), a barren "
            "plateau: at larger widths this model could not be trained no matter "
            "how much data it were given. At the four qubits actually used the "
            "gradients remain finite, so the plateau does not by itself explain "
            "the v1.1 null; it bounds how far the design could ever scale."
        )
    elif fit.get("barren") is False:
        lines.append(
            "Gradient variance does not decay exponentially with qubit count over "
            "the range tested, so a barren plateau does not explain the v1.1 "
            "null and trainability was not the binding constraint."
        )
    encodings = {row["encoding"]: row for row in results["encoding_expressibility"]}
    if "angle" in encodings and "iqp" in encodings:
        angle, iqp = encodings["angle"], encodings["iqp"]
        lines.append(
            f"The two feature maps differ sharply: angle encoding, the one v1.0 "
            f"and v1.1 used, scored KL {angle['kl_divergence_from_haar']:.4f} "
            f"with entangling capability {angle['meyer_wallach_mean']:.4f}, while "
            f"IQP scored KL {iqp['kl_divergence_from_haar']:.4f} with "
            f"{iqp['meyer_wallach_mean']:.4f}. This is the mechanism behind the "
            "Part 2 geometric-difference gap between the two encodings."
        )
    return " ".join(lines)


def write_figures(results: dict[str, object], output: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    # 1. expressibility and entangling capability side by side
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    names = [row["ansatz"] for row in results["expressibility"]]
    axes[0].bar(names, [row["kl_divergence_from_haar"] for row in results["expressibility"]])
    axes[0].set_ylabel("KL from Haar (lower = more expressible)")
    axes[0].set_title(f"Expressibility at {DEPLOYED_QUBITS} qubits")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(
        [row["ansatz"] for row in results["entangling_capability"]],
        [row["meyer_wallach_mean"] for row in results["entangling_capability"]],
        color="tab:orange",
    )
    axes[1].set_ylabel("mean Meyer-Wallach Q")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Entangling capability")
    axes[1].tick_params(axis="x", rotation=15)
    figure.tight_layout()
    path = output / "ansatz_diagnostics.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    hashes["ansatz_diagnostics"] = sha256_file(path)

    # 2. barren plateau
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for depth in sorted({row["layers"] for row in results["gradient_variance"]}):
        selected = sorted(
            (row for row in results["gradient_variance"] if row["layers"] == depth),
            key=lambda row: row["qubits"],
        )
        positive = [row for row in selected if row["gradient_variance"] > 0]
        if len(positive) < 2:
            continue
        axis.semilogy(
            [row["qubits"] for row in positive],
            [row["gradient_variance"] for row in positive],
            marker="o",
            label=f"{depth} layers",
        )
    axis.set_xlabel("qubits")
    axis.set_ylabel(r"Var$[\partial \langle Z_0\rangle / \partial \theta]$")
    axis.set_title("Barren plateau check (v1.0 entangling stack)")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "barren_plateau.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    hashes["barren_plateau"] = sha256_file(path)

    # 3. encodings
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    rows = results["encoding_expressibility"]
    axes[0].bar([r["encoding"] for r in rows], [r["kl_divergence_from_haar"] for r in rows])
    axes[0].set_ylabel("KL from Haar")
    axes[0].set_title("Feature-map expressibility")
    axes[1].bar(
        [r["encoding"] for r in rows],
        [r["meyer_wallach_mean"] for r in rows],
        color="tab:green",
    )
    axes[1].set_ylabel("mean Meyer-Wallach Q")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Feature-map entangling capability")
    figure.tight_layout()
    path = output / "encoding_diagnostics.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    hashes["encoding_diagnostics"] = sha256_file(path)
    return hashes


def main() -> None:
    args = parse_args()
    output = assert_no_locked_test(args.output_dir)
    seed_record = seed_everything(args.seed)

    qubit_sweep = (2, 4, 6) if args.smoke else PLATEAU_QUBITS
    depth_sweep = (1, 4) if args.smoke else PLATEAU_DEPTHS
    express_samples = 400 if args.smoke else args.expressibility_samples
    entangle_samples = 400 if args.smoke else args.entangling_samples
    gradient_samples = 60 if args.smoke else args.gradient_samples

    print("--- expressibility and entangling capability ---", flush=True)
    express_rows = []
    entangle_rows = []
    for name in ANSATZ_NAMES:
        express_rows.append(
            expressibility(
                name, DEPLOYED_QUBITS, samples=express_samples, seed=args.seed
            )
        )
        entangle_rows.append(
            entangling_capability(
                name, DEPLOYED_QUBITS, samples=entangle_samples, seed=args.seed
            )
        )
        print(
            f"  {name:>18}  KL {express_rows[-1]['kl_divergence_from_haar']:.4f}  "
            f"Q {entangle_rows[-1]['meyer_wallach_mean']:.4f}",
            flush=True,
        )

    print("--- gradient variance sweep ---", flush=True)
    gradient_rows = []
    for name in ANSATZ_NAMES:
        for depth in depth_sweep:
            for qubits in qubit_sweep:
                gradient_rows.append(
                    gradient_variance(
                        name,
                        qubits=qubits,
                        layers=depth,
                        samples=gradient_samples,
                        seed=args.seed,
                    )
                )
        print(f"  {name} done", flush=True)

    fits: dict[str, object] = {}
    for name in ANSATZ_NAMES:
        deepest = max(depth_sweep)
        selected = sorted(
            (
                row
                for row in gradient_rows
                if row["ansatz"] == name and row["layers"] == deepest
            ),
            key=lambda row: row["qubits"],
        )
        fits[name] = barren_plateau_slope(
            [row["qubits"] for row in selected],
            [row["gradient_variance"] for row in selected],
        )
        fits[name]["layers"] = deepest

    print("--- feature-map diagnostics ---", flush=True)
    encoding_rows = [
        encoding_expressibility(
            encoding, DEPLOYED_QUBITS, samples=express_samples, seed=args.seed
        )
        for encoding in ENCODINGS
    ]
    for row in encoding_rows:
        print(
            f"  {row['encoding']:>18}  KL {row['kl_divergence_from_haar']:.4f}  "
            f"Q {row['meyer_wallach_mean']:.4f}",
            flush=True,
        )

    results: dict[str, object] = {
        "deployed_qubits": DEPLOYED_QUBITS,
        "expressibility": express_rows,
        "entangling_capability": entangle_rows,
        "gradient_variance": gradient_rows,
        "barren_plateau_fit": fits,
        "encoding_expressibility": encoding_rows,
    }
    results["interpretation"] = interpret(results)

    artifact_hashes = write_figures(results, output) if args.figures else {}
    path, digest = write_results(
        output / "results.json",
        study=STUDY,
        part="part3_circuit_diagnostics",
        config={
            "version": VERSION,
            "qubit_sweep": [int(v) for v in qubit_sweep],
            "depth_sweep": [int(v) for v in depth_sweep],
            "expressibility_samples": express_samples,
            "entangling_samples": entangle_samples,
            "gradient_samples": gradient_samples,
            "smoke": bool(args.smoke),
            "seeding": seed_record,
            "labels_used": False,
            "models_trained": False,
            "data_used": False,
        },
        results=results,
        artifact_hashes=artifact_hashes,
        seed=args.seed,
        locked_test_accessed=False,
    )

    print("")
    print("--- BARREN PLATEAU FIT (deepest sweep) ---")
    for name, fit in fits.items():
        if fit.get("slope_log_variance_per_qubit") is None:
            print(f"  {name:>18}  insufficient data")
            continue
        print(
            f"  {name:>18}  slope {fit['slope_log_variance_per_qubit']:+.4f}/qubit  "
            f"decay x{fit['decay_base_per_qubit']:.3f}  barren={fit['barren']}"
        )
    print("")
    print(results["interpretation"])
    print("")
    print(f"Results: {path}")
    print(f"Results SHA-256: {digest}")
    print("Data used: False | Labels used: False | Locked test accessed: False")


if __name__ == "__main__":
    main()
