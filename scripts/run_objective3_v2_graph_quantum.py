#!/usr/bin/env python3
"""Part 6: does the graph topology carry information the circuit can use?

Trains five parameter-matched variants over coarsened patch graphs and tests H3
of the v2.0 protocol. The decisive comparison is graph_quantum against
complete_quantum: identical circuit, identical parameters, identical gates, and
the only difference is whether the two-qubit couplings follow the real adjacency
or a complete graph. If the topology carries nothing, the two are equal.

Graphs are coarsened once and cached, so the five variants and every seed train
on byte-identical inputs and the comparison cannot drift.

Validation labels are used for model selection. The locked test cohort is never
opened.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective3_v2 import STUDY, VERSION
from cxr_thesis.objective3_v2.graph_quantum import GraphQuantumHead, coarsen
from cxr_thesis.objective3_v2.guards import assert_no_locked_test, require_existing
from cxr_thesis.objective3_v2.io_utils import sha256_file, verify_sha256, write_results
from cxr_thesis.objective3_v2.seeds import protocol_seeds, seed_everything
from cxr_thesis.objective3_v2.stats import bootstrap_ci, paired_wilcoxon

PART = "part6_graph_quantum"
PRIMARY_LABELS = [
    "Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass", "Consolidation",
    "Pneumothorax", "Pleural_Thickening", "Cardiomegaly", "Emphysema", "Edema",
    "Fibrosis",
]
VARIANTS = GraphQuantumHead.VARIANTS
H3_THRESHOLD = 0.005
DEFAULT_SEEDS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--expected-val-sha256")
    parser.add_argument("--cases", type=int, default=8000)
    parser.add_argument("--validation-cases", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--supernodes", type=int, default=4, choices=(2, 4))
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------
# coarsening cache
# --------------------------------------------------------------------------


def build_cache(
    manifest: Path, graph_root: Path, cases: int, supernodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Coarsen every graph once and return pooled features, adjacency, labels."""

    from cxr_thesis.objective1.graphs import GraphSample
    from cxr_thesis.objective2.graph_generation import safe_graph_name

    frame = pd.read_csv(manifest).head(cases)
    pooled_all, adjacency_all, labels_all = [], [], []
    node_counts, edge_counts, densities = [], [], []

    for _, record in frame.iterrows():
        path = assert_no_locked_test(
            Path(graph_root) / f"{safe_graph_name(record['image_id'])}.npz"
        )
        sample = GraphSample.load(path)
        pooled, adjacency = coarsen(
            sample.x, sample.edge_index, sample.node_position, supernodes
        )
        pooled_all.append(pooled)
        adjacency_all.append(adjacency)
        labels_all.append([float(record[label]) for label in PRIMARY_LABELS])
        node_counts.append(int(sample.x.shape[0]))
        edge_counts.append(int(sample.edge_index.shape[1]))
        densities.append(float((adjacency > 0).sum()) / (supernodes * (supernodes - 1)))

    audit = {
        "graphs": len(pooled_all),
        "supernodes": supernodes,
        "node_count_min": int(np.min(node_counts)),
        "node_count_max": int(np.max(node_counts)),
        "edge_count_min": int(np.min(edge_counts)),
        "edge_count_max": int(np.max(edge_counts)),
        "mean_supernode_density": float(np.mean(densities)),
        "graphs_with_no_inter_quadrant_edges": int(sum(1 for d in densities if d == 0)),
    }
    return (
        np.stack(pooled_all).astype(np.float32),
        np.stack(adjacency_all).astype(np.float32),
        np.array(labels_all, dtype=np.float32),
        audit,
    )


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def macro_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    from cxr_thesis.objective2.metrics import multilabel_metrics

    return float(
        multilabel_metrics(scores, targets, thresholds=0.5)["macro"]["auroc"]
    )


def train_one(
    variant: str,
    seed: int,
    data: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train one variant at one seed on the cached coarsened graphs."""

    seed_everything(seed)
    model = GraphQuantumHead(
        len(PRIMARY_LABELS),
        data["train_x"].shape[2],
        variant=variant,
        supernodes=args.supernodes,
        layers=args.layers,
    )
    positives = data["train_y"].sum(dim=0)
    weights = (data["train_y"].shape[0] - positives) / positives.clamp(min=1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights.clamp(max=50.0))
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best, best_epoch, stale = -1.0, 0, 0
    cases = data["train_x"].shape[0]
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(cases)
        for start in range(0, cases, args.batch_size):
            index = order[start : start + args.batch_size]
            optimiser.zero_grad(set_to_none=True)
            logits = model(data["train_x"][index], data["train_a"][index])
            loss = criterion(logits, data["train_y"][index])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(
                model(data["val_x"], data["val_a"])
            ).cpu().numpy()
        auroc = macro_auroc(data["val_y"].cpu().numpy(), scores)
        if auroc > best:
            best, best_epoch, stale = auroc, epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    budget = model.budget()
    return {
        "variant": variant,
        "seed": int(seed),
        "best_epoch": best_epoch,
        "validation_macro_auroc": float(best),
        "wall_clock_seconds": time.perf_counter() - started,
        **budget,
        "test_evaluated": False,
    }


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def compare(runs: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """Per-variant summary plus the paired comparisons that decide H3."""

    indexed = {
        (row["variant"], row["seed"]): row["validation_macro_auroc"] for row in runs
    }
    present = [v for v in VARIANTS if any(r["variant"] == v for r in runs)]

    table = []
    for variant in present:
        values = np.array(
            [indexed[(variant, s)] for s in seeds if (variant, s) in indexed]
        )
        interval = bootstrap_ci(np.mean, values, resamples=10_000, seed=7)
        table.append(
            {
                "variant": variant,
                "seeds": int(values.size),
                "mean_macro_auroc": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "ci95_low": interval.confidence_interval_95[0],
                "ci95_high": interval.confidence_interval_95[1],
            }
        )

    comparisons = []
    for reference in ("complete_quantum", "no_entangle", "classical_gnn", "random_fixed"):
        paired = [
            s
            for s in seeds
            if ("graph_quantum", s) in indexed and (reference, s) in indexed
        ]
        if len(paired) < 3:
            continue
        left = np.array([indexed[("graph_quantum", s)] for s in paired])
        right = np.array([indexed[(reference, s)] for s in paired])
        test = paired_wilcoxon(left, right)
        interval = bootstrap_ci(np.mean, left - right, resamples=10_000, seed=11)
        comparisons.append(
            {
                "comparison": f"graph_quantum_minus_{reference}",
                "seeds": len(paired),
                "mean_delta": test.mean_difference,
                "ci95_low": interval.confidence_interval_95[0],
                "ci95_high": interval.confidence_interval_95[1],
                "wilcoxon_p": test.p_value,
                "wins": int(((left - right) > 0).sum()),
                "wins_above_threshold": int(((left - right) >= H3_THRESHOLD).sum()),
                "bootstrap_p_report": interval.p_value_report,
            }
        )

    decisive = next(
        (c for c in comparisons if c["comparison"].endswith("complete_quantum")), None
    )
    verdict = {
        "hypothesis": (
            "adjacency-conditioned entangling beats complete-graph entangling by "
            f">= {H3_THRESHOLD} macro AUROC, paired Wilcoxon over seeds"
        ),
        "decisive_comparison": "graph_quantum_minus_complete_quantum",
        "passed": bool(
            decisive is not None
            and decisive["mean_delta"] >= H3_THRESHOLD
            and decisive["wilcoxon_p"] < 0.05
        ),
        "evidence": decisive,
        "note": (
            "graph_quantum and complete_quantum are the same circuit with the "
            "same parameters; only the coupling weights differ. A null here "
            "means the coarsened topology carries no usable information, not "
            "that entanglement is useless: compare no_entangle for that."
        ),
    }
    return {"table": table, "comparisons": comparisons, "h3_verdict": verdict}


def write_figure(analysis: dict[str, Any], output: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    rows = analysis["table"]
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    names = [r["variant"] for r in rows]
    means = [r["mean_macro_auroc"] for r in rows]
    lower = [r["mean_macro_auroc"] - r["ci95_low"] for r in rows]
    upper = [r["ci95_high"] - r["mean_macro_auroc"] for r in rows]
    axis.bar(names, means, yerr=[lower, upper], capsize=4, color="tab:blue")
    axis.set_ylabel("validation macro AUROC")
    axis.set_title("Graph-structured circuit and its controls (bootstrap 95% CI)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    path = output / "graph_quantum_variants.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return {"graph_quantum_variants": sha256_file(path)}


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------


def smoke(args: argparse.Namespace) -> None:
    """Synthetic graphs where the topology genuinely carries label information."""

    output = assert_no_locked_test(args.output_dir)
    generator = np.random.default_rng(0)
    cases, features = 240, 7
    pooled = generator.normal(size=(cases, args.supernodes, features)).astype(np.float32)
    adjacency = np.zeros((cases, args.supernodes, args.supernodes), dtype=np.float32)
    signal = generator.integers(0, 2, size=cases)
    for index in range(cases):
        base = 0.9 if signal[index] else 0.1
        upper = generator.uniform(0.0, base, size=(args.supernodes, args.supernodes))
        symmetric = np.triu(upper, 1)
        adjacency[index] = (symmetric + symmetric.T).astype(np.float32)
    labels = np.zeros((cases, len(PRIMARY_LABELS)), dtype=np.float32)
    labels[:, 0] = signal

    split = 160
    data = {
        "train_x": torch.tensor(pooled[:split]),
        "train_a": torch.tensor(adjacency[:split]),
        "train_y": torch.tensor(labels[:split]),
        "val_x": torch.tensor(pooled[split:]),
        "val_a": torch.tensor(adjacency[split:]),
        "val_y": torch.tensor(labels[split:]),
    }
    args.epochs, args.patience = 4, 2
    seeds = protocol_seeds(3)
    runs = [
        train_one(variant, seed, data, args)
        for variant in ("graph_quantum", "complete_quantum")
        for seed in seeds
    ]
    analysis = compare(runs, seeds)
    for row in analysis["table"]:
        print(f"  {row['variant']:>17} auroc {row['mean_macro_auroc']:.4f}")
    print(f"H3 verdict on synthetic data: {analysis['h3_verdict']['passed']}")
    print("Test evaluated: False | Locked test accessed: False")
    print("GRAPH QUANTUM SMOKE PASSED")


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke(args)
        return

    output = assert_no_locked_test(args.output_dir)
    require_existing([args.train_manifest, args.val_manifest, args.graph_root])
    if args.expected_train_sha256:
        verify_sha256(args.train_manifest, args.expected_train_sha256)
    if args.expected_val_sha256:
        verify_sha256(args.val_manifest, args.expected_val_sha256)
    seed_record = seed_everything(42)

    print("Coarsening graphs...", flush=True)
    train_x, train_a, train_y, train_audit = build_cache(
        args.train_manifest, args.graph_root, args.cases, args.supernodes
    )
    val_x, val_a, val_y, val_audit = build_cache(
        args.val_manifest, args.graph_root, args.validation_cases, args.supernodes
    )
    print(
        f"  train {train_audit['graphs']} graphs, "
        f"validation {val_audit['graphs']} graphs, "
        f"mean supernode density {train_audit['mean_supernode_density']:.3f}",
        flush=True,
    )

    data = {
        "train_x": torch.tensor(train_x), "train_a": torch.tensor(train_a),
        "train_y": torch.tensor(train_y), "val_x": torch.tensor(val_x),
        "val_a": torch.tensor(val_a), "val_y": torch.tensor(val_y),
    }

    seeds = protocol_seeds(args.seeds)
    runs = []
    total = len(args.variants) * len(seeds)
    for variant in args.variants:
        for seed in seeds:
            runs.append(train_one(variant, seed, data, args))
            print(
                f"[{len(runs)}/{total}] {variant} seed={seed} "
                f"auroc={runs[-1]['validation_macro_auroc']:.4f} "
                f"({runs[-1]['wall_clock_seconds']:.0f}s)",
                flush=True,
            )

    analysis = compare(runs, seeds)
    artifact_hashes = write_figure(analysis, output) if args.figures else {}
    path, digest = write_results(
        output / "results.json",
        study=STUDY,
        part=PART,
        config={
            "version": VERSION,
            "supernodes": args.supernodes,
            "layers": args.layers,
            "variants": list(args.variants),
            "seeds": seeds,
            "train_cases": args.cases,
            "validation_cases": args.validation_cases,
            "coarsening": "spatial quadrant, median split per axis",
            "entangler": "CRZ(theta_ij * a_ij), identity where no edge exists",
            "h3_threshold": H3_THRESHOLD,
            "seeding": seed_record,
        },
        results={
            "train_graph_audit": train_audit,
            "validation_graph_audit": val_audit,
            "table": analysis["table"],
            "comparisons": analysis["comparisons"],
            "h3_verdict": analysis["h3_verdict"],
            "runs": runs,
            "test_evaluated": False,
        },
        artifact_hashes=artifact_hashes,
        seed=seeds[0],
        locked_test_accessed=False,
    )

    print("")
    print(f"{'variant':>18} {'mean AUROC':>11} {'95% CI':>22}")
    for row in analysis["table"]:
        interval = f"[{row['ci95_low']:.4f},{row['ci95_high']:.4f}]"
        print(f"{row['variant']:>18} {row['mean_macro_auroc']:>11.4f} {interval:>22}")
    print("")
    for row in analysis["comparisons"]:
        interval = f"[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}]"
        print(
            f"  {row['comparison']:>42} {row['mean_delta']:>+8.4f} {interval:>20} "
            f"p={row['wilcoxon_p']:.4f}"
        )
    print("")
    print(f"H3 PASSED: {analysis['h3_verdict']['passed']}")
    print(f"  {analysis['h3_verdict']['note']}")
    print("")
    print(f"Results: {path}")
    print(f"Results SHA-256: {digest}")
    print("Test evaluated: False | Locked test accessed: False")


if __name__ == "__main__":
    main()
