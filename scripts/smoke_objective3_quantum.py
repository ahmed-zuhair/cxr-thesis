#!/usr/bin/env python3
"""Run a dependency-light Objective 3 quantum/classical architecture smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective3.models import (
    EnhancedHybridGraphHead,
    HybridGraphHead,
    bottleneck_parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "objective3" / "nih_quantum_gat.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--architecture",
        choices=("v1_concat", "v1_1_reupload_gated"),
        default="v1_concat",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    import pennylane as qml

    if qml.__version__ != "0.45.1":
        raise RuntimeError("Objective 3 requires exactly PennyLane 0.45.1")
    torch.manual_seed(args.seed)
    embeddings = torch.randn(args.batch_size, 160)
    targets = torch.randint(0, 2, (args.batch_size, 12)).float()
    model_class = (
        EnhancedHybridGraphHead
        if args.architecture == "v1_1_reupload_gated"
        else HybridGraphHead
    )
    expected_bottleneck_parameters = (
        36 if args.architecture == "v1_1_reupload_gated" else 24
    )
    models = {
        name: model_class(12, embedding_dim=160, bottleneck=name)
        for name in ("classical_matched", "quantum")
    }
    results: dict[str, object] = {}
    for name, model in models.items():
        logits = model(embeddings)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        results[name] = {
            "output_shape": list(logits.shape),
            "loss": float(loss.detach()),
            "bottleneck_parameters": bottleneck_parameter_count(model.bottleneck),
            "total_trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "gradients_finite": bool(gradients_finite),
        }
    checks = {
        "classical_output_shape": results["classical_matched"]["output_shape"]
        == [args.batch_size, 12],
        "quantum_output_shape": results["quantum"]["output_shape"]
        == [args.batch_size, 12],
        "classical_bottleneck_parameters": results["classical_matched"][
            "bottleneck_parameters"
        ]
        == expected_bottleneck_parameters,
        "quantum_bottleneck_parameters": results["quantum"]["bottleneck_parameters"]
        == expected_bottleneck_parameters,
        "exact_parameter_match": results["classical_matched"][
            "total_trainable_parameters"
        ]
        == results["quantum"]["total_trainable_parameters"],
        "finite_gradients": all(
            bool(result["gradients_finite"]) for result in results.values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Objective 3 smoke checks failed: {checks}")
    print("--- OBJECTIVE 3 QUANTUM SMOKE RESULT ---")
    print(
        json.dumps(
            {
                "pennylane": qml.__version__,
                "torch": torch.__version__,
                "python": sys.version,
                "platform": platform.platform(),
                "config_sha256": sha256_file(args.config),
                "architecture_version": args.architecture,
                "circuit": (
                    "4 qubits x 3 data-reuploading blocks"
                    if args.architecture == "v1_1_reupload_gated"
                    else "4 qubits x 2 strongly-entangling layers"
                ),
                "simulator": "default.qubit analytic",
                "execution_device": "cpu",
                "models": results,
                "checks": checks,
                "medical_data_accessed": False,
                "test_labels_accessed": False,
                "research_result": False,
                "allowed_for_publication": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("OBJECTIVE 3 QUANTUM ARCHITECTURE SMOKE SUCCESSFUL")


if __name__ == "__main__":
    main()
