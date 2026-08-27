from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cxr_thesis.objective2.data import GraphBatch
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective3.embeddings import (
    load_embedding_shard,
    save_embedding_shard,
)
from cxr_thesis.objective3.models import (
    ClassicalMatchedBottleneck,
    HybridGraphHead,
    QuantumBottleneck,
    bottleneck_parameter_count,
)
from cxr_thesis.objective3.training import (
    apply_standardizer,
    fit_standardizer,
    initialize_shared_layers,
    load_embedding_corpus,
    shared_layer_state,
)


class Objective3ArchitectureTests(unittest.TestCase):
    def test_graph_classifier_exposes_fused_embedding(self) -> None:
        batch = GraphBatch(
            x=torch.rand(8, 7),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
            ),
            batch_index=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            clinical=torch.rand(2, 9),
            labels=torch.randint(0, 2, (2, 12)).float(),
        )
        model = build_classifier("gat", 12, node_dim=7)
        embedding = model.encode(batch)
        self.assertEqual(tuple(embedding.shape), (2, 160))
        self.assertEqual(tuple(model(batch).shape), (2, 12))

    def test_classical_control_has_frozen_parameter_budget(self) -> None:
        bottleneck = ClassicalMatchedBottleneck()
        self.assertEqual(bottleneck_parameter_count(bottleneck), 24)
        output = bottleneck(torch.rand(3, 4))
        self.assertEqual(tuple(output.shape), (3, 4))
        output.mean().backward()

    def test_hybrid_classical_head_shape(self) -> None:
        model = HybridGraphHead(12, bottleneck="classical_matched")
        output = model(torch.rand(3, 160))
        self.assertEqual(tuple(output.shape), (3, 12))

    @unittest.skipUnless(find_spec("pennylane"), "PennyLane is optional")
    def test_quantum_control_matches_classical_parameter_budget(self) -> None:
        classical = ClassicalMatchedBottleneck()
        quantum = QuantumBottleneck()
        self.assertEqual(bottleneck_parameter_count(classical), 24)
        self.assertEqual(bottleneck_parameter_count(quantum), 24)
        inputs = torch.rand(3, 4, requires_grad=True)
        output = quantum(inputs)
        self.assertEqual(tuple(output.shape), (3, 4))
        output.mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all().item())

    def test_quantum_smoke_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "smoke_objective3_quantum.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--batch-size", result.stdout)

    def test_private_embedding_shard_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            values = np.random.default_rng(42).normal(size=(3, 160)).astype(np.float32)
            save_embedding_shard(path, values, ["i1", "i2", "i3"])
            restored, identifiers = load_embedding_shard(
                path, expected_image_ids=["i1", "i2", "i3"]
            )
            self.assertTrue(np.array_equal(restored, values))
            self.assertEqual(identifiers.tolist(), ["i1", "i2", "i3"])

    def test_embedding_extraction_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "extract_objective3_gat_embeddings.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-checkpoint-sha256", result.stdout)

    def test_training_only_standardizer(self) -> None:
        training = np.vstack(
            [np.zeros((1, 160), dtype=np.float32), np.full((1, 160), 2.0)]
        )
        validation = np.full((1, 160), 3.0, dtype=np.float32)
        mean, standard_deviation = fit_standardizer(training)
        transformed_training = apply_standardizer(
            training, mean, standard_deviation
        )
        transformed_validation = apply_standardizer(
            validation, mean, standard_deviation
        )
        self.assertTrue(np.allclose(transformed_training.mean(axis=0), 0.0))
        self.assertTrue(np.allclose(transformed_training.std(axis=0), 1.0))
        self.assertTrue(np.allclose(transformed_validation, 2.0))

    def test_embedding_corpus_preserves_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = root / "shards"
            first = np.zeros((2, 160), dtype=np.float32)
            second = np.ones((1, 160), dtype=np.float32)
            save_embedding_shard(shards / "part_000.npz", first, ["a", "b"])
            save_embedding_shard(shards / "part_001.npz", second, ["c"])
            train = pd.DataFrame({"image_id": ["a", "b"]})
            validation = pd.DataFrame({"image_id": ["c"]})
            index = {
                "shards": [
                    {"shard": "part_000", "start": 0, "stop": 2},
                    {"shard": "part_001", "start": 2, "stop": 3},
                ]
            }
            restored_train, restored_validation = load_embedding_corpus(
                train, validation, index, shards
            )
            self.assertTrue(np.array_equal(restored_train, first))
            self.assertTrue(np.array_equal(restored_validation, second))

    @unittest.skipUnless(find_spec("pennylane"), "PennyLane is optional")
    def test_paired_heads_share_exact_initialization(self) -> None:
        classical = HybridGraphHead(12, bottleneck="classical_matched")
        quantum = HybridGraphHead(12, bottleneck="quantum")
        initialize_shared_layers(classical, 42)
        initialize_shared_layers(quantum, 42)
        classical_state = shared_layer_state(classical)
        quantum_state = shared_layer_state(quantum)
        self.assertEqual(classical_state.keys(), quantum_state.keys())
        for name in classical_state:
            self.assertTrue(torch.equal(classical_state[name], quantum_state[name]))

    def test_paired_training_clis_import(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for name in (
            "train_objective3_head.py",
            "train_objective3_with_private_recovery.py",
        ):
            with self.subTest(script=name):
                result = subprocess.run(
                    [sys.executable, str(repository / "scripts" / name), "--help"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("--variant", result.stdout)

    @unittest.skipUnless(find_spec("pennylane"), "PennyLane is optional")
    def test_paired_training_cli_end_to_end(self) -> None:
        labels = [
            "Infiltration",
            "Effusion",
            "Atelectasis",
            "Nodule",
            "Mass",
            "Consolidation",
            "Pneumothorax",
            "Pleural_Thickening",
            "Cardiomegaly",
            "Emphysema",
            "Edema",
            "Fibrosis",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.csv"
            validation_path = root / "validation.csv"
            embedding_root = root / "embeddings"
            shard_root = embedding_root / "private" / "shards"
            rows = []
            for index in range(12):
                row = {
                    "patient_id": f"p{index}",
                    "image_id": f"i{index}",
                    "split": "train" if index < 8 else "val",
                }
                for label_index, label in enumerate(labels):
                    row[f"label_{label}"] = (index + label_index) % 2
                rows.append(row)
            frame = pd.DataFrame(rows)
            frame.iloc[:8].to_csv(train_path, index=False)
            frame.iloc[8:].to_csv(validation_path, index=False)
            train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
            validation_hash = hashlib.sha256(validation_path.read_bytes()).hexdigest()
            values = np.random.default_rng(42).normal(size=(12, 160)).astype(np.float32)
            shard_path = save_embedding_shard(
                shard_root / "part_000.npz",
                values,
                frame["image_id"].tolist(),
            )
            shard_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            gat_hash = "a" * 64
            recovery_index = {
                "encoder": "gat",
                "encoder_frozen": True,
                "embedding_dimension": 160,
                "train_manifest_sha256": train_hash,
                "validation_manifest_sha256": validation_hash,
                "gat_checkpoint_sha256": gat_hash,
                "train_cases": 8,
                "validation_cases": 4,
                "test_manifest_opened": False,
                "test_labels_accessed": False,
                "test_evaluated": False,
                "allowed_for_public_upload": False,
                "shards": [
                    {
                        "shard": "part_000",
                        "start": 0,
                        "stop": 12,
                        "cases": 12,
                        "split": "mixed",
                        "sha256": shard_hash,
                    }
                ],
            }
            index_path = embedding_root / "private" / "embedding_recovery_index.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(recovery_index), encoding="utf-8")
            repository = Path(__file__).resolve().parents[1]
            for variant in ("classical_matched", "quantum"):
                with self.subTest(variant=variant):
                    output = root / variant
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(
                                repository
                                / "scripts"
                                / "train_objective3_head.py"
                            ),
                            "--variant",
                            variant,
                            "--train-manifest",
                            str(train_path),
                            "--val-manifest",
                            str(validation_path),
                            "--embedding-root",
                            str(embedding_root),
                            "--output-dir",
                            str(output),
                            "--expected-train-sha256",
                            train_hash,
                            "--expected-val-sha256",
                            validation_hash,
                            "--expected-gat-sha256",
                            gat_hash,
                            "--expected-train-cases",
                            "8",
                            "--expected-val-cases",
                            "4",
                            "--epochs",
                            "1",
                            "--patience",
                            "1",
                            "--batch-size",
                            "4",
                        ],
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, msg=result.stderr)
                    self.assertIn(
                        "OBJECTIVE 3 PAIRED HEAD TRAINING AND VALIDATION SUCCESSFUL",
                        result.stdout,
                    )
                    summary = json.loads(
                        (output / "validation_summary.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(summary["bottleneck_parameters"], 24)
                    self.assertEqual(summary["total_trainable_parameters"], 2648)
                    self.assertFalse(summary["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
