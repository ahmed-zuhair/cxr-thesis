from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cxr_thesis.objective1.config import load_config
from cxr_thesis.objective1.graphs import GraphSample
from cxr_thesis.objective2.cohort_recovery import (
    greedy_complete_patient_selection,
    recover_exact_cohort_bytes,
    serialize_cohort,
    sha256_bytes,
)
from cxr_thesis.objective2.data import (
    GraphClassificationDataset,
    ImageClassificationDataset,
    collate_graph_samples,
)
from cxr_thesis.objective2.evaluation import paired_bootstrap_comparison
from cxr_thesis.objective2.graph_generation import build_frozen_roi_graph
from cxr_thesis.objective2.losses import AsymmetricLoss, transform_positive_weights
from cxr_thesis.objective2.metrics import multilabel_metrics, select_f1_thresholds
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective2.training import (
    restore_rng_state,
    save_training_state,
)
from scripts.evaluate_objective2_locked_test import validate_final_lock_payload


class Objective2CohortRecoveryTests(unittest.TestCase):
    def test_label_blind_exact_complete_patient_recovery(self) -> None:
        rows = []
        patient_counts = {"1": 2, "2": 1, "10": 3, "11": 2}
        image_index = 0
        for patient_id, count in patient_counts.items():
            for _ in range(count):
                rows.append(
                    {
                        "dataset": "synthetic",
                        "patient_id": patient_id,
                        "study_id": f"study-{image_index}",
                        "image_id": f"image-{image_index:03d}",
                        "image_path": f"/private/image-{image_index:03d}.png",
                        "modality": "CXR",
                        "view": "PA",
                        "split": "test",
                        "label_a": image_index % 2,
                    }
                )
                image_index += 1
        full = pd.DataFrame(rows)
        identity = full[["patient_id", "split"]].copy()
        numeric_order = sorted(patient_counts, key=lambda value: int(value))
        selected = greedy_complete_patient_selection(
            identity,
            ordered_patients=numeric_order,
            seed=42,
            target_images=5,
        )
        expected = serialize_cohort(
            full,
            selected_patients=selected,
            row_order="manifest",
            selection_order=selected,
        )
        recovered, record = recover_exact_cohort_bytes(
            identity,
            full,
            split="test",
            seed=42,
            target_images=5,
            expected_patients=len(selected),
            expected_sha256=sha256_bytes(expected),
        )
        self.assertEqual(recovered, expected)
        self.assertFalse(record["selection_used_labels"])

        changed_labels = full.copy()
        changed_labels["label_a"] = 1 - changed_labels["label_a"]
        changed_identity = changed_labels[["patient_id", "split"]]
        selected_after_label_change = greedy_complete_patient_selection(
            changed_identity,
            ordered_patients=numeric_order,
            seed=42,
            target_images=5,
        )
        self.assertEqual(selected_after_label_change, selected)

    def test_locked_test_cohort_recovery_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(
                    repository / "scripts" / "recover_objective2_locked_test_cohort.py"
                ),
                "--help",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-test-sha256", result.stdout)


class Objective2LockedTestGuardTests(unittest.TestCase):
    def test_remote_final_lock_requires_exact_frozen_signature(self) -> None:
        checkpoint_hashes = {
            model: f"hash-{model}"
            for model in ("cnn", "attention_cnn", "vit", "gcn", "gat")
        }
        payload = {
            "test_manifest_sha256": "test-hash",
            "checkpoint_sha256": checkpoint_hashes,
            "completed_models": ["cnn", "attention_cnn", "vit", "gcn", "gat"],
            "validation_thresholds_reused_without_change": True,
            "test_used_for_model_selection": False,
            "test_evaluated": True,
        }
        validate_final_lock_payload(
            payload,
            expected_test_sha256="test-hash",
            checkpoint_hashes=checkpoint_hashes,
        )
        invalid = dict(payload)
        invalid["test_manifest_sha256"] = "different"
        with self.assertRaises(RuntimeError):
            validate_final_lock_payload(
                invalid,
                expected_test_sha256="test-hash",
                checkpoint_hashes=checkpoint_hashes,
            )


class Objective2ModelTests(unittest.TestCase):
    def test_three_image_models(self) -> None:
        image = torch.rand(2, 1, 32, 32)
        clinical = torch.rand(2, 9)
        for name in ("cnn", "attention_cnn", "vit"):
            model = build_classifier(name, 12, image_size=32)
            output = model(image, clinical)
            self.assertEqual(tuple(output.shape), (2, 12))
            output.mean().backward()

    def test_gcn_and_gat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(2):
                image_id = f"image-{index}"
                GraphSample(
                    x=np.random.default_rng(index)
                    .normal(size=(4, 7))
                    .astype(np.float32),
                    edge_index=np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64),
                    edge_attr=np.zeros((4, 5), dtype=np.float32),
                    node_type=np.asarray(["image_patch"] * 4),
                    node_position=np.zeros((4, 2), dtype=np.float32),
                ).save(root / f"{image_id}.npz")
                rows.append(
                    {
                        "image_id": image_id,
                        "age": 50,
                        "sex": "M",
                        "view": "PA",
                        "label_a": index,
                        "label_b": 1 - index,
                    }
                )
            dataset = GraphClassificationDataset(
                pd.DataFrame(rows), ["label_a", "label_b"], root
            )
            batch = collate_graph_samples([dataset[0], dataset[1]])
            for name in ("gcn", "gat"):
                model = build_classifier(name, 2, node_dim=7)
                output = model(batch)
                self.assertEqual(tuple(output.shape), (2, 2))
                output.mean().backward()

    def test_enhanced_densenet121_forward_without_downloading_weights(self) -> None:
        try:
            import torchvision  # noqa: F401
        except ImportError:
            self.skipTest("torchvision is not installed")
        model = build_classifier(
            "densenet121", 12, image_size=64, pretrained=False, dropout=0.2
        )
        output = model(torch.rand(2, 3, 64, 64), torch.rand(2, 9))
        self.assertEqual(tuple(output.shape), (2, 12))
        output.mean().backward()


class Objective2EnhancedTrainingTests(unittest.TestCase):
    def test_epoch_varying_cxr_augmentation_is_reproducible(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            image = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 4
            cv2.imwrite(str(image_path), image)
            manifest = pd.DataFrame(
                [
                    {
                        "image_path": str(image_path),
                        "age": 50,
                        "sex": "F",
                        "view": "PA",
                        "label_a": 1,
                    }
                ]
            )
            dataset = ImageClassificationDataset(
                manifest,
                ["label_a"],
                image_size=64,
                augment=True,
                seed=42,
                augmentation_profile="cxr_mild",
                epoch_varying_augmentation=True,
                output_channels=3,
                normalisation="imagenet",
            )
            dataset.set_epoch(1)
            first = dataset[0]["image"].clone()
            repeated = dataset[0]["image"].clone()
            dataset.set_epoch(2)
            second = dataset[0]["image"].clone()
            self.assertEqual(tuple(first.shape), (3, 64, 64))
            self.assertTrue(torch.equal(first, repeated))
            self.assertFalse(torch.equal(first, second))

    def test_imbalance_controls_are_finite_and_bounded(self) -> None:
        weights = transform_positive_weights(
            np.asarray([1, 25, 50], dtype=np.float32),
            100,
            transform="sqrt",
            maximum=5.0,
        )
        self.assertTrue(np.isfinite(weights).all())
        self.assertLessEqual(float(weights.max()), 5.0)
        logits = torch.tensor([[2.0, -1.0], [-2.0, 1.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = AsymmetricLoss()(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_enhanced_training_cli_writes_test_blind_checkpoint(self) -> None:
        try:
            import torchvision  # noqa: F401
        except ImportError:
            self.skipTest("torchvision is not installed")
        import cv2

        repository = Path(__file__).resolve().parents[1]
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
            rows = []
            for index in range(6):
                image_path = root / f"image-{index}.png"
                image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
                image = np.roll(image, index * 3, axis=1)
                cv2.imwrite(str(image_path), image)
                row = {
                    "dataset": "synthetic",
                    "patient_id": str(index),
                    "image_id": f"image-{index}",
                    "image_path": str(image_path),
                    "split": "train" if index < 4 else "val",
                    "age": 40 + index,
                    "sex": "F" if index % 2 else "M",
                    "view": "PA",
                }
                for label_index, label in enumerate(labels):
                    row[f"label_{label}"] = (index + label_index) % 2
                rows.append(row)
            frame = pd.DataFrame(rows)
            train_path = root / "train.csv"
            validation_path = root / "val.csv"
            frame.iloc[:4].to_csv(train_path, index=False)
            frame.iloc[4:].to_csv(validation_path, index=False)
            output = root / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(repository / "scripts" / "train_objective2_classifier.py"),
                    "--model",
                    "densenet121",
                    "--train-manifest",
                    str(train_path),
                    "--val-manifest",
                    str(validation_path),
                    "--output-dir",
                    str(output),
                    "--data-root",
                    "/",
                    "--epochs",
                    "1",
                    "--patience",
                    "1",
                    "--batch-size",
                    "2",
                    "--workers",
                    "0",
                    "--image-size",
                    "64",
                    "--augmentation-profile",
                    "cxr_mild",
                    "--epoch-varying-augmentation",
                    "--positive-weight-transform",
                    "sqrt",
                    "--max-positive-weight",
                    "5",
                    "--scheduler",
                    "cosine",
                    "--accumulation-steps",
                    "2",
                    "--gradient-clip-norm",
                    "1",
                    "--no-amp",
                ],
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHONPATH": str(repository / "src")},
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            checkpoint = torch.load(
                output / "best.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["model_name"], "densenet121")
            self.assertFalse(checkpoint["test_evaluated"])
            self.assertEqual(checkpoint["model_config"]["input_channels"], 3)
            self.assertTrue((output / "last.sha256").is_file())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for float16 AMP")
    def test_gat_cuda_amp_message_dtype_matches_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            GraphSample(
                x=np.random.default_rng(42).normal(size=(8, 7)).astype(np.float32),
                edge_index=np.asarray(
                    [
                        [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6],
                        [0, 1, 2, 3, 4, 5, 6, 7, 1, 3, 5, 7],
                    ],
                    dtype=np.int64,
                ),
                edge_attr=np.zeros((12, 5), dtype=np.float32),
                node_type=np.asarray(["image_patch"] * 8),
                node_position=np.zeros((8, 2), dtype=np.float32),
            ).save(root / "image-amp.npz")
            row = {
                "image_id": "image-amp",
                "age": 50,
                "sex": "M",
                "view": "PA",
                "label_a": 1,
                "label_b": 0,
            }
            dataset = GraphClassificationDataset(
                pd.DataFrame([row]), ["label_a", "label_b"], root
            )
            batch = collate_graph_samples([dataset[0]]).to("cuda")
            model = build_classifier("gat", 2, node_dim=7).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(batch)
            self.assertEqual(tuple(output.shape), (1, 2))
            self.assertTrue(torch.isfinite(output).all())
            output.mean().backward()


class Objective2GraphGenerationTests(unittest.TestCase):
    def test_graph_shard_recovery_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repository / "src")
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "generate_objective2_graph_shards.py"),
                "--help",
            ],
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--hf-repo", result.stdout)

    def test_graph_generation_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repository / "src")
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "generate_objective2_graphs.py"),
                "--help",
            ],
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--graph-dir", result.stdout)

    def test_frozen_probability_builds_private_roi_graph_without_mask_file(
        self,
    ) -> None:
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "objective1"
            / "default.yaml"
        )
        image = np.zeros((224, 224), dtype=np.uint8)
        probability = np.zeros((224, 224), dtype=np.float32)
        image[35:195, 35:100] = 100
        image[35:195, 124:189] = 180
        probability[35:195, 35:100] = 0.95
        probability[35:195, 124:189] = 0.95
        record = {
            "image_id": "image-1",
            "patient_id": "patient-1",
            "dataset": "synthetic",
            "split": "train",
        }
        result = build_frozen_roi_graph(
            image,
            probability,
            threshold=0.55,
            config=config,
            record=record,
            checkpoint_sha256="abc123",
        )
        self.assertEqual(result.graph.x.shape[1], 7)
        self.assertGreater(result.graph.x.shape[0], 0)
        self.assertGreater(result.graph.edge_index.shape[1], 0)
        self.assertTrue(result.mask_quality["is_nonempty"])
        self.assertEqual(
            result.graph.metadata["mask_source"],
            "frozen_adapted_unet_probability",
        )
        self.assertEqual(result.graph.metadata["mask_checkpoint_sha256"], "abc123")

    def test_empty_frozen_probability_is_rejected(self) -> None:
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "objective1"
            / "default.yaml"
        )
        with self.assertRaisesRegex(ValueError, "empty ROI"):
            build_frozen_roi_graph(
                np.zeros((224, 224), dtype=np.uint8),
                np.zeros((224, 224), dtype=np.float32),
                threshold=0.55,
                config=config,
                record={
                    "image_id": "image-1",
                    "patient_id": "patient-1",
                    "dataset": "synthetic",
                    "split": "train",
                },
                checkpoint_sha256="abc123",
            )


class Objective2MetricTests(unittest.TestCase):
    def test_metrics_and_validation_thresholds(self) -> None:
        probabilities = np.asarray(
            [[0.9, 0.2], [0.8, 0.3], [0.2, 0.7], [0.1, 0.8]], dtype=float
        )
        targets = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=int)
        thresholds = select_f1_thresholds(probabilities, targets)
        metrics = multilabel_metrics(probabilities, targets, thresholds=thresholds)
        self.assertEqual(thresholds.shape, (2,))
        self.assertAlmostEqual(metrics["macro"]["auroc"], 1.0)
        self.assertAlmostEqual(metrics["macro"]["auprc"], 1.0)
        self.assertAlmostEqual(metrics["macro"]["f1"], 1.0)

    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        targets = np.asarray(
            [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0]], dtype=np.int8
        )
        reference = np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.8, 0.8], [0.2, 0.2]]
        )
        weaker = 1.0 - reference
        arguments = {
            "probabilities": {"cnn": reference, "gat": weaker},
            "targets": targets,
            "thresholds": {
                "cnn": np.asarray([0.5, 0.5]),
                "gat": np.asarray([0.5, 0.5]),
            },
            "reference_model": "cnn",
            "replicates": 20,
            "seed": 42,
        }
        first = paired_bootstrap_comparison(**arguments)
        second = paired_bootstrap_comparison(**arguments)
        self.assertEqual(first, second)
        difference = first["paired_model_minus_reference"]["gat"]["auroc"]
        self.assertLess(difference["model_minus_reference_mean"], 0.0)


class Objective2RecoveryTests(unittest.TestCase):
    def test_locked_test_graph_shard_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(
                    repository
                    / "scripts"
                    / "generate_objective2_locked_test_graph_shards.py"
                ),
                "--help",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--test-manifest", result.stdout)

    def test_locked_test_evaluation_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "evaluate_objective2_locked_test.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-test-sha256", result.stdout)

    def test_locked_test_publication_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "publish_objective2_locked_test.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--evaluation-output", result.stdout)

    def test_five_model_locked_test_evaluation_is_finalized_once(self) -> None:
        repository = Path(__file__).resolve().parents[1]
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

        def file_hash(path: Path) -> str:
            digest = hashlib.sha256()
            digest.update(path.read_bytes())
            return digest.hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_root = root / "graphs"
            checkpoint_root = root / "checkpoints"
            graph_root.mkdir()
            checkpoint_root.mkdir()
            rows = []
            for index in range(2):
                image_path = root / f"image-{index}.png"
                import cv2

                cv2.imwrite(
                    str(image_path), np.full((32, 32), 80 + index * 80, dtype=np.uint8)
                )
                GraphSample(
                    x=np.random.default_rng(index)
                    .normal(size=(4, 7))
                    .astype(np.float32),
                    edge_index=np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64),
                    edge_attr=np.zeros((4, 5), dtype=np.float32),
                    node_type=np.asarray(["image_patch"] * 4),
                    node_position=np.zeros((4, 2), dtype=np.float32),
                ).save(graph_root / f"image-{index}.npz")
                row = {
                    "dataset": "synthetic",
                    "patient_id": str(index),
                    "study_id": f"study-{index}",
                    "image_id": f"image-{index}",
                    "image_path": str(image_path),
                    "modality": "CXR",
                    "view": "PA",
                    "split": "test",
                    "age": 50,
                    "sex": "M",
                }
                for label_index, label in enumerate(labels):
                    row[f"label_{label}"] = (index + label_index) % 2
                rows.append(row)
            manifest = root / "test.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)

            model_names = ("cnn", "attention_cnn", "vit", "gcn", "gat")
            checkpoint_paths = {}
            for model_name in model_names:
                model = build_classifier(
                    model_name, len(labels), image_size=32, node_dim=7
                )
                checkpoint_path = checkpoint_root / f"{model_name}.pt"
                torch.save(
                    {
                        "model_name": model_name,
                        "model_state": model.state_dict(),
                        "label_names": labels,
                        "epoch": 1,
                        "validation_thresholds": [0.5] * len(labels),
                        "validation_metrics": {"macro": {"auroc": 0.5}},
                        "test_evaluated": False,
                    },
                    checkpoint_path,
                )
                checkpoint_paths[model_name] = checkpoint_path

            output = root / "evaluation"
            command = [
                sys.executable,
                str(repository / "scripts" / "evaluate_objective2_locked_test.py"),
                "--test-manifest",
                str(manifest),
                "--graph-root",
                str(graph_root),
                "--output-dir",
                str(output),
                "--data-root",
                "/",
                "--expected-test-sha256",
                file_hash(manifest),
                "--expected-test-cases",
                "2",
                "--expected-test-patients",
                "2",
                "--image-size",
                "32",
                "--image-batch-size",
                "2",
                "--graph-batch-size",
                "2",
                "--workers",
                "0",
                "--bootstrap-replicates",
                "5",
            ]
            for model_name, checkpoint_path in checkpoint_paths.items():
                option = model_name.replace("_", "-")
                command.extend(
                    [
                        f"--{option}-checkpoint",
                        str(checkpoint_path),
                        f"--expected-{option}-sha256",
                        file_hash(checkpoint_path),
                    ]
                )
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            lock_path = output / "FINAL_LOCKED_TEST_EVALUATION.json"
            self.assertTrue(lock_path.is_file())
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(lock["completed_models"], list(model_names))
            self.assertTrue(lock["test_evaluated"])
            repeated = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already finalized", repeated.stderr)

    def test_candidate_publication_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "publish_objective2_candidate.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--release-tag", result.stdout)
        self.assertIn("densenet121", result.stdout)

    def test_graph_fresh_runtime_recovery_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repository / "src")
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "recover_objective2_graph_shards.py"),
                "--help",
            ],
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--cohort-root", result.stdout)

    def test_private_training_recovery_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repository / "src")
        result = subprocess.run(
            [
                sys.executable,
                str(
                    repository / "scripts" / "train_objective2_with_private_recovery.py"
                ),
                "--help",
            ],
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--hf-path", result.stdout)

    def test_epoch_recovery_is_complete_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "last.pt"
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            generator = torch.Generator().manual_seed(42)
            generator_state = generator.get_state().clone()
            save_training_state(
                target,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                data_loader_generator=generator,
                epoch_completed=3,
                best_auroc=0.75,
                stale_epochs=1,
                history=[{"epoch": 3, "validation_macro_auroc": 0.75}],
                signature={"model": "attention_cnn", "test_cases_accessed": 0},
                best_checkpoint_sha256="abc",
                resume_count=0,
            )
            self.assertTrue(target.is_file())
            self.assertFalse((target.parent / ".last.pt.tmp").exists())
            state = torch.load(target, map_location="cpu", weights_only=False)
            self.assertEqual(state["epoch_completed"], 3)
            self.assertFalse(state["test_evaluated"])
            self.assertTrue(
                torch.equal(state["data_loader_generator_state"], generator_state)
            )
            random_before = random.random()
            restore_rng_state(state["rng_state"])
            random_after_first_restore = random.random()
            restore_rng_state(state["rng_state"])
            random_after_second_restore = random.random()
            self.assertEqual(random_before, random_after_first_restore)
            self.assertEqual(random_after_first_restore, random_after_second_restore)


if __name__ == "__main__":
    unittest.main()
