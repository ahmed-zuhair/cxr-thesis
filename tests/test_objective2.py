from __future__ import annotations

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
from cxr_thesis.objective2.data import GraphClassificationDataset, collate_graph_samples
from cxr_thesis.objective2.graph_generation import build_frozen_roi_graph
from cxr_thesis.objective2.metrics import multilabel_metrics, select_f1_thresholds
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective2.training import (
    restore_rng_state,
    save_training_state,
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
                    x=np.random.default_rng(index).normal(size=(4, 7)).astype(np.float32),
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

    def test_frozen_probability_builds_private_roi_graph_without_mask_file(self) -> None:
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


class Objective2RecoveryTests(unittest.TestCase):
    def test_graph_fresh_runtime_recovery_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repository / "src")
        result = subprocess.run(
            [
                sys.executable,
                str(
                    repository
                    / "scripts"
                    / "recover_objective2_graph_shards.py"
                ),
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
                    repository
                    / "scripts"
                    / "train_objective2_with_private_recovery.py"
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
