from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cxr_thesis.objective1.graphs import GraphSample
from cxr_thesis.objective2.data import GraphClassificationDataset, collate_graph_samples
from cxr_thesis.objective2.metrics import multilabel_metrics, select_f1_thresholds
from cxr_thesis.objective2.models import build_classifier


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


if __name__ == "__main__":
    unittest.main()
