from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cxr_thesis.objective1.config import Objective1Config, load_config
from cxr_thesis.objective1.features import (
    encode_clinical_features,
    extract_handcrafted_2d,
    extract_handcrafted_3d,
)
from cxr_thesis.objective1.graphs import (
    GraphSample,
    build_multimodal_graph,
    build_patch_graph_2d,
    build_patch_graph_3d,
)
from cxr_thesis.objective1.manifest import build_nih_manifest, validate_manifest
from cxr_thesis.objective1.pipeline import run_cxr_manifest
from cxr_thesis.objective1.preprocessing import (
    preprocess_ct_volume,
    resize_with_padding,
    restore_mask,
    transform_mask,
)
from cxr_thesis.objective1.segmentation_data import ROISegmentationDataset
from cxr_thesis.objective1.segmentation import (
    UNet2D,
    dice_score,
    hausdorff95,
    iou_score,
    postprocess_binary_mask,
)


class ManifestTests(unittest.TestCase):
    def test_patient_leakage_is_rejected(self) -> None:
        frame = pd.DataFrame(
            [
                dict(dataset="demo", patient_id="p1", study_id="s1", image_id="i1", image_path="a.png", modality="CXR", view="PA", split="train"),
                dict(dataset="demo", patient_id="p1", study_id="s2", image_id="i2", image_path="b.png", modality="CXR", view="PA", split="test"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            validate_manifest(frame)

    def test_build_nih_manifest_preserves_patient_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            train_names, test_names = [], []
            for patient in range(1, 6):
                name = f"{patient:08d}_000.png"
                rows.append(
                    {
                        "Image Index": name,
                        "Finding Labels": "Effusion" if patient % 2 else "No Finding",
                        "Follow-up #": 0,
                        "Patient ID": patient,
                        "Patient Age": 50 + patient,
                        "Patient Gender": "M" if patient % 2 else "F",
                        "View Position": "PA",
                        "OriginalImagePixelSpacing_x": 0.14,
                        "OriginalImagePixelSpacing_y": 0.14,
                    }
                )
                (test_names if patient == 5 else train_names).append(name)
            pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
            (root / "train.txt").write_text("\n".join(train_names), encoding="utf-8")
            (root / "test.txt").write_text("\n".join(test_names), encoding="utf-8")
            frame = build_nih_manifest(
                root / "metadata.csv",
                root / "train.txt",
                root / "test.txt",
                root / "images",
                val_fraction=0.25,
                seed=7,
            )
            summary = validate_manifest(frame)
            self.assertEqual(summary["rows"], 5)
            self.assertEqual(frame.loc[frame.image_id.str.contains("00000005"), "split"].item(), "test")
            self.assertEqual(set(frame.split), {"train", "val", "test"})
            self.assertIn("pixel_spacing_x", frame.columns)

    def test_build_nih_manifest_resolves_kaggle_nested_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "images_001" / "images"
            nested.mkdir(parents=True)
            rows = []
            for patient in range(1, 4):
                name = f"{patient:08d}_000.png"
                Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(nested / name)
                rows.append(
                    {
                        "Image Index": name,
                        "Finding Labels": "No Finding",
                        "Follow-up #": 0,
                        "Patient ID": patient,
                    }
                )
            pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
            (root / "train.txt").write_text("00000001_000.png\n00000002_000.png\n", encoding="utf-8")
            (root / "test.txt").write_text("00000003_000.png\n", encoding="utf-8")
            frame = build_nih_manifest(
                root / "metadata.csv",
                root / "train.txt",
                root / "test.txt",
                root,
                val_fraction=0.5,
            )
            self.assertTrue(all(Path(path).is_file() for path in frame["image_path"]))


class PreprocessingTests(unittest.TestCase):
    def test_resize_preserves_aspect_and_mask_alignment(self) -> None:
        image = np.arange(50 * 100, dtype=np.uint16).reshape(50, 100)
        resized, geometry = resize_with_padding(image, (200, 200))
        mask = np.ones_like(image, dtype=np.uint8)
        transformed = transform_mask(mask, geometry)
        restored = restore_mask(transformed, geometry)
        self.assertEqual(resized.shape, (200, 200))
        self.assertEqual((geometry.resized_height, geometry.resized_width), (100, 200))
        self.assertEqual(int(transformed.sum()), 100 * 200)
        self.assertFalse(transformed[:50].any())
        np.testing.assert_array_equal(restored, mask)

    def test_ct_window_and_resampling(self) -> None:
        volume = np.linspace(-1200, 600, 4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
        output, metadata = preprocess_ct_volume(volume, (2.0, 1.0, 1.0), Objective1Config().preprocessing)
        self.assertEqual(output.shape, (8, 8, 8))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)
        self.assertEqual(metadata["target_spacing"], [1.0, 1.0, 1.0])


class SegmentationTests(unittest.TestCase):
    def test_metrics(self) -> None:
        target = np.zeros((32, 32), dtype=np.uint8)
        target[8:24, 8:24] = 1
        self.assertAlmostEqual(dice_score(target, target), 1.0)
        self.assertAlmostEqual(iou_score(target, target), 1.0)
        self.assertAlmostEqual(hausdorff95(target, target), 0.0)

    def test_unet_output_and_postprocessing(self) -> None:
        model = UNet2D(channels=(4, 8, 16, 32))
        output = model(torch.randn(2, 1, 65, 71))
        self.assertEqual(tuple(output.shape), (2, 1, 65, 71))
        probability = np.zeros((32, 32), dtype=np.float32)
        probability[5:20, 5:20] = 0.9
        probability[28:30, 28:30] = 0.9
        mask = postprocess_binary_mask(probability, Objective1Config().segmentation)
        self.assertTrue(mask[10, 10])


class FeatureAndGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.ogrid[:64, :64]
        self.image = (x + y).astype(np.float32)
        self.mask = ((x - 32) ** 2 / 24**2 + (y - 32) ** 2 / 28**2 <= 1).astype(np.uint8)

    def test_feature_families(self) -> None:
        features = extract_handcrafted_2d(self.image, self.mask)
        self.assertIn("roi_lbp_hist_00", features)
        self.assertIn("roi_hog_hist_00", features)
        self.assertIn("roi_left_right_asymmetry", features)
        clinical = encode_clinical_features({"age": 60, "sex": "F", "view": "PA"})
        self.assertEqual(clinical["clinical_sex_female"], 1.0)
        volume = np.stack([self.image] * 4)
        mask3d = np.stack([self.mask] * 4)
        features3d = extract_handcrafted_3d(volume, mask3d, (2.0, 1.0, 1.0))
        self.assertGreater(features3d["roi3d_volume_mm3"], 0)

    def test_2d_3d_and_multimodal_graphs(self) -> None:
        graph2d = build_patch_graph_2d(self.image, self.mask, grid=(4, 4), knn_k=2)
        graph2d.validate()
        self.assertEqual(graph2d.node_position.shape[1], 2)
        volume = np.stack([self.image] * 8)
        mask3d = np.stack([self.mask] * 8)
        graph3d = build_patch_graph_3d(volume, mask3d, grid=(2, 4, 4), knn_k=1)
        graph3d.validate()
        self.assertEqual(graph3d.node_position.shape[1], 3)
        multimodal = build_multimodal_graph(
            {"left_lung": np.ones(8), "right_lung": np.zeros(8)},
            {"texture": 0.2, "shape": 0.4},
            {"age": 0.5, "view": 1.0},
        )
        self.assertEqual(set(multimodal.node_type), {"roi", "radiomics", "clinical"})
        with tempfile.TemporaryDirectory() as directory:
            path = multimodal.save(Path(directory) / "graph.npz")
            loaded = GraphSample.load(path)
            np.testing.assert_array_equal(multimodal.edge_index, loaded.edge_index)

    def test_end_to_end_manifest_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.fromarray(self.image.astype(np.uint8)).save(image_path)
            Image.fromarray(self.mask.astype(np.uint8) * 255).save(mask_path)
            frame = pd.DataFrame(
                [
                    dict(
                        dataset="demo",
                        patient_id="p1",
                        study_id="s1",
                        image_id="i1",
                        image_path=str(image_path),
                        mask_path=str(mask_path),
                        modality="CXR",
                        view="PA",
                        split="train",
                        age=45,
                        sex="F",
                    )
                ]
            )
            output = run_cxr_manifest(frame, Objective1Config(), root / "derived")
            self.assertEqual(len(output), 1)
            self.assertTrue((root / "derived" / "graphs" / "i1.npz").is_file())
            self.assertTrue((root / "derived" / "features.csv").is_file())
            dataset = ROISegmentationDataset(
                frame,
                root,
                Objective1Config().preprocessing,
                split="train",
            )
            image_tensor, mask_tensor, image_id = dataset[0]
            self.assertEqual(tuple(image_tensor.shape), (1, 224, 224))
            self.assertEqual(tuple(mask_tensor.shape), (1, 224, 224))
            self.assertEqual(image_id, "i1")


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "objective1" / "default.yaml")
        self.assertEqual(config.graph.patch_grid_2d, (7, 7))


if __name__ == "__main__":
    unittest.main()
