from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective6.cohorts import (
    derive_padchest_age,
    patient_partition,
)
from cxr_thesis.objective6.data import (
    collate_reports,
    select_label_complete_subset,
)
from cxr_thesis.objective6.evaluation import (
    bleu_statistics,
    cider_d_score,
    cider_document_frequency,
    clinical_scores,
    corpus_bleu,
    exact_token_meteor,
    explicit_contradictions,
    parse_padchest6_labels,
    repeated_ngram,
    rouge_l_f1,
)
from cxr_thesis.objective6.models import DenseNetTransformerReportGenerator
from cxr_thesis.objective6.text import (
    ReportVocabulary,
    normalise_report,
    tokenise_report,
)
from scripts.extract_objective6_retrieval_embedding_shard import (
    MANIFESTS as RETRIEVAL_WORKER_MANIFESTS,
)
from scripts.extract_objective6_retrieval_embeddings_with_recovery import (
    MANIFESTS as RETRIEVAL_WRAPPER_MANIFESTS,
)
from scripts.train_objective6_with_private_recovery import snapshot


class Objective6TextTests(unittest.TestCase):
    def test_spanish_normalisation_and_tokenisation(self) -> None:
        self.assertEqual(normalise_report("  Sin   hallazgos ÁGUDOS. "), "sin hallazgos águdos.")
        self.assertEqual(
            tokenise_report("Sin hallazgos águdos."),
            ["sin", "hallazgos", "águdos", "."],
        )

    def test_vocabulary_is_deterministic_and_round_trips(self) -> None:
        reports = ["sin hallazgos .", "sin derrame pleural .", "derrame pleural ."]
        first = ReportVocabulary.build(reports, minimum_frequency=1, maximum_size=20)
        second = ReportVocabulary.build(reversed(reports), minimum_frequency=1, maximum_size=20)
        self.assertEqual(first.tokens, second.tokens)
        encoded = first.encode("sin derrame pleural .", maximum_length=10)
        self.assertEqual(first.decode(encoded), "sin derrame pleural.")
        self.assertEqual(ReportVocabulary.from_dict(first.to_dict()), first)

    def test_locked_report_metrics_reward_an_exact_match(self) -> None:
        reference = tokenise_report("Sin derrame pleural ni neumotórax.")
        different = tokenise_report("Cardiomegalia.")
        exact = bleu_statistics(reference, reference)
        mismatch = bleu_statistics(reference, different)
        self.assertAlmostEqual(corpus_bleu(np.stack([exact]), 4), 1.0)
        self.assertGreater(corpus_bleu(np.stack([exact]), 4), corpus_bleu(np.stack([mismatch]), 4))
        self.assertAlmostEqual(rouge_l_f1(reference, reference), 1.0)
        self.assertGreater(exact_token_meteor(reference, reference), 0.99)
        frequency = cider_document_frequency([reference, different])
        self.assertGreater(
            cider_d_score(reference, reference, frequency, 2),
            cider_d_score(reference, different, frequency, 2),
        )

    def test_locked_clinical_metrics_and_safety(self) -> None:
        labels = parse_padchest6_labels("Cardiomegaly|Pleural Effusion")
        self.assertEqual(labels.tolist(), [0, 1, 0, 0, 1, 0])
        score = clinical_scores(np.stack([labels]), np.stack([labels]))
        self.assertEqual(score["micro_concept_f1"], 1.0)
        contradictions, mentions = explicit_contradictions(
            "Sin cardiomegalia. Derrame pleural.", labels
        )
        self.assertEqual((contradictions, mentions), (1, 2))
        self.assertTrue(repeated_ngram(tokenise_report("a b c d a b c d")))


class Objective6ModelTests(unittest.TestCase):
    def test_forward_and_generation_shapes_without_weight_download(self) -> None:
        model = DenseNetTransformerReportGenerator(
            32,
            d_model=32,
            heads=4,
            layers=1,
            feedforward_dim=64,
            maximum_length=12,
            pretrained=False,
        )
        model.eval()
        image = torch.rand(2, 3, 64, 64)
        clinical = torch.rand(2, 9)
        input_ids = torch.tensor([[1, 5, 6, 2], [1, 7, 8, 2]])
        with torch.no_grad():
            output = model(image, clinical, input_ids)
            generated = model.generate(image, clinical, maximum_length=5)
        self.assertEqual(output["report_logits"].shape, (2, 4, 32))
        self.assertEqual(output["clinical_logits"].shape, (2, 6))
        self.assertEqual(generated.shape[0], 2)
        self.assertLessEqual(generated.shape[1], 5)
        self.assertFalse(any(parameter.requires_grad for parameter in model.image_encoder.parameters()))

    def test_v1_1_concept_token_partial_unfreezing_and_beam_search(self) -> None:
        baseline = DenseNetTransformerReportGenerator(
            24, d_model=32, heads=4, layers=1, feedforward_dim=64,
            maximum_length=10, pretrained=False, use_concept_token=False,
        )
        self.assertFalse(any("concept_projection" in key for key in baseline.state_dict()))
        enhanced = DenseNetTransformerReportGenerator(
            24, d_model=32, heads=4, layers=1, feedforward_dim=64,
            maximum_length=10, pretrained=False, use_concept_token=True,
        )
        load = enhanced.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(
            set(load.missing_keys),
            {
                "concept_projection.0.weight", "concept_projection.0.bias",
                "concept_projection.2.weight", "concept_projection.2.bias",
            },
        )
        self.assertFalse(load.unexpected_keys)
        enhanced.set_final_image_block_trainable()
        trainable = {
            name for name, parameter in enhanced.image_encoder.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(
            name.startswith(("denseblock4", "norm5"))
            for name in trainable
        ))
        enhanced.eval()
        with torch.no_grad():
            generated = enhanced.generate_beam(
                torch.rand(1, 3, 64, 64), torch.rand(1, 9),
                maximum_length=6, beam_width=3,
                length_normalization_alpha=0.7, no_repeat_ngram_size=4,
            )
        self.assertEqual(generated.shape[0], 1)
        self.assertLessEqual(generated.shape[1], 6)


class Objective6CliTests(unittest.TestCase):
    def test_retrieval_manifest_hashes_are_complete_and_consistent(self) -> None:
        self.assertEqual(RETRIEVAL_WORKER_MANIFESTS, RETRIEVAL_WRAPPER_MANIFESTS)
        for _, digest in RETRIEVAL_WRAPPER_MANIFESTS.values():
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(character in "0123456789abcdef" for character in digest))

    def test_private_recovery_snapshot_accepts_existing_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            checkpoint = output / "last.pt"
            torch.save({"epoch_completed": 3, "test_evaluated": False}, checkpoint)
            checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (output / "last.pt.sha256").write_text(
                f"{checksum}  last.pt\n", encoding="utf-8"
            )
            target = root / "already_created"
            target.mkdir()

            paths, epoch = snapshot(output, target)

            self.assertEqual(epoch, 3)
            self.assertEqual(
                {path.name for path in paths}, {"last.pt", "last.pt.sha256"}
            )
            self.assertTrue((target / "last.pt").is_file())

    def test_read_only_audit_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "audit_objective6_report_data.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--objective5-private-root", result.stdout)

    def test_protocol_lock_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repository / "scripts" / "lock_objective6_report_protocol.py"), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--audit-summary", result.stdout)

    def test_protocol_publication_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repository / "scripts" / "publish_objective6_report_protocol.py"), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--private-hf-repo", result.stdout)

    def test_training_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repository / "scripts" / "train_objective6_report_generator.py"), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--source-checkpoint", result.stdout)

    def test_private_recovery_training_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repository / "scripts" / "train_objective6_with_private_recovery.py"), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-train-sha256", result.stdout)

    def test_validation_evaluation_lock_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "lock_objective6_validation_evaluation.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--multimodal-output", result.stdout)

    def test_validation_evaluation_publication_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "publish_objective6_validation_protocol.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-protocol-sha256", result.stdout)

    def test_validation_generation_shard_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for script in (
            "generate_objective6_validation_shard.py",
            "generate_objective6_validation_with_private_recovery.py",
        ):
            result = subprocess.run(
                [sys.executable, str(repository / "scripts" / script), "--help"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("--shard-count", result.stdout)

    def test_retrieval_baseline_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        scripts = (
            "extract_objective6_retrieval_embedding_shard.py",
            "extract_objective6_retrieval_embeddings_with_recovery.py",
            "build_objective6_validation_retrieval.py",
        )
        for script in scripts:
            result = subprocess.run(
                [sys.executable, str(repository / "scripts" / script), "--help"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_validation_comparison_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "evaluate_objective6_validation.py"),
                "--help",
            ],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--retrieval-root", result.stdout)

    def test_validation_result_publication_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(
                    repository
                    / "scripts"
                    / "publish_objective6_validation_results.py"
                ),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-summary-sha256", result.stdout)

    def test_enhancement_protocol_clis_import(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for script, expected in (
            ("lock_objective6_enhancement_protocol.py", "--v1-summary"),
            ("publish_objective6_enhancement_protocol.py", "--lock-directory"),
        ):
            result = subprocess.run(
                [sys.executable, str(repository / "scripts" / script), "--help"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn(expected, result.stdout)

    def test_enhancement_training_clis_import(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for script in (
            "train_objective6_enhanced_report_generator.py",
            "train_objective6_enhanced_with_private_recovery.py",
        ):
            result = subprocess.run(
                [sys.executable, str(repository / "scripts" / script), "--help"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("--enhancement-protocol", result.stdout)

    def test_enhancement_validation_generation_clis_import(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for script in (
            "generate_objective6_enhanced_validation_shard.py",
            "generate_objective6_enhanced_validation_with_private_recovery.py",
        ):
            result = subprocess.run(
                [sys.executable, str(repository / "scripts" / script), "--help"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("--enhancement-protocol", result.stdout)

    def test_enhancement_validation_evaluation_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(
                    repository
                    / "scripts"
                    / "evaluate_objective6_enhancement_validation.py"
                ),
                "--help",
            ],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--enhanced-root", result.stdout)

    def test_enhancement_validation_publication_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(
                    repository
                    / "scripts"
                    / "publish_objective6_enhancement_validation.py"
                ),
                "--help",
            ],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-summary-sha256", result.stdout)

    def test_training_cli_writes_resumable_test_blind_checkpoint(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(4):
                image_path = root / f"image_{index}.png"
                Image.fromarray(
                    np.full((64, 64), 32 + index * 20, dtype=np.uint8)
                ).save(image_path)
                rows.append({
                    "image_path": str(image_path),
                    "patient_id": f"P{index}",
                    "study_id": f"S{index}",
                    "report": "sin hallazgos ." if index % 2 == 0 else "derrame pleural .",
                    "age": 40 + index,
                    "sex": "F" if index % 2 == 0 else "M",
                    "view": "PA",
                    "split": "train" if index < 2 else "val",
                })
            train_path = root / "train.csv"
            val_path = root / "val.csv"
            pd.DataFrame(rows[:2]).to_csv(train_path, index=False)
            pd.DataFrame(rows[2:]).to_csv(val_path, index=False)
            source = build_classifier("densenet121", 6, pretrained=False)
            checkpoint_path = root / "source.pt"
            torch.save({"model_state": source.state_dict()}, checkpoint_path)
            source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            output = root / "output"
            base = [
                sys.executable,
                str(repository / "scripts" / "train_objective6_report_generator.py"),
                "--variant", "multimodal",
                "--train-manifest", str(train_path),
                "--val-manifest", str(val_path),
                "--source-checkpoint", str(checkpoint_path),
                "--expected-source-sha256", source_hash,
                "--output-dir", str(output),
                "--batch-size", "2",
                "--workers", "0",
                "--image-size", "64",
                "--maximum-length", "12",
                "--maximum-vocabulary-size", "64",
                "--minimum-token-frequency", "1",
                "--patience", "4",
                "--no-amp",
            ]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(repository / "src")
            first = subprocess.run(
                [*base, "--epochs", "1"], cwd=repository, env=environment,
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)
            recovery = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(recovery["epoch_completed"], 1)
            self.assertFalse(recovery["test_evaluated"])
            resumed = subprocess.run(
                [*base, "--epochs", "2", "--resume"], cwd=repository, env=environment,
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(resumed.returncode, 0, msg=resumed.stdout + resumed.stderr)
            final = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(final["epoch_completed"], 2)
            self.assertEqual(final["resume_count"], 1)
            self.assertFalse(final["test_evaluated"])


class Objective6CohortTests(unittest.TestCase):
    def test_smoke_subset_is_deterministic_and_label_complete(self) -> None:
        labels = [
            "atelectasis", "cardiomegalia", "consolidacion", "edema",
            "derrame pleural", "neumotorax", "sin hallazgos",
        ]
        frame = pd.DataFrame({
            "case": list(range(70)),
            "labels": [labels[index % len(labels)] for index in range(70)],
        })
        first = select_label_complete_subset(frame, 20, seed=42)
        second = select_label_complete_subset(frame, 20, seed=42)
        pd.testing.assert_frame_equal(first, second)
        targets = np.stack(first["labels"].map(parse_padchest6_labels))
        self.assertTrue(np.all(targets.sum(axis=0) > 0))
        self.assertTrue(np.all(targets.sum(axis=0) < len(first)))

    def test_patient_partition_is_deterministic_and_patient_level(self) -> None:
        self.assertEqual(patient_partition("PadChest-001"), patient_partition("001"))
        self.assertIn(patient_partition("90210"), {"train", "val", "test"})

    def test_age_is_derived_from_study_year_and_birth_year(self) -> None:
        age = derive_padchest_age(
            pd.Series([20140915, 20150101, 20190101]),
            pd.Series([1930, 2010, 1800]),
        )
        self.assertEqual(float(age.iloc[0]), 84.0)
        self.assertEqual(float(age.iloc[1]), 5.0)
        self.assertTrue(pd.isna(age.iloc[2]))

    def test_report_collation_pads_only_to_batch_maximum(self) -> None:
        sample = {
            "image": torch.zeros(3, 8, 8),
            "clinical": torch.zeros(9),
            "report_ids": torch.tensor([1, 4, 2]),
        }
        longer = dict(sample)
        longer["report_ids"] = torch.tensor([1, 5, 6, 2])
        batch = collate_reports([sample, longer])
        self.assertEqual(batch["report_ids"].shape, (2, 4))
        self.assertEqual(int(batch["report_ids"][0, -1]), 0)

    def test_report_collation_preserves_clinical_guidance_targets(self) -> None:
        samples = [
            {
                "image": torch.zeros(3, 8, 8),
                "clinical": torch.zeros(9),
                "report_ids": torch.tensor([1, 2]),
                "clinical_labels": torch.tensor([1, 0, 0, 0, 1, 0]).float(),
            },
            {
                "image": torch.ones(3, 8, 8),
                "clinical": torch.ones(9),
                "report_ids": torch.tensor([1, 4, 2]),
                "clinical_labels": torch.tensor([0, 1, 0, 0, 0, 0]).float(),
            },
        ]
        batch = collate_reports(samples)
        self.assertEqual(batch["clinical_labels"].shape, (2, 6))


if __name__ == "__main__":
    unittest.main()
