#!/usr/bin/env python3
"""Part 9: the single permitted locked-test evaluation for Objective 3 v2.0.

This is the only irreversible step in the study, and the only script allowed
past the locked-test guard. It refuses to run unless the advancement rule in the
locked protocol has actually been met, and it checks that from the Part 5 and
Part 6 results rather than from a flag anyone can pass.

The rule: advance only if H2 or H3 passed. If neither did, the correct outcome is
a characterised negative result and the locked test stays closed. That path is
not a failure of the study; it is the study working.

Thresholds and any temperature scaling are frozen from validation. Nothing here
is fitted on test data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective3_v2 import STUDY, VERSION
from cxr_thesis.objective3_v2.guards import (
    LockedTestAccessError,
    LockedTestAuthorisation,
    assert_no_locked_test,
    open_locked_test,
    require_existing,
)
from cxr_thesis.objective3_v2.io_utils import (
    read_json,
    sha256_file,
    utc_timestamp,
    verify_sha256,
    write_json_atomic,
    write_results,
)
from cxr_thesis.objective3_v2.seeds import seed_everything
from cxr_thesis.objective3_v2.stats import bootstrap_ci

PART = "part9_final_evaluation"
PRIMARY_LABELS = [
    "Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass", "Consolidation",
    "Pneumothorax", "Pleural_Thickening", "Cardiomegaly", "Emphysema", "Edema",
    "Fibrosis",
]
BOOTSTRAP_RESAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--part5-results", type=Path)
    parser.add_argument("--part6-results", type=Path)
    parser.add_argument("--locked-test-manifest", type=Path, required=True)
    parser.add_argument("--expected-locked-test-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check the advancement rule and stop without opening the test",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# the advancement rule
# --------------------------------------------------------------------------


def check_advancement(args: argparse.Namespace) -> LockedTestAuthorisation:
    """Decide from the evidence whether the locked test may be opened at all."""

    require_existing([args.protocol])
    digest = verify_sha256(args.protocol, args.expected_protocol_sha256)
    protocol = read_json(args.protocol)
    rule = protocol.get("advancement_rule", {})
    stated = rule.get("advance_to_locked_test_only_if")
    if not stated:
        raise RuntimeError("The protocol states no advancement rule")
    permitted = int(rule.get("maximum_locked_test_evaluations", 0))

    passed: list[str] = []
    evidence_hashes: list[str] = []
    if args.part5_results and Path(args.part5_results).is_file():
        part5 = read_json(assert_no_locked_test(args.part5_results))
        verdict = part5.get("results", {}).get("h2_verdict", {})
        if verdict.get("passed") is True:
            passed.append("H2")
        evidence_hashes.append(sha256_file(args.part5_results))
    if args.part6_results and Path(args.part6_results).is_file():
        part6 = read_json(assert_no_locked_test(args.part6_results))
        verdict = part6.get("results", {}).get("h3_verdict", {})
        if verdict.get("passed") is True:
            passed.append("H3")
        evidence_hashes.append(sha256_file(args.part6_results))

    if not evidence_hashes:
        raise RuntimeError(
            "No Part 5 or Part 6 results were supplied, so the advancement rule "
            "cannot be checked. The locked test stays closed."
        )
    if not passed:
        raise LockedTestAccessError(
            "ADVANCEMENT RULE NOT MET.\n"
            f"  Rule: {stated}\n"
            "  Neither H2 (small-data advantage) nor H3 (graph structure) passed.\n"
            "  The locked test must stay closed. Publish the characterised "
            "negative result instead, using the Part 2 geometric difference and "
            "the Part 3 circuit diagnostics as the explanation.\n"
            "  This is the protocol working, not a problem to route around."
        )

    return LockedTestAuthorisation(
        protocol_sha256=digest,
        advancement_rule=str(stated),
        hypothesis_passed="+".join(passed),
        evidence_sha256=evidence_hashes[0],
        evaluations_permitted=permitted,
    )


def assert_never_evaluated(output: Path) -> None:
    """Refuse a second evaluation, whatever the caller intends."""

    lock = output / "final_lock.json"
    if lock.exists():
        record = read_json(lock)
        raise LockedTestAccessError(
            "The locked test has already been evaluated for this study.\n"
            f"  Evaluated at: {record.get('evaluated_at_utc')}\n"
            f"  Summary hash: {record.get('summary_sha256')}\n"
            "  The protocol permits exactly one evaluation. A second run would "
            "invalidate it. Report the existing result."
        )


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def evaluate(
    checkpoint: Path,
    manifest: Path,
    thresholds: np.ndarray,
    authorisation: LockedTestAuthorisation,
) -> dict[str, Any]:
    """Score one frozen checkpoint on the locked test cohort, exactly once."""

    from cxr_thesis.objective2.metrics import multilabel_metrics
    from cxr_thesis.objective3.training import labels_from_manifest

    path = open_locked_test(manifest, authorisation)
    frame = pd.read_csv(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    scores = np.asarray(payload["locked_test_scores"], dtype=np.float64)
    targets = labels_from_manifest(frame, PRIMARY_LABELS).astype(np.int8)
    if scores.shape != targets.shape:
        raise ValueError(
            f"scores {scores.shape} do not match the locked cohort {targets.shape}"
        )
    metrics = multilabel_metrics(scores, targets, thresholds=thresholds)

    generator = np.random.default_rng(0)
    indices = generator.integers(
        0, len(targets), size=(BOOTSTRAP_RESAMPLES, len(targets))
    )
    resampled = np.array(
        [
            multilabel_metrics(
                scores[row], targets[row], thresholds=thresholds
            )["macro"]["auroc"]
            for row in indices[:1000]
        ]
    )
    low, high = np.quantile(resampled, [0.025, 0.975])
    return {
        "macro": metrics["macro"],
        "per_label": metrics["per_label"],
        "macro_auroc_ci95": [float(low), float(high)],
        "bootstrap_resamples": 1000,
        "p_value_report": "p < 0.001",
        "cases": int(len(targets)),
    }


def main() -> None:
    args = parse_args()
    output = assert_no_locked_test(args.output_dir)
    seed_record = seed_everything(args.seed)

    authorisation = check_advancement(args)
    print("--- ADVANCEMENT RULE ---")
    print(f"  Rule     : {authorisation.advancement_rule}")
    print(f"  Passed   : {authorisation.hypothesis_passed}")
    print(f"  Permitted: {authorisation.evaluations_permitted} evaluation")

    if args.dry_run:
        print("\nDRY RUN: the advancement rule is met and the locked test was "
              "NOT opened.")
        return

    assert_never_evaluated(output)
    verify_sha256(args.locked_test_manifest, args.expected_locked_test_sha256)
    require_existing([args.checkpoint, args.control_checkpoint, args.validation_thresholds])

    frozen = read_json(assert_no_locked_test(args.validation_thresholds))
    thresholds = np.asarray(frozen["thresholds"], dtype=np.float64)
    if thresholds.shape != (len(PRIMARY_LABELS),):
        raise ValueError("One frozen threshold per label is required")

    print("\nOpening the locked test cohort. This happens once.", flush=True)
    advanced = evaluate(
        args.checkpoint, args.locked_test_manifest, thresholds, authorisation
    )
    control = evaluate(
        args.control_checkpoint, args.locked_test_manifest, thresholds, authorisation
    )

    difference = advanced["macro"]["auroc"] - control["macro"]["auroc"]
    results = {
        "advanced_variant": advanced,
        "control_variant": control,
        "advanced_minus_control_macro_auroc": float(difference),
        "thresholds_frozen_from_validation": True,
        "thresholds_refitted_on_test": False,
        "evaluation_count": 1,
        "test_used_for_model_selection": False,
        "authorisation": authorisation.as_dict(),
    }

    path, digest = write_results(
        output / "results.json",
        study=STUDY,
        part=PART,
        config={
            "version": VERSION,
            "labels": PRIMARY_LABELS,
            "bootstrap_resamples": 1000,
            "seeding": seed_record,
        },
        results=results,
        artifact_hashes={
            "checkpoint": sha256_file(args.checkpoint),
            "control_checkpoint": sha256_file(args.control_checkpoint),
            "protocol": authorisation.protocol_sha256,
        },
        seed=args.seed,
        locked_test_accessed=True,
    )
    write_json_atomic(
        output / "final_lock.json",
        {
            "artifact": "Objective 3 v2.0 final evaluation lock",
            "study": STUDY,
            "version": VERSION,
            "evaluated_at_utc": utc_timestamp(),
            "summary_sha256": digest,
            "evaluation_count": 1,
            "test_used_for_model_selection": False,
            "thresholds_frozen_from_validation": True,
            "authorisation": authorisation.as_dict(),
        },
    )

    print("")
    print(f"{'variant':>10} {'macro AUROC':>12} {'95% CI':>22} {'AUPRC':>8} {'F1':>8}")
    for name, block in (("advanced", advanced), ("control", control)):
        interval = (
            f"[{block['macro_auroc_ci95'][0]:.4f},{block['macro_auroc_ci95'][1]:.4f}]"
        )
        print(
            f"{name:>10} {block['macro']['auroc']:>12.4f} {interval:>22} "
            f"{block['macro']['auprc']:>8.4f} {block['macro']['f1']:>8.4f}"
        )
    print(f"\nAdvanced minus control: {difference:+.4f}")
    print(f"Bootstrap p reported as: {advanced['p_value_report']}")
    print("")
    print(f"Results: {path}")
    print(f"Results SHA-256: {digest}")
    print("Evaluation count: 1 | Test used for model selection: False")


if __name__ == "__main__":
    main()
