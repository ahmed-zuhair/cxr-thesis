#!/usr/bin/env python3
"""Lock the Objective 3 v2.0 preregistration before any v2.0 training.

v1.0 and v1.1 remain published and final. This is a new study, motivated by the
v1.1 null, not a retrospective amendment to it.

The script refuses to run until the sizing pilot has been completed, because the
seed count and equivalence margin are derived from the pilot's observed
variability rather than guessed. The pilot seeds are recorded here and then
discarded: the study itself re-runs with a disjoint seed list, so the pilot
cannot bias the test it sizes.
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
from cxr_thesis.objective3_v2.guards import assert_no_locked_test, require_existing
from cxr_thesis.objective3_v2.io_utils import (
    read_json,
    sha256_file,
    utc_timestamp,
    verify_sha256,
    write_json_atomic,
)
from cxr_thesis.objective3_v2.stats import min_detectable_effect, required_pairs

CANDIDATE_MARGINS = (0.005, 0.010)
# A pilot standard deviation is itself an estimate. Sizing on the point estimate
# under-powers the study roughly half the time, so size on a conservative upper
# confidence bound instead.
SD_UPPER_BOUND_CONFIDENCE = 0.80
# Statistical sufficiency is not the only constraint: a handful of seeds invites
# the objection that the result is a fluke, however the arithmetic works out.
MINIMUM_SEEDS = 10
PILOT_SEED_BASE = 900_042
PROTECTED_FALSE_FIELDS = (
    "test_evaluated",
    "test_manifest_opened",
    "test_labels_accessed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v11-protocol", type=Path, required=True)
    parser.add_argument("--expected-v11-protocol-sha256", required=True)
    parser.add_argument("--v11-summary", type=Path, required=True)
    parser.add_argument("--expected-v11-summary-sha256", required=True)
    parser.add_argument(
        "--pilot-results",
        type=Path,
        required=True,
        help="Results JSON from the Part 4 Job 0 sizing pilot",
    )
    parser.add_argument(
        "--pilot-seeds",
        type=int,
        default=10,
        help="Number of paired seeds the pilot ran",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        required=True,
        help="Compute budget: the largest seed count you can afford per variant",
    )
    parser.add_argument("--expected-gat-sha256", required=True)
    parser.add_argument("--train-cohort-sha256", required=True)
    parser.add_argument("--val-cohort-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    return parser.parse_args()


def load_pilot_standard_deviation(path: Path) -> tuple[float, str]:
    """Read the per-seed delta SD the pilot measured, and hash the file."""

    payload = read_json(path)
    results = payload.get("results", payload)
    for key in (
        "delta_standard_deviation",
        "per_seed_delta_sd",
        "standard_deviation",
    ):
        if key in results:
            deviation = float(results[key])
            break
    else:
        raise KeyError(
            "The pilot results JSON must contain 'delta_standard_deviation' "
            f"under 'results'. Found keys: {sorted(results)}"
        )
    if not deviation > 0:
        raise ValueError("The pilot standard deviation must be positive")
    for field in PROTECTED_FALSE_FIELDS:
        if payload.get(field, False) or results.get(field, False):
            raise RuntimeError(f"The pilot is not test-blind: {field} is true")
    if payload.get("locked_test_accessed", False):
        raise RuntimeError("The pilot recorded locked_test_accessed = true")
    return deviation, sha256_file(path)


def inflate_standard_deviation(deviation: float, pilot_seeds: int) -> float:
    """Upper confidence bound on a standard deviation estimated from a pilot.

    Uses the chi-square interval for a variance: with ``df = n - 1``,
    ``sd_upper = sd * sqrt(df / chi2.ppf(1 - confidence, df))``.
    """

    from scipy import stats as scipy_stats

    degrees = pilot_seeds - 1
    if degrees < 1:
        raise ValueError("The pilot needs at least two seeds")
    critical = scipy_stats.chi2.ppf(1.0 - SD_UPPER_BOUND_CONFIDENCE, degrees)
    return float(deviation * np.sqrt(degrees / critical))


def choose_design(
    deviation: float,
    max_seeds: int,
    alpha: float,
    power: float,
) -> dict[str, object]:
    """Pick the tightest affordable equivalence margin and its seed count."""

    options = []
    for margin in CANDIDATE_MARGINS:
        needed = max(
            MINIMUM_SEEDS,
            required_pairs(margin, deviation, alpha=alpha, power=power),
        )
        options.append(
            {
                "margin": float(margin),
                "seeds_required": int(needed),
                "affordable": bool(needed <= max_seeds),
            }
        )
    affordable = [option for option in options if option["affordable"]]
    if not affordable:
        cheapest = min(options, key=lambda option: option["seeds_required"])
        raise RuntimeError(
            "No candidate margin fits the compute budget.\n"
            f"  observed per-seed SD : {deviation:.6f}\n"
            f"  budget               : {max_seeds} seeds per variant\n"
            f"  cheapest option      : margin {cheapest['margin']} needs "
            f"{cheapest['seeds_required']} seeds\n"
            "Either raise --max-seeds, or reduce the variance (more epochs, "
            "averaged restarts), or widen CANDIDATE_MARGINS deliberately and "
            "record why in the protocol."
        )
    chosen = min(affordable, key=lambda option: option["margin"])
    return {
        "chosen_margin": chosen["margin"],
        "chosen_seeds": chosen["seeds_required"],
        "candidates": options,
    }


def build_protocol(args: argparse.Namespace) -> dict[str, object]:
    require_existing(
        [args.v11_protocol, args.v11_summary, args.pilot_results]
    )
    verify_sha256(args.v11_protocol, args.expected_v11_protocol_sha256)
    verify_sha256(args.v11_summary, args.expected_v11_summary_sha256)

    v11_summary = read_json(args.v11_summary)
    for field in PROTECTED_FALSE_FIELDS:
        if v11_summary.get(field) is not False:
            raise RuntimeError(f"The v1.1 summary is not test-blind: {field}")
    if v11_summary.get("advance_to_single_final_evaluation") is not False:
        raise RuntimeError(
            "v1.1 must record that it did not advance to final evaluation"
        )

    deviation, pilot_hash = load_pilot_standard_deviation(args.pilot_results)
    sizing_deviation = inflate_standard_deviation(deviation, args.pilot_seeds)
    design = choose_design(
        sizing_deviation, args.max_seeds, args.alpha, args.power
    )
    seed_count = int(design["chosen_seeds"])
    margin = float(design["chosen_margin"])
    study_seeds = [42 + 1000 * index for index in range(seed_count)]
    pilot_seeds = [PILOT_SEED_BASE + index for index in range(args.pilot_seeds)]
    if set(study_seeds) & set(pilot_seeds):
        raise RuntimeError("Pilot and study seed lists must be disjoint")

    return {
        "artifact": "Objective 3 v2.0 preregistration",
        "study": STUDY,
        "version": VERSION,
        "objective": 3,
        "locked_at_utc": utc_timestamp(),
        "status": "locked before any v2.0 training and before final evaluation",
        "supersedes_v1_1": False,
        "relationship_to_v1_1": (
            "v1.0 and v1.1 remain published and final. v2.0 is a new study "
            "motivated by the v1.1 null result, not a retrospective amendment."
        ),
        "motivation": (
            "In v1.0 and v1.1 the quantum layer acted on a 160-dimensional vector "
            "produced by a frozen classical GAT, so graph structure was already "
            "collapsed before the circuit saw anything. v1.1 also tested a 0.005 "
            "threshold using three seeds and a seed-win count, a criterion whose "
            "minimum achievable p-value is 0.25."
        ),
        "predecessor_artifacts": {
            "v1_1_protocol_sha256": args.expected_v11_protocol_sha256,
            "v1_1_summary_sha256": args.expected_v11_summary_sha256,
            "v1_1_quantum_mean_macro_auroc": v11_summary.get(
                "quantum_mean_validation_macro_auroc"
            ),
            "v1_1_classical_mean_macro_auroc": v11_summary.get(
                "classical_mean_validation_macro_auroc"
            ),
            "v1_1_seeds": v11_summary.get("seeds"),
        },
        "frozen_inputs": {
            "frozen_gat_sha256": args.expected_gat_sha256,
            "train_cohort_sha256": args.train_cohort_sha256,
            "validation_cohort_sha256": args.val_cohort_sha256,
            "embedding_dimension": 160,
            "labels": [
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
            ],
            "label_count": 12,
            "pennylane_version": "0.45.1",
        },
        "sizing_pilot": {
            "purpose": "size the study only; never used as evidence for any hypothesis",
            "pilot_results_sha256": pilot_hash,
            "pilot_seeds": pilot_seeds,
            "pilot_seed_count": int(args.pilot_seeds),
            "observed_per_seed_delta_sd": float(deviation),
            "sizing_sd_upper_confidence_bound": float(sizing_deviation),
            "sd_upper_bound_confidence": SD_UPPER_BOUND_CONFIDENCE,
            "sizing_uses_upper_bound_not_point_estimate": True,
            "pilot_seeds_discarded_before_study": True,
            "pilot_disjoint_from_study_seeds": True,
        },
        "design": {
            "alpha": float(args.alpha),
            "power": float(args.power),
            "equivalence_margin": margin,
            "seeds_per_configuration": seed_count,
            "seed_list": study_seeds,
            "compute_budget_max_seeds": int(args.max_seeds),
            "margin_candidates_considered": design["candidates"],
            "minimum_seeds_floor": MINIMUM_SEEDS,
            "minimum_detectable_effect_at_chosen_n": min_detectable_effect(
                sizing_deviation, seed_count, args.alpha, args.power
            ),
            "minimum_detectable_effect_at_v1_1_n3": min_detectable_effect(
                sizing_deviation, 3, args.alpha, args.power
            ),
        },
        "hypotheses": {
            "H1_theory": {
                "statement": (
                    "The geometric difference g(K_C || K_Q) between the best "
                    "classical kernel and the quantum kernel on the frozen "
                    "embeddings is small relative to sqrt(N)."
                ),
                "prediction": (
                    "If g_CQ / sqrt(N) < 0.5 then no quantum advantage is "
                    "achievable in the large-sample regime, and the v1.1 null is "
                    "explained rather than merely restated."
                ),
                "requires_training": False,
                "reference": "Huang et al., Nat. Commun. 12:2631 (2021)",
            },
            "H2_sample_efficiency": {
                "statement": (
                    "In the small-data regime (n_train <= 1000) the quantum model "
                    "outperforms the parameter-matched classical model by at least "
                    "0.005 macro AUROC."
                ),
                "test": "paired Wilcoxon signed-rank",
                "threshold_auroc": 0.005,
                "seeds": 10,
                "alpha": float(args.alpha),
            },
            "H3_graph_structure": {
                "statement": (
                    "A circuit whose entangling gates follow the true graph "
                    "adjacency outperforms the same circuit entangled over all "
                    "node pairs by at least 0.005 macro AUROC."
                ),
                "test": "paired Wilcoxon signed-rank",
                "threshold_auroc": 0.005,
                "seeds": 10,
                "alpha": float(args.alpha),
            },
            "H4_equivalence": {
                "statement": (
                    "At full training size, quantum and classical are "
                    f"statistically equivalent within +/- {margin} macro AUROC."
                ),
                "test": "two one-sided tests (TOST)",
                "margin": margin,
                "seeds": seed_count,
                "alpha": float(args.alpha),
                "note": (
                    "Failing to reject a null of no difference is not evidence of "
                    "equivalence. TOST is required for the positive claim."
                ),
            },
        },
        "comparison_set": {
            "parameter_matched": True,
            "variants": [
                "graph_quantum",
                "complete_quantum",
                "no_entangle",
                "classical_gnn",
                "random_fixed",
            ],
            "key_ablation": (
                "graph_quantum vs complete_quantum isolates whether the graph "
                "topology carries information the circuit can use."
            ),
            "random_fixed_control": (
                "circuit parameters frozen at random initialisation, classical "
                "readout trained, to separate the quantum feature map from "
                "training the circuit."
            ),
        },
        "advancement_rule": {
            "advance_to_locked_test_only_if": "H2 passes OR H3 passes",
            "maximum_locked_test_evaluations": 1,
            "variants_evaluated_on_locked_test": 2,
            "thresholds_and_temperatures_frozen_from_validation": True,
            "if_neither_passes": (
                "publish a characterised negative result with H1 and H4 as the "
                "explanation, and do not open the locked test"
            ),
        },
        "stopping_rule": {
            "seeds_per_configuration": seed_count,
            "no_seeds_added_after_seeing_results": True,
            "no_architecture_tuning_after_this_lock": True,
            "seed_list_fixed_at_lock_time": True,
        },
        "test_blindness": {
            "test_evaluated": False,
            "test_manifest_opened": False,
            "test_labels_accessed": False,
            "new_test_cohort_selected": False,
        },
        "publication": {
            "allowed_for_publication": True,
            "patient_identifiers_included": False,
            "image_identifiers_included": False,
            "private_checkpoints_included": False,
            "case_level_outputs_included": False,
        },
    }


def render_markdown(payload: dict[str, object]) -> str:
    design = payload["design"]
    pilot = payload["sizing_pilot"]
    lines = [
        "# Objective 3 v2.0 — Preregistration",
        "",
        f"**Locked:** {payload['locked_at_utc']}  ",
        f"**Study:** `{payload['study']}` **Version:** {payload['version']}  ",
        f"**Status:** {payload['status']}",
        "",
        "## Relationship to v1.1",
        "",
        str(payload["relationship_to_v1_1"]),
        "",
        "## Motivation",
        "",
        str(payload["motivation"]),
        "",
        "## Sizing pilot",
        "",
        f"- Pilot seeds: {pilot['pilot_seed_count']} "
        f"(seeds {pilot['pilot_seeds'][0]}–{pilot['pilot_seeds'][-1]}, discarded)",
        f"- Observed per-seed delta SD: **{pilot['observed_per_seed_delta_sd']:.6f}**",
        f"- Sizing SD ({int(pilot['sd_upper_bound_confidence'] * 100)}% upper bound): "
        f"**{pilot['sizing_sd_upper_confidence_bound']:.6f}** — the study is sized on "
        "this, not the point estimate, because a pilot SD is itself uncertain",
        f"- Pilot results SHA-256: `{pilot['pilot_results_sha256']}`",
        "- The pilot sizes the study only. Its seeds are disjoint from the study "
        "seed list and are not reused, so it cannot bias the test.",
        "",
        "## Design",
        "",
        f"- Equivalence margin: **±{design['equivalence_margin']}** macro AUROC",
        f"- Seeds per configuration: **{design['seeds_per_configuration']}**",
        f"- Alpha {design['alpha']}, power {design['power']}",
        f"- Minimum detectable effect at n={design['seeds_per_configuration']}: "
        f"**{design['minimum_detectable_effect_at_chosen_n']:.6f}**",
        f"- For comparison, at v1.1's n=3 it would have been "
        f"**{design['minimum_detectable_effect_at_v1_1_n3']:.6f}**",
        "",
        "| Candidate margin | Seeds required | Affordable |",
        "|---|---:|---|",
    ]
    for option in design["margin_candidates_considered"]:
        lines.append(
            f"| ±{option['margin']} | {option['seeds_required']} | "
            f"{'yes' if option['affordable'] else 'no'} |"
        )
    lines += ["", "## Hypotheses", ""]
    for name, body in payload["hypotheses"].items():
        lines += [f"### {name}", "", str(body["statement"]), ""]
        for key in ("prediction", "test", "margin", "threshold_auroc", "seeds", "note"):
            if key in body:
                lines.append(f"- **{key}**: {body[key]}")
        lines.append("")
    rule = payload["advancement_rule"]
    lines += [
        "## Advancement rule",
        "",
        f"- Advance to the locked test only if: **{rule['advance_to_locked_test_only_if']}**",
        f"- Maximum locked-test evaluations: **{rule['maximum_locked_test_evaluations']}**",
        f"- If neither passes: {rule['if_neither_passes']}",
        "",
        "## Stopping rule",
        "",
        f"- {design['seeds_per_configuration']} seeds per configuration, fixed now",
        "- No seeds added after seeing results",
        "- No architecture tuning after this lock",
        "",
        "## Seed list (fixed at lock time)",
        "",
        "```",
        ", ".join(str(seed) for seed in design["seed_list"]),
        "```",
        "",
        "## Test blindness at lock time",
        "",
    ]
    for key, value in payload["test_blindness"].items():
        lines.append(f"- {key}: **{value}**")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = assert_no_locked_test(args.output_dir)
    if output.exists():
        raise FileExistsError(
            f"{output} already exists. A locked protocol must never be overwritten."
        )

    payload = build_protocol(args)
    output.mkdir(parents=True)
    protocol_path = write_json_atomic(output / "protocol.json", payload)
    markdown_path = output / "protocol.md"
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    protocol_hash = sha256_file(protocol_path)
    markdown_hash = sha256_file(markdown_path)
    lock_path = write_json_atomic(
        output / "final_lock.json",
        {
            "artifact": "Objective 3 v2.0 preregistration lock",
            "study": STUDY,
            "version": VERSION,
            "locked_at_utc": utc_timestamp(),
            "protocol_json_sha256": protocol_hash,
            "protocol_md_sha256": markdown_hash,
            "pilot_results_sha256": payload["sizing_pilot"]["pilot_results_sha256"],
            "v1_1_protocol_sha256": args.expected_v11_protocol_sha256,
            "v1_1_summary_sha256": args.expected_v11_summary_sha256,
            "frozen_gat_sha256": args.expected_gat_sha256,
            "training_started": False,
            "test_evaluated": False,
            "test_manifest_opened": False,
            "test_labels_accessed": False,
        },
    )
    for path in (protocol_path, markdown_path, lock_path):
        (path.with_name(path.name + ".sha256")).write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
        )

    design = payload["design"]
    print("--- OBJECTIVE 3 v2.0 PROTOCOL LOCKED ---")
    print(f"Pilot per-seed SD        : {payload['sizing_pilot']['observed_per_seed_delta_sd']:.6f}")
    print(f"Sizing SD (upper bound)  : {payload['sizing_pilot']['sizing_sd_upper_confidence_bound']:.6f}")
    print(f"Equivalence margin       : +/-{design['equivalence_margin']}")
    print(f"Seeds per configuration  : {design['seeds_per_configuration']}")
    print(f"MDE at chosen n          : {design['minimum_detectable_effect_at_chosen_n']:.6f}")
    print(f"MDE had n been 3 (v1.1)  : {design['minimum_detectable_effect_at_v1_1_n3']:.6f}")
    print(f"protocol.json SHA-256    : {protocol_hash}")
    print(f"protocol.md   SHA-256    : {markdown_hash}")
    print(f"final_lock.json SHA-256  : {sha256_file(lock_path)}")
    print("Training started         : False")
    print("Test evaluated           : False")
    print("PROTOCOL LOCK SUCCESSFUL")


if __name__ == "__main__":
    main()
