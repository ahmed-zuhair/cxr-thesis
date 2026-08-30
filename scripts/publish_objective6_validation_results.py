#!/usr/bin/env python3
"""Publish privacy-safe Objective 6 aggregate validation results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "81424c30f1619707325f0a83ef9a6fba3a859743e3b4ee0c33ac68dba6161438"
LOCK_SHA256 = "e48b11cc0af8be0866b873ae91dd5f4c55738b39927d6dec52d2f29cf5f8275a"
SYSTEMS = {
    "nearest_training_image_retrieval", "image_only", "multimodal"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-figure-sha256", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-title", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def deterministic_zip(files: list[Path], target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for source in sorted(files, key=lambda item: item.name):
            information = zipfile.ZipInfo(source.name, (1980, 1, 1, 0, 0, 0))
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = 0o100644 << 16
            archive.writestr(information, source.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Objective 6 validation-results archive is corrupt")
    return sha256(target)


def run_git(arguments: list[str], environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, env=environment,
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def github_request(method: str, url: str, token: str, **kwargs: Any) -> Any:
    import requests

    headers = dict(kwargs.pop("headers", {}))
    headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return requests.request(method, url, headers=headers, timeout=120, **kwargs)


def validate(summary: dict[str, Any], final_lock: dict[str, Any]) -> None:
    if summary.get("artifact") != (
        "Objective 6 locked validation report-generation comparison"
    ):
        raise RuntimeError("Unexpected Objective 6 validation summary")
    if (
        summary.get("validation_cases") != 6280
        or summary.get("primary_system") != "multimodal"
        or summary.get("primary_system_changed_after_validation") is not False
        or summary.get("locked_test_manifest_opened") is not False
        or summary.get("locked_test_reports_accessed") is not False
        or summary.get("locked_test_evaluated") is not False
    ):
        raise RuntimeError("Objective 6 validation safety state changed")
    if set(summary.get("systems", {})) != SYSTEMS:
        raise RuntimeError("Objective 6 validation comparator set changed")
    for system, metrics in summary["systems"].items():
        required = {
            "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L",
            "METEOR_exact_token", "CIDEr-D", "micro_concept_precision",
            "micro_concept_recall", "micro_concept_f1", "macro_concept_f1",
            "explicit_negation_contradiction_rate", "empty_report_rate",
            "repeated_4gram_report_rate", "training_report_exact_match_rate",
            "unique_generated_report_fraction",
        }
        if not required.issubset(metrics):
            raise RuntimeError(f"Missing preregistered metrics for {system}")
    comparisons = summary.get("paired_primary_comparisons", {})
    if set(comparisons) != {
        "multimodal_minus_image_only",
        "multimodal_minus_nearest_training_image_retrieval",
    }:
        raise RuntimeError("Objective 6 paired comparisons changed")
    if final_lock.get("artifact") != "Final Objective 6 validation comparison lock":
        raise RuntimeError("Unexpected Objective 6 validation result lock")
    if (
        final_lock.get("immutable") is not True
        or final_lock.get("validation_protocol_sha256") != PROTOCOL_SHA256
        or final_lock.get("validation_lock_sha256") != LOCK_SHA256
        or final_lock.get("primary_system") != "multimodal"
        or final_lock.get("primary_system_changed") is not False
        or final_lock.get("validation_evaluation_count") != 1
        or final_lock.get("locked_test_evaluated") is not False
        or final_lock.get("locked_test_evaluation_count") != 0
    ):
        raise RuntimeError("Objective 6 validation result lock changed")


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")

    public = args.evaluation_output / "public"
    summary_source = public / "objective6_validation_comparison_summary_public.json"
    summary_checksum = summary_source.with_suffix(".json.sha256")
    figure_source = public / "objective6_validation_comparison.png"
    lock_source = public / "FINAL_OBJECTIVE6_VALIDATION_COMPARISON.json"
    lock_checksum = lock_source.with_suffix(".json.sha256")
    for path in (
        summary_source, summary_checksum, figure_source, lock_source, lock_checksum
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual = {
        "summary": sha256(summary_source),
        "figure": sha256(figure_source),
        "lock": sha256(lock_source),
    }
    expected = {
        "summary": args.expected_summary_sha256,
        "figure": args.expected_figure_sha256,
        "lock": args.expected_lock_sha256,
    }
    if actual != expected:
        raise RuntimeError(f"Objective 6 public artifact hashes changed: {actual}")
    if summary_checksum.read_text(encoding="utf-8").split()[0] != actual["summary"]:
        raise RuntimeError("Objective 6 summary checksum mismatch")
    if lock_checksum.read_text(encoding="utf-8").split()[0] != actual["lock"]:
        raise RuntimeError("Objective 6 final-lock checksum mismatch")
    summary = json.loads(summary_source.read_text(encoding="utf-8"))
    final_lock = json.loads(lock_source.read_text(encoding="utf-8"))
    validate(summary, final_lock)
    if (
        final_lock.get("summary_sha256") != actual["summary"]
        or final_lock.get("figure_sha256") != actual["figure"]
    ):
        raise RuntimeError("Objective 6 final lock does not identify public outputs")

    result_root = ROOT / args.result_path
    source_files = (
        summary_source, summary_checksum, figure_source, lock_source, lock_checksum
    )
    expected_names = {
        *(path.name for path in source_files),
        "objective6_validation_comparison.png.sha256",
        "README.md", "artifact_inventory_public.json",
    }
    if result_root.exists():
        existing = {path.name for path in result_root.iterdir() if path.is_file()}
        if existing != expected_names:
            raise RuntimeError("Partial Objective 6 validation-result publication exists")
    else:
        result_root.mkdir(parents=True)
        for source in source_files:
            shutil.copy2(source, result_root / source.name)
        (result_root / "objective6_validation_comparison.png.sha256").write_text(
            f"{actual['figure']}  objective6_validation_comparison.png\n",
            encoding="utf-8",
        )
        delta = summary["paired_primary_comparisons"][
            "multimodal_minus_image_only"
        ]
        (result_root / "README.md").write_text(
            "# Objective 6 locked validation comparison\n\n"
            "This privacy-safe package reports the single preregistered validation "
            "comparison of nearest-image retrieval, image-only generation, and "
            "multimodal generation. The multimodal system remained the primary "
            "system fixed before generation. It did not exceed image-only generation "
            "on the primary PadChest-6 macro concept F1 endpoint; all lexical, "
            "clinical, contradiction, and safety metrics are reported without "
            "post-result model switching.\n\n"
            f"Multimodal minus image-only macro concept F1: "
            f"{delta['macro_concept_f1']['point_difference']:.9f}.\n\n"
            "No reports, case-level outputs, identifiers, images, private manifests, "
            "private checkpoints, or locked-test results are included.\n",
            encoding="utf-8",
        )
        inventory: dict[str, Any] = {
            "artifact": "Objective 6 validation-results public inventory",
            "summary_sha256": actual["summary"],
            "figure_sha256": actual["figure"],
            "final_lock_sha256": actual["lock"],
            "raw_or_generated_reports_included": False,
            "case_level_outputs_included": False,
            "patient_or_image_identifiers_included": False,
            "medical_images_included": False,
            "private_manifests_included": False,
            "private_checkpoints_included": False,
            "locked_test_results_included": False,
            "files": {},
        }
        for path in sorted(result_root.iterdir()):
            inventory["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        atomic_json(inventory, result_root / "artifact_inventory_public.json")

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result_root.iterdir()
        if path.suffix.lower() in {".json", ".md"}
    )
    forbidden = (
        '"patient_id"', '"image_id"', '"image_path"', '"case_code"',
        '"reference_report"', '"generated_report"',
    )
    violations = [value for value in forbidden if value in serialized]
    if violations:
        raise RuntimeError(f"Objective 6 public privacy scan failed: {violations}")

    archive = Path(
        "/kaggle/working/backups/"
        "objective6_validation_comparison_public_v1.0.0.zip"
    )
    archive_hash = deterministic_zip(list(result_root.iterdir()), archive)
    archive_checksum = archive.with_suffix(".zip.sha256")
    archive_checksum.write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8"
    )

    from huggingface_hub import CommitOperationAdd, HfApi

    hf_api = HfApi(token=hf_token)
    if bool(hf_api.model_info(args.hf_repo, token=hf_token).private):
        raise RuntimeError("Public checkpoint repository is unexpectedly private")
    hf_files = list(result_root.iterdir()) + [archive, archive_checksum]
    hf_commit = hf_api.create_commit(
        repo_id=args.hf_repo, repo_type="model", token=hf_token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.hf_path.strip('/')}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in hf_files
        ],
        commit_message="results: publish Objective 6 locked validation comparison",
    )

    run_git(["config", "user.name", "Ahmed Zuhair Sabah"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    result_prefix = str(args.result_path).replace("\\", "/").rstrip("/")
    status = run_git(["status", "--porcelain"])
    unexpected = []
    for line in status.splitlines():
        changed = line[3:].strip().split(" -> ")[-1].rstrip("/")
        inside = changed == result_prefix or changed.startswith(f"{result_prefix}/")
        parent = result_prefix.startswith(f"{changed}/")
        if not inside and not parent:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(f"Unexpected Git changes: {unexpected}")
    if status:
        run_git(["add", "--", result_prefix])
        staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
        if not staged or any(not path.startswith(f"{result_prefix}/") for path in staged):
            raise RuntimeError(f"Unexpected staged files: {staged}")
        run_git(["commit", "-m", "results: publish Objective 6 validation comparison"])
    with tempfile.TemporaryDirectory(prefix="git_askpass_") as directory:
        askpass = Path(directory) / "askpass.sh"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in *Username*) echo "x-access-token" ;; '
            '*) echo "$GITHUB_TOKEN" ;; esac\n', encoding="utf-8",
        )
        askpass.chmod(0o700)
        environment = dict(os.environ)
        environment["GIT_ASKPASS"] = str(askpass)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        run_git(["push", "origin", "main"], environment)
    github_commit = run_git(["rev-parse", "HEAD"])

    release_endpoint = (
        f"https://api.github.com/repos/{args.github_repo}/releases/tags/"
        f"{args.release_tag}"
    )
    response = github_request("GET", release_endpoint, github_token)
    if response.status_code == 404:
        response = github_request(
            "POST", f"https://api.github.com/repos/{args.github_repo}/releases",
            github_token,
            json={
                "tag_name": args.release_tag,
                "target_commitish": github_commit,
                "name": args.release_title,
                "body": (
                    "Privacy-safe aggregate results from the single locked Objective 6 "
                    "validation comparison. The locked test remained unopened."
                ),
                "draft": False, "prerelease": False,
            },
        )
    response.raise_for_status()
    release = response.json()
    existing_assets = {asset["name"] for asset in release.get("assets", [])}
    upload_url = release["upload_url"].split("{")[0]
    for asset in (archive, archive_checksum):
        if asset.name in existing_assets:
            continue
        with asset.open("rb") as stream:
            uploaded = github_request(
                "POST", upload_url, github_token,
                params={"name": asset.name},
                headers={"Content-Type": "application/octet-stream"}, data=stream,
            )
        uploaded.raise_for_status()

    print(json.dumps({
        "summary_sha256": actual["summary"],
        "figure_sha256": actual["figure"],
        "final_lock_sha256": actual["lock"],
        "public_archive_sha256": archive_hash,
        "hf_commit": getattr(hf_commit, "oid", None),
        "hf_path": (
            f"https://huggingface.co/{args.hf_repo}/tree/main/"
            f"{args.hf_path.strip('/')}"
        ),
        "github_commit": github_commit,
        "github_results": (
            f"https://github.com/{args.github_repo}/tree/main/{result_prefix}"
        ),
        "github_release": release["html_url"],
        "primary_system": "multimodal",
        "primary_system_changed_after_validation": False,
        "locked_test_evaluated": False,
        "privacy_scan_passed": True,
    }, indent=2, sort_keys=True))
    print("OBJECTIVE 6 LOCKED VALIDATION RESULTS PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
