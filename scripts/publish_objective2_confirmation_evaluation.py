#!/usr/bin/env python3
"""Publish sanitized aggregate Objective 2 independent-confirmation results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("cnn", "densenet121")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-final-lock-sha256", required=True)
    parser.add_argument("--expected-confirmation-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-title", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_git(arguments: list[str], *, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def deterministic_zip(files: list[Path], target: Path) -> str:
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as bundle:
        for source in sorted(files, key=lambda item: item.name):
            information = zipfile.ZipInfo(
                source.name,
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = 0o100644 << 16
            bundle.writestr(information, source.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Confirmation public archive integrity failed")
    return sha256_file(target)


def privacy_scan(root: Path) -> None:
    forbidden_extensions = {
        ".pt",
        ".npz",
        ".npy",
        ".csv",
        ".dcm",
        ".nii",
        ".nii.gz",
    }
    identifier_pattern = re.compile(r"\b\d{8}_\d{3}\.png\b|\bnih-\d{8}", re.IGNORECASE)
    forbidden_keys = {
        "patient_id",
        "image_id",
        "image_path",
        "mask_path",
        "filename",
        "probabilities",
        "targets",
    }
    violations: list[str] = []

    def keys(value) -> set[str]:
        if isinstance(value, dict):
            nested = set(map(str, value))
            for item in value.values():
                nested.update(keys(item))
            return nested
        if isinstance(value, list):
            nested: set[str] = set()
            for item in value:
                nested.update(keys(item))
            return nested
        return set()

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in forbidden_extensions:
            violations.append(f"forbidden binary/private table: {path.name}")
        if path.suffix.lower() in {".json", ".md", ".sha256"}:
            text = path.read_text(encoding="utf-8")
            if identifier_pattern.search(text):
                violations.append(f"identifier pattern: {path.name}")
            if path.suffix.lower() == ".json":
                overlap = keys(json.loads(text)) & forbidden_keys
                if overlap:
                    violations.append(
                        f"private JSON keys in {path.name}: {sorted(overlap)}"
                    )
    if violations:
        raise RuntimeError("Privacy scan failed:\n" + "\n".join(violations))


def github_request(method: str, url: str, token: str, **kwargs):
    import requests

    headers = dict(kwargs.pop("headers", {}))
    headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return requests.request(
        method,
        url,
        headers=headers,
        timeout=120,
        **kwargs,
    )


def copy_exact(source: Path, target: Path) -> None:
    if target.is_file():
        if target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Existing publication artifact differs: {target}")
        return
    shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")

    public_source = args.evaluation_output / "public"
    summary_source = public_source / "independent_confirmation_summary_public.json"
    figure_source = public_source / "independent_confirmation_comparison.png"
    final_lock_source = (
        args.evaluation_output / "FINAL_INDEPENDENT_CONFIRMATION_EVALUATION.json"
    )
    for path in (summary_source, figure_source, final_lock_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(summary_source) != args.expected_summary_sha256:
        raise RuntimeError("Confirmation summary SHA-256 does not match")
    if sha256_file(final_lock_source) != args.expected_final_lock_sha256:
        raise RuntimeError("Confirmation final-lock SHA-256 does not match")

    summary = json.loads(summary_source.read_text(encoding="utf-8"))
    lock = json.loads(final_lock_source.read_text(encoding="utf-8"))
    checks = {
        "summary_hash_in_lock": lock.get("summary_sha256")
        == args.expected_summary_sha256,
        "figure_hash_in_lock": lock.get("figure_sha256") == sha256_file(figure_source),
        "confirmation_hash": lock.get("confirmation_manifest_sha256")
        == args.expected_confirmation_sha256,
        "protocol_hash": lock.get("protocol_sha256") == args.expected_protocol_sha256,
        "models": lock.get("completed_models") == list(MODEL_ORDER),
        "thresholds_frozen": lock.get("validation_thresholds_reused_without_change")
        is True,
        "confirmation_not_selection": lock.get("confirmation_used_for_model_selection")
        is False,
        "confirmation_evaluated": lock.get("confirmation_evaluated") is True,
        "evaluation_count": lock.get("confirmation_evaluation_count") == 1,
        "summary_models": list(summary.get("models", {})) == list(MODEL_ORDER),
        "independent": summary.get("independent_confirmation_cohort") is True,
        "protocol_precedes_evaluation": summary.get(
            "protocol_published_before_confirmation_evaluation"
        )
        is True,
        "post_test_enhancement_disclosed": summary.get(
            "enhancement_developed_after_original_locked_test"
        )
        is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Confirmation publication checks failed: {checks}")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", args.source_commit, "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("Declared evaluator source commit is not in history")

    result_root = REPOSITORY_ROOT / args.result_path
    result_prefix = str(args.result_path).replace("\\", "/").rstrip("/") + "/"
    unexpected_changes = []
    for line in run_git(["status", "--porcelain"]).splitlines():
        changed_path = line[3:].replace("\\", "/")
        if changed_path.rstrip("/") == result_prefix.rstrip("/"):
            continue
        if not changed_path.startswith(result_prefix):
            unexpected_changes.append(line)
    if unexpected_changes:
        raise RuntimeError(f"Unexpected repository changes: {unexpected_changes}")

    result_root.mkdir(parents=True, exist_ok=True)
    copy_exact(summary_source, result_root / summary_source.name)
    copy_exact(figure_source, result_root / figure_source.name)
    public_lock = {
        key: lock[key]
        for key in (
            "artifact",
            "confirmation_manifest_sha256",
            "protocol_sha256",
            "checkpoint_sha256",
            "summary_sha256",
            "figure_sha256",
            "completed_models",
            "validation_thresholds_reused_without_change",
            "confirmation_used_for_model_selection",
            "confirmation_evaluated",
            "confirmation_evaluation_count",
        )
    }
    public_lock_path = result_root / "confirmation_evaluation_lock_public.json"
    atomic_json(public_lock, public_lock_path)

    cnn = summary["models"]["cnn"]["confirmation_metrics"]["macro"]
    dense = summary["models"]["densenet121"]["confirmation_metrics"]["macro"]
    difference = summary["bootstrap"]["paired_model_minus_reference"]["densenet121"]
    readme = f"""# Objective 2 Independent Confirmation

This is the pre-specified, one-time independent confirmation of the frozen
original CNN and enhanced DenseNet-121 on 5,000 images from 568 complete
patients who did not appear in the original locked-test cohort.

| Model | Macro AUROC | Macro AUPRC | Macro F1 |
|---|---:|---:|---:|
| Original CNN | {cnn["auroc"]:.6f} | {cnn["auprc"]:.6f} | {cnn["f1"]:.6f} |
| Enhanced DenseNet-121 | {dense["auroc"]:.6f} | {dense["auprc"]:.6f} | {dense["f1"]:.6f} |

Paired bootstrap DenseNet-minus-CNN mean differences were
{difference["auroc"]["model_minus_reference_mean"]:.6f} AUROC,
{difference["auprc"]["model_minus_reference_mean"]:.6f} AUPRC, and
{difference["f1"]["model_minus_reference_mean"]:.6f} macro F1. All corresponding
95% bootstrap confidence intervals excluded zero. The stored empirical
two-sided values were 0.0 with 1,000 resamples; conservatively, this is reported
as p < 0.002 rather than as an exact zero probability.

The enhancement was designed after the original locked-test comparison, so that
old test cohort was not reused as untouched evidence. This independent cohort
was selected label-blind, its protocol was publicly timestamped before
evaluation, validation thresholds were reused unchanged, and the cohort was not
used for model selection or threshold tuning.

No patient/image identifiers, medical images, private manifests, checkpoints,
or case-level predictions are included.
"""
    readme_path = result_root / "README.md"
    if readme_path.is_file() and readme_path.read_text(encoding="utf-8") != readme:
        raise RuntimeError("Existing confirmation README differs")
    readme_path.write_text(readme, encoding="utf-8")

    inventory_path = result_root / "artifact_inventory_public.json"
    inventory = {
        "artifact": "Objective 2 independent confirmation public inventory",
        "confirmation_manifest_sha256": args.expected_confirmation_sha256,
        "protocol_sha256": args.expected_protocol_sha256,
        "summary_sha256": args.expected_summary_sha256,
        "private_manifest_included": False,
        "case_level_predictions_included": False,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "medical_images_included": False,
        "checkpoints_included": False,
        "files": {},
    }
    for path in sorted(result_root.iterdir()):
        if path.is_file() and path.name != inventory_path.name:
            inventory["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    atomic_json(inventory, inventory_path)

    expected_files = {
        "README.md",
        "artifact_inventory_public.json",
        "confirmation_evaluation_lock_public.json",
        summary_source.name,
        figure_source.name,
    }
    actual_files = {path.name for path in result_root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"Unexpected public files: {sorted(actual_files - expected_files)}"
        )
    privacy_scan(result_root)

    backups = Path("/kaggle/working/backups")
    backups.mkdir(parents=True, exist_ok=True)
    archive = backups / "objective2_independent_confirmation_public_v1.0.0.zip"
    archive_hash = deterministic_zip(
        [path for path in result_root.iterdir() if path.is_file()], archive
    )
    checksum = archive.with_suffix(".zip.sha256")
    checksum.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    hf_api = HfApi(token=hf_token)
    if bool(hf_api.model_info(args.hf_repo, token=hf_token).private):
        raise RuntimeError("Public checkpoint repository is unexpectedly private")
    hf_files = [path for path in result_root.iterdir() if path.is_file()]
    hf_files.extend([archive, checksum])
    remote_files = set(
        hf_api.list_repo_files(args.hf_repo, repo_type="model", token=hf_token)
    )
    hf_cache = backups / "objective2_confirmation_hf_verification"
    hf_cache.mkdir(parents=True, exist_ok=True)
    operations = []
    for path in hf_files:
        remote_path = f"{args.hf_path.strip('/')}/{path.name}"
        if remote_path in remote_files:
            downloaded = Path(
                hf_hub_download(
                    args.hf_repo,
                    filename=remote_path,
                    repo_type="model",
                    token=hf_token,
                    local_dir=hf_cache,
                    force_download=True,
                )
            )
            if sha256_file(downloaded) != sha256_file(path):
                raise RuntimeError(f"Existing HF artifact differs: {remote_path}")
            continue
        operations.append(
            CommitOperationAdd(
                path_in_repo=remote_path,
                path_or_fileobj=str(path),
            )
        )
    if operations:
        hf_api.create_commit(
            repo_id=args.hf_repo,
            repo_type="model",
            token=hf_token,
            operations=operations,
            commit_message="results: publish independent Objective 2 confirmation",
        )

    run_git(["config", "user.name", "Ahmed Zuhair"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    run_git(["add", "--", str(args.result_path).replace("\\", "/")])
    staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
    if any(not path.startswith(result_prefix) for path in staged):
        raise RuntimeError(f"Unexpected staged files: {staged}")
    if staged:
        run_git(["commit", "-m", "results: publish independent confirmation"])
    with tempfile.TemporaryDirectory(prefix="git_askpass_") as directory:
        askpass = Path(directory) / "askpass.sh"
        askpass.write_text(
            "#!/bin/sh\n"
            'case "$1" in *Username*) echo "x-access-token" ;; '
            '*) echo "$GITHUB_TOKEN" ;; esac\n',
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        environment = dict(os.environ)
        environment["GIT_ASKPASS"] = str(askpass)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        run_git(["push", "origin", "main"], environment=environment)
    github_commit = run_git(["rev-parse", "HEAD"])

    release_url = (
        f"https://api.github.com/repos/{args.github_repo}/releases/tags/"
        f"{args.release_tag}"
    )
    response = github_request("GET", release_url, github_token)
    if response.status_code == 404:
        response = github_request(
            "POST",
            f"https://api.github.com/repos/{args.github_repo}/releases",
            github_token,
            json={
                "tag_name": args.release_tag,
                "target_commitish": github_commit,
                "name": args.release_title,
                "body": (
                    "Aggregate one-time independent confirmation of the frozen "
                    "original CNN and enhanced DenseNet-121."
                ),
                "draft": False,
                "prerelease": False,
            },
        )
    response.raise_for_status()
    release = response.json()
    existing_assets = {item["name"] for item in release.get("assets", [])}
    upload_base = release["upload_url"].split("{")[0]
    for asset in (archive, checksum):
        if asset.name in existing_assets:
            continue
        with asset.open("rb") as handle:
            uploaded = github_request(
                "POST",
                upload_base,
                github_token,
                params={"name": asset.name},
                headers={"Content-Type": "application/octet-stream"},
                data=handle,
            )
        uploaded.raise_for_status()

    print(
        json.dumps(
            {
                "cnn_confirmation_macro_auroc": cnn["auroc"],
                "densenet_confirmation_macro_auroc": dense["auroc"],
                "densenet_minus_cnn_auroc": difference["auroc"][
                    "model_minus_reference_mean"
                ],
                "summary_sha256": args.expected_summary_sha256,
                "final_lock_sha256": args.expected_final_lock_sha256,
                "archive_sha256": archive_hash,
                "hf_path": (
                    f"https://huggingface.co/{args.hf_repo}/tree/main/"
                    f"{args.hf_path.strip('/')}"
                ),
                "github_results": (
                    f"https://github.com/{args.github_repo}/tree/main/"
                    f"{str(args.result_path).replace(chr(92), '/')}"
                ),
                "github_commit": github_commit,
                "github_release": release["html_url"],
                "privacy_scan_passed": True,
                "private_manifest_published": False,
                "case_level_predictions_published": False,
                "confirmation_used_for_model_selection": False,
                "confirmation_evaluation_count": 1,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("OBJECTIVE 2 INDEPENDENT CONFIRMATION PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
