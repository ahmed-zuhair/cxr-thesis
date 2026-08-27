#!/usr/bin/env python3
"""Publish one sanitized Objective 2 validation candidate to HF and GitHub."""

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

import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=("gcn", "gat", "densenet121"),
    )
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-title", required=True)
    parser.add_argument("--seed", type=int, default=42)
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def make_figure(history: pd.DataFrame, best_epoch: int, target: Path, model: str) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o")
    axes[0].set_title(f"{model.upper()} training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(history["epoch"], history["validation_macro_auroc"], marker="o", color="green")
    axes[1].axvline(best_epoch, linestyle="--", color="black", label=f"Best epoch {best_epoch}")
    axes[1].set_title("Validation macro AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[2].plot(history["epoch"], history["validation_macro_auprc"], marker="o", color="darkorange")
    axes[2].set_title("Validation macro AUPRC")
    axes[2].set_xlabel("Epoch")
    figure.suptitle(f"Objective 2: {model.upper()} validation training")
    figure.tight_layout()
    figure.savefig(target, dpi=180, bbox_inches="tight", metadata={"Software": "cxr-thesis"})
    plt.close(figure)


def deterministic_zip(files: list[tuple[Path, str]], target: Path) -> str:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for source, name in files:
            information = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = 0o100644 << 16
            bundle.writestr(information, source.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Public archive integrity test failed")
    return sha256_file(target)


def privacy_scan(result_root: Path) -> None:
    forbidden_extensions = {".pt", ".npz", ".dcm", ".nii", ".npy"}
    forbidden_columns = {"patient_id", "image_id", "image_path", "mask_path", "filename"}
    identifier_pattern = re.compile(r"\b\d{8}_\d{3}\.png\b|\bnih-\d{8}", re.IGNORECASE)
    violations: list[str] = []
    for path in result_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in forbidden_extensions:
            violations.append(f"forbidden binary: {path.name}")
        if path.suffix.lower() == ".csv":
            columns = set(pd.read_csv(path, nrows=0).columns)
            overlap = columns & forbidden_columns
            if overlap:
                violations.append(f"private CSV columns in {path.name}: {sorted(overlap)}")
        if path.suffix.lower() in {".json", ".md", ".csv", ".sha256"}:
            text = path.read_text(encoding="utf-8")
            if identifier_pattern.search(text):
                violations.append(f"case identifier pattern in {path.name}")
            if path.suffix.lower() == ".json":
                payload = json.loads(text)

                def keys(value) -> set[str]:
                    if isinstance(value, dict):
                        return set(map(str, value)) | set().union(*(keys(item) for item in value.values()))
                    if isinstance(value, list):
                        return set().union(*(keys(item) for item in value)) if value else set()
                    return set()

                overlap = keys(payload) & forbidden_columns
                if overlap:
                    violations.append(f"private JSON keys in {path.name}: {sorted(overlap)}")
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
    response = requests.request(method, url, headers=headers, timeout=120, **kwargs)
    return response


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")
    source_files = {
        name: args.training_output / name
        for name in ("best.pt", "best.sha256", "history.csv", "validation_summary.json")
    }
    missing = [name for name, path in source_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Training artifacts are missing: {missing}")
    expected_checkpoint_hash = source_files["best.sha256"].read_text(encoding="utf-8").split()[0]
    actual_checkpoint_hash = sha256_file(source_files["best.pt"])
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise RuntimeError("Checkpoint SHA-256 does not match")
    checkpoint = torch.load(source_files["best.pt"], map_location="cpu", weights_only=False)
    summary = json.loads(source_files["validation_summary.json"].read_text(encoding="utf-8"))
    history = pd.read_csv(source_files["history.csv"])
    if checkpoint.get("model_name") != args.model or summary.get("model") != args.model:
        raise RuntimeError("Model identity mismatch")
    if checkpoint.get("test_evaluated") is not False or summary.get("test_evaluated") is not False:
        raise RuntimeError("Candidate is not test-blind")
    if int(summary["test_cases_accessed"]) != 0:
        raise RuntimeError("Test cases were accessed")
    if int(summary["train_cases"]) != 30_000 or int(summary["validation_cases"]) != 5_000:
        raise RuntimeError("Unexpected cohort sizes")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.source_commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("The declared source commit is not in the current history")
    if run_git(["status", "--porcelain"]):
        raise RuntimeError("Repository must be clean before publication")

    result_root = REPOSITORY_ROOT / args.result_path
    if result_root.exists():
        raise FileExistsError(f"Public result directory already exists: {result_root}")
    result_root.mkdir(parents=True)
    figure_path = result_root / "training_history.png"
    make_figure(history, int(summary["best_epoch"]), figure_path, args.model)
    history.to_csv(result_root / "history.csv", index=False)
    shutil.copy2(source_files["best.sha256"], result_root / "best.sha256")
    architectures = {
        "gcn": "graph convolutional network",
        "gat": "graph attention network",
        "densenet121": "ImageNet-pretrained DenseNet-121",
    }
    architecture = architectures[args.model]
    public_summary = {
        "artifact": f"Objective 2 {args.model.upper()} validation-selected candidate",
        "model": args.model,
        "architecture": architecture,
        "parameters": int(summary["parameters"]),
        "training_cases": 30_000,
        "validation_cases": 5_000,
        "epochs_completed": len(history),
        "best_epoch": int(summary["best_epoch"]),
        "seed": args.seed,
        "labels": summary["labels"],
        "validation_thresholds": summary["validation_thresholds"],
        "validation_metrics": summary["validation_metrics"],
        "selection_criterion": "validation macro AUROC",
        "checkpoint_sha256": actual_checkpoint_hash,
        "training_figure_sha256": sha256_file(figure_path),
        "source_code_commit": args.source_commit,
        "locked_test_manifest_read": False,
        "locked_test_labels_accessed": False,
        "locked_test_evaluated": False,
        "test_used_for_model_selection": False,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "medical_images_included": False,
        "private_manifests_included": False,
        "status": "validation-selected candidate; final model comparison incomplete",
    }
    if args.model in {"gcn", "gat"}:
        public_summary["same_frozen_graph_inputs_for_gcn_and_gat"] = True
    atomic_json(public_summary, result_root / "validation_summary_public.json")
    readme = f"""# Objective 2 {args.model.upper()} validation candidate

- Architecture: {architecture}
- Training images: 30,000
- Validation images: 5,000
- Labels: 12
- Best epoch: {summary['best_epoch']}
- Validation macro AUROC: {summary['validation_metrics']['macro']['auroc']:.6f}
- Validation macro AUPRC: {summary['validation_metrics']['macro']['auprc']:.6f}
- Checkpoint SHA-256: `{actual_checkpoint_hash}`
- Source commit: `{args.source_commit}`

The checkpoint was selected using validation macro AUROC. The locked test cohort
was not read, evaluated, or used for threshold selection. This directory contains
no patient identifiers, image identifiers, medical images, private manifests, or
case-level predictions. The checkpoint is stored on Hugging Face and attached to
the corresponding GitHub Release.
"""
    (result_root / "README.md").write_text(readme, encoding="utf-8")
    inventory = {
        "artifact": f"Objective 2 {args.model.upper()} public artifact inventory",
        "checkpoint_sha256": actual_checkpoint_hash,
        "files": {},
        "private_data_included": False,
        "locked_test_results_included": False,
    }
    for path in sorted(result_root.iterdir()):
        if path.is_file():
            inventory["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(inventory, result_root / "artifact_inventory_public.json")
    privacy_scan(result_root)

    release_root = Path("/kaggle/working/backups")
    release_root.mkdir(parents=True, exist_ok=True)
    archive_name = f"objective2_{args.model}_validation_candidate_public_v1.0.0.zip"
    archive = release_root / archive_name
    archive_files = [(path, path.name) for path in sorted(result_root.iterdir()) if path.is_file()]
    archive_files.append((source_files["best.pt"], "best.pt"))
    archive_hash = deterministic_zip(archive_files, archive)
    archive_checksum = archive.with_suffix(".zip.sha256")
    archive_checksum.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")

    from huggingface_hub import CommitOperationAdd, HfApi

    hf_api = HfApi(token=hf_token)
    hf_info = hf_api.model_info(args.hf_repo, token=hf_token)
    if bool(hf_info.private):
        raise RuntimeError("Public checkpoint repository is unexpectedly private")
    hf_files = [(path, path.name) for path in sorted(result_root.iterdir()) if path.is_file()]
    hf_files.extend(
        [
            (source_files["best.pt"], "best.pt"),
            (archive, archive.name),
            (archive_checksum, archive_checksum.name),
        ]
    )
    hf_api.create_commit(
        repo_id=args.hf_repo,
        repo_type="model",
        token=hf_token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.hf_path.strip('/')}/{name}",
                path_or_fileobj=str(path),
            )
            for path, name in hf_files
        ],
        commit_message=f"results: publish Objective 2 {args.model.upper()} validation candidate",
    )

    run_git(["config", "user.name", "Ahmed Zuhair"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    run_git(["add", "--", str(args.result_path).replace("\\", "/")])
    staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
    allowed_prefix = str(args.result_path).replace("\\", "/").rstrip("/") + "/"
    if not staged or any(not item.startswith(allowed_prefix) for item in staged):
        raise RuntimeError(f"Unexpected staged files: {staged}")
    run_git(["commit", "-m", f"results: publish Objective 2 {args.model.upper()} validation candidate"])
    with tempfile.TemporaryDirectory(prefix="git_askpass_") as directory:
        askpass = Path(directory) / "askpass.sh"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in *Username*) echo "x-access-token" ;; *) echo "$GITHUB_TOKEN" ;; esac\n',
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        environment = dict(os.environ)
        environment["GIT_ASKPASS"] = str(askpass)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        run_git(["push", "origin", "main"], environment=environment)
    github_commit = run_git(["rev-parse", "HEAD"])

    release_url = f"https://api.github.com/repos/{args.github_repo}/releases/tags/{args.release_tag}"
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
                "body": "Validation-selected, test-blind Objective 2 candidate.",
                "draft": False,
                "prerelease": False,
            },
        )
    response.raise_for_status()
    release = response.json()
    existing_assets = {item["name"] for item in release.get("assets", [])}
    assets = [archive, archive_checksum, source_files["best.pt"], source_files["best.sha256"]]
    upload_base = release["upload_url"].split("{")[0]
    for asset in assets:
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
                "model": args.model,
                "best_epoch": int(summary["best_epoch"]),
                "validation_macro_auroc": summary["validation_metrics"]["macro"]["auroc"],
                "validation_macro_auprc": summary["validation_metrics"]["macro"]["auprc"],
                "checkpoint_sha256": actual_checkpoint_hash,
                "public_archive_sha256": archive_hash,
                "hf_path": f"https://huggingface.co/{args.hf_repo}/tree/main/{args.hf_path.strip('/')}",
                "github_commit": github_commit,
                "github_results": f"https://github.com/{args.github_repo}/tree/main/{str(args.result_path).replace(chr(92), '/')}",
                "github_release": release["html_url"],
                "locked_test_evaluated": False,
                "privacy_scan_passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"OBJECTIVE 2 {args.model.upper()} VALIDATION CANDIDATE PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
