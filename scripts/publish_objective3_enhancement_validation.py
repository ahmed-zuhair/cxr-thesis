#!/usr/bin/env python3
"""Publish the sanitized negative Objective 3 v1.1 validation result."""

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

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-amendment-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_git(arguments: list[str], environment=None) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=REPOSITORY_ROOT, env=environment,
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def make_figure(summary: dict[str, object], path: Path) -> None:
    seeds = summary["seeds"]
    by_key = {
        (run["variant"], run["seed"]): run
        for run in summary["runs"]
    }
    classical = np.asarray([
        by_key[("classical_matched", seed)]["validation_macro_auroc"]
        for seed in seeds
    ])
    quantum = np.asarray([
        by_key[("quantum", seed)]["validation_macro_auroc"]
        for seed in seeds
    ])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(seeds))
    width = 0.36
    axes[0].bar(x - width / 2, classical, width, label="Classical matched")
    axes[0].bar(x + width / 2, quantum, width, label="Quantum")
    axes[0].set_xticks(x, [str(seed) for seed in seeds])
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Validation macro AUROC")
    axes[0].set_title("Matched validation performance")
    axes[0].legend()
    differences = quantum - classical
    colors = ["#2ca02c" if value > 0 else "#d62728" for value in differences]
    axes[1].bar(x, differences, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x, [str(seed) for seed in seeds])
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Quantum minus classical AUROC")
    axes[1].set_title("Paired difference (advance rule failed)")
    figure.suptitle("Objective 3 v1.1: Quantum vs Parameter-Matched Control")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def deterministic_zip(files: list[Path], target: Path) -> str:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for source in sorted(files, key=lambda item: item.name):
            info = zipfile.ZipInfo(source.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, source.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive integrity test failed")
    return sha256_file(target)


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")
    for path in (args.summary, args.checksum, args.amendment, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {
        "summary": sha256_file(args.summary),
        "amendment": sha256_file(args.amendment),
        "config": sha256_file(args.config),
    }
    expected = {
        "summary": args.expected_summary_sha256,
        "amendment": args.expected_amendment_sha256,
        "config": args.expected_config_sha256,
    }
    if hashes != expected:
        raise RuntimeError(f"Publication hashes do not match: {hashes}")
    if args.checksum.read_text(encoding="utf-8").split()[0] != hashes["summary"]:
        raise RuntimeError("Validation summary checksum does not match")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    checks = {
        "artifact": summary.get("artifact")
        == "Objective 3 v1.1 enhanced paired validation result",
        "architecture": summary.get("architecture_version")
        == "v1_1_reupload_gated",
        "seeds": summary.get("seeds") == [42, 43, 44],
        "mean_not_positive": float(
            summary.get("mean_quantum_minus_classical_validation_macro_auroc", 1.0)
        ) <= 0.0,
        "insufficient_wins": int(summary.get("quantum_seed_wins", 3)) < 2,
        "no_advance": summary.get("advance_to_single_final_evaluation") is False,
        "no_tuning": summary.get("additional_architecture_tuning_allowed") is False,
        "test_manifest": summary.get("test_manifest_opened") is False,
        "test_labels": summary.get("test_labels_accessed") is False,
        "test": summary.get("test_evaluated") is False,
        "public": summary.get("allowed_for_publication") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Objective 3 negative-result checks failed: {failed}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.source_commit, "HEAD"],
        cwd=REPOSITORY_ROOT, check=False,
    ).returncode:
        raise RuntimeError("Declared source commit is not in repository history")
    if run_git(["status", "--porcelain"]):
        raise RuntimeError("Repository must be clean before publication")

    result_root = REPOSITORY_ROOT / args.result_path
    result_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.summary, result_root / args.summary.name)
    shutil.copy2(args.checksum, result_root / args.checksum.name)
    shutil.copy2(args.amendment, result_root / args.amendment.name)
    shutil.copy2(args.config, result_root / args.config.name)
    figure = result_root / "objective3_v1_1_paired_validation.png"
    make_figure(summary, figure)
    mean_difference = summary[
        "mean_quantum_minus_classical_validation_macro_auroc"
    ]
    readme = f"""# Objective 3 v1.1 paired validation result

- Classical mean validation macro AUROC: {summary['classical_mean_validation_macro_auroc']:.6f}
- Quantum mean validation macro AUROC: {summary['quantum_mean_validation_macro_auroc']:.6f}
- Mean quantum-minus-classical AUROC: {mean_difference:.6f}
- Quantum seed wins: {summary['quantum_seed_wins']} of 3 (2 required)
- Advance to final evaluation: **No**
- Summary SHA-256: `{hashes['summary']}`

The bounded v1.1 quantum enhancement did not satisfy either preregistered
advance condition. Therefore no final cohort was selected, no test labels were
accessed, no final evaluation was performed, and additional Objective 3
architecture tuning is prohibited. This is a valid negative comparative result.
No checkpoints, identifiers, images, private manifests, or predictions are
included.
"""
    (result_root / "README.md").write_text(readme, encoding="utf-8")
    inventory = {
        "artifact": "Objective 3 v1.1 negative validation public inventory",
        "summary_sha256": hashes["summary"],
        "advance_to_final_evaluation": False,
        "objective3_closed": True,
        "private_checkpoints_included": False,
        "identifiers_included": False,
        "medical_images_included": False,
        "private_manifests_included": False,
        "case_level_predictions_included": False,
        "files": {},
    }
    for path in sorted(result_root.iterdir()):
        if path.is_file() and path.name != "artifact_inventory_public.json":
            inventory["files"][path.name] = {
                "bytes": path.stat().st_size, "sha256": sha256_file(path)
            }
    inventory_path = result_root / "artifact_inventory_public.json"
    atomic_json(inventory, inventory_path)
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result_root.iterdir()
        if path.suffix.lower() in {".json", ".md", ".yaml"}
    )
    forbidden = ('"patient_id"', '"image_id"', '"image_path"', '"mask_path"')
    violations = [item for item in forbidden if item in text]
    if violations:
        raise RuntimeError(f"Privacy scan failed: {violations}")

    backups = Path("/kaggle/working/backups")
    backups.mkdir(parents=True, exist_ok=True)
    archive = backups / "objective3_v1_1_validation_public_v1.0.0.zip"
    archive_hash = deterministic_zip(list(result_root.iterdir()), archive)
    archive_checksum = archive.with_suffix(".zip.sha256")
    archive_checksum.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=hf_token)
    if bool(api.model_info(args.hf_repo, token=hf_token).private):
        raise RuntimeError("Public checkpoint repository is unexpectedly private")
    files = [path for path in result_root.iterdir() if path.is_file()]
    files.extend([archive, archive_checksum])
    commit = api.create_commit(
        repo_id=args.hf_repo, repo_type="model", token=hf_token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.hf_path.strip('/')}/{path.name}",
                path_or_fileobj=str(path),
            ) for path in files
        ],
        commit_message="results: publish Objective 3 v1.1 validation result",
    )

    run_git(["config", "user.name", "Ahmed Zuhair"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    relative = str(args.result_path).replace("\\", "/")
    run_git(["add", "--", relative])
    staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
    if any(not name.startswith(relative.rstrip("/") + "/") for name in staged):
        raise RuntimeError(f"Unexpected staged files: {staged}")
    if staged:
        run_git(["commit", "-m", "results: close Objective 3 v1.1 validation"])
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
        run_git(["push", "origin", "main"], environment)
    github_commit = run_git(["rev-parse", "HEAD"])
    print(json.dumps({
        "summary_sha256": hashes["summary"],
        "public_archive_sha256": archive_hash,
        "classical_mean_validation_macro_auroc": summary["classical_mean_validation_macro_auroc"],
        "quantum_mean_validation_macro_auroc": summary["quantum_mean_validation_macro_auroc"],
        "mean_quantum_minus_classical_macro_auroc": mean_difference,
        "quantum_seed_wins": summary["quantum_seed_wins"],
        "advance_to_final_evaluation": False,
        "objective3_closed": True,
        "hf_commit": getattr(commit, "oid", None),
        "hf_path": f"https://huggingface.co/{args.hf_repo}/tree/main/{args.hf_path.strip('/')}",
        "github_commit": github_commit,
        "github_results": f"https://github.com/{args.github_repo}/tree/main/{relative}",
        "test_evaluated": False,
        "privacy_scan_passed": True,
    }, indent=2, sort_keys=True))
    print("OBJECTIVE 3 V1.1 NEGATIVE VALIDATION RESULT PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
