# Objective 3 v2.0 Part 4 on Kaggle

The Part 4 overlay contains the powered-validation runner and the pure
statistical-analysis runner. It must be applied to the full `cxr-thesis`
repository because Job 1 imports the unchanged v1.1 trainer and model modules.

## 1. Attach the overlay

Upload `objective3_v2_part4_overlay.zip` as a private Kaggle Dataset and attach
it to the notebook with **Add Input**. Kaggle may unpack the ZIP automatically.

## 2. Clone the real repository and apply the overlay

```python
from pathlib import Path
import shutil
import subprocess

working = Path("/kaggle/working")
repository = working / "cxr-thesis"
if not repository.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/ahmed-zuhair/cxr-thesis.git",
            str(repository),
        ],
        check=True,
    )

overlay_files = {
    "run_objective3_v2_powered_validation.py": (
        repository / "scripts/run_objective3_v2_powered_validation.py"
    ),
    "run_objective3_v2_statistical_analysis.py": (
        repository / "scripts/run_objective3_v2_statistical_analysis.py"
    ),
}
for name, destination in overlay_files.items():
    matches = list(Path("/kaggle/input").rglob(name))
    assert len(matches) == 1, f"Expected one attached {name}; found {len(matches)}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], destination)

print(repository)
```

## 3. Run both smoke paths

```python
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        str(repository / "scripts/run_objective3_v2_powered_validation.py"),
        "--smoke",
        "--output-root",
        "/kaggle/working/results/objective3_v2/powered_validation/smoke",
    ],
    cwd=repository,
    check=True,
)

subprocess.run(
    [
        sys.executable,
        str(repository / "scripts/run_objective3_v2_statistical_analysis.py"),
        "--smoke",
        "--figures",
        "--output-dir",
        "/kaggle/working/results/objective3_v2/statistical_analysis/smoke",
    ],
    cwd=repository,
    check=True,
)
```

## 4. Full-run preflight

The full job must not start unless all of these persistent inputs have been
restored after a Kaggle restart:

```python
from pathlib import Path
from importlib.util import find_spec

protocol = Path(
    "/kaggle/working/results/objective3_v2/protocol/v2.0.0/protocol.json"
)
protocol_sidecar = protocol.with_name("protocol.json.sha256")
train_manifest = Path(
    "/kaggle/working/outputs/objective2_nih_cohort_lock_seed42_v1.0.0/"
    "private/train_cohort_private.csv"
)
validation_manifest = Path(
    "/kaggle/working/outputs/objective2_nih_cohort_lock_seed42_v1.0.0/"
    "private/val_cohort_private.csv"
)
embedding_root = Path(
    "/kaggle/working/outputs/objective3_frozen_gat_embeddings_seed42_v1.0.0"
)

for required in (
    protocol,
    protocol_sidecar,
    train_manifest,
    validation_manifest,
    embedding_root,
):
    assert required.exists(), f"Restore required input before training: {required}"

assert find_spec("huggingface_hub") is not None, (
    "The existing private-recovery dependency is unavailable; stop before training."
)
print("FULL-RUN INPUT PRECHECK PASSED")
```

Install the already-pinned project quantum dependency if it is not present:

```python
if find_spec("pennylane") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pennylane==0.45.1"],
        check=True,
    )
```

Load the private recovery secret without printing it:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN").strip()
assert bool(os.environ["HF_TOKEN"])
print("HF token loaded:", bool(os.environ["HF_TOKEN"]))
```

## 5. Run Job 1: powered validation

```python
expected_protocol_sha256 = protocol_sidecar.read_text().split()[0]
powered_output = Path(
    "/kaggle/working/outputs/objective3_v2_powered_validation_v2.0.0"
)

subprocess.run(
    [
        sys.executable,
        str(repository / "scripts/run_objective3_v2_powered_validation.py"),
        "--protocol", str(protocol),
        "--expected-protocol-sha256", expected_protocol_sha256,
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(validation_manifest),
        "--embedding-root", str(embedding_root),
        "--output-root", str(powered_output),
        "--hf-repo", "ahmed-zuhair/cxr-thesis-private-recovery",
        "--hf-base-path", "objective3_v2/powered_validation/v2.0.0",
        "--expected-train-sha256",
        "eaca73a0fe7b0883216b67228e89dbee83e89646acfe32ea00ccf99b842cfef6",
        "--expected-val-sha256",
        "3d1f3a984ab92cac43dbd91696639f46fb4046540dbf80c968415f2106506704",
        "--expected-gat-sha256",
        "f34c3db2038c136077011659daee2a1a7d799cc6f87652ddd94d8b5fced7c70d",
        "--epochs", "30",
        "--patience", "6",
        "--batch-size", "256",
        "--learning-rate", "0.001",
        "--weight-decay", "0.0001",
        "--dropout", "0.2",
        "--poll-seconds", "2",
    ],
    cwd=repository,
    check=True,
)
```

Rerun the identical cell after an interruption. Completed run summaries and
aggregate shards are verified by hash and reused.

## 6. Run Job 2: statistical analysis

```python
analysis_output = Path(
    "/kaggle/working/results/objective3_v2/statistical_analysis/v2.0.0"
)
subprocess.run(
    [
        sys.executable,
        str(repository / "scripts/run_objective3_v2_statistical_analysis.py"),
        "--validation-results", str(powered_output / "results.json"),
        "--protocol", str(protocol),
        "--expected-protocol-sha256", expected_protocol_sha256,
        "--output-dir", str(analysis_output),
        "--bootstrap-resamples", "10000",
        "--seed", "42",
        "--figures",
    ],
    cwd=repository,
    check=True,
)
```

Save a Kaggle notebook version with outputs immediately after each completed
stage. `/kaggle/working` is not persistent across a fully restarted session.
