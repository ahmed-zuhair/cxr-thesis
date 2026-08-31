# Objective 3 v2.0 Part 3 on Kaggle

The ZIP containing this guide is self-contained for the circuit-diagnostics job.
It uses synthetic circuit parameters only and does not require a dataset.

## 1. Add the ZIP to the notebook

Upload `objective3_v2_part3_kaggle_bundle.zip` as a private Kaggle Dataset, then
attach that dataset to the notebook with **Add Input**. Do not type a placeholder
dataset name; the extraction cell below finds the attached ZIP by its exact safe
filename.

## 2. Extract the attached bundle

```python
from pathlib import Path
from zipfile import ZipFile

archive_name = "objective3_v2_part3_kaggle_bundle.zip"
candidates = list(Path("/kaggle/input").glob(f"*/{archive_name}"))
assert len(candidates) == 1, (
    f"Expected one attached {archive_name}, found {len(candidates)}. "
    "Attach the uploaded private dataset in the Kaggle Data panel."
)
archive = candidates[0]
markers = ("locked_test", "test_manifest", "test_labels")
if any(marker in str(archive).lower() for marker in markers):
    raise RuntimeError("Refusing an input path that appears to contain protected data")

destination = Path("/kaggle/working/cxr-thesis")
destination.mkdir(parents=True, exist_ok=True)
with ZipFile(archive) as bundle:
    members = bundle.namelist()
    if any(
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or any(marker in name.lower() for marker in markers)
        for name in members
    ):
        raise RuntimeError("Bundle member path failed the safety check")
    bundle.extractall(destination)

runner = destination / "scripts/run_objective3_v2_circuit_diagnostics.py"
assert runner.is_file(), f"Missing runner after extraction: {runner}"
print(runner)
```

## 3. Run the smoke check

```python
%cd /kaggle/working/cxr-thesis
```

```bash
!python scripts/run_objective3_v2_circuit_diagnostics.py \
  --smoke \
  --seed 42 \
  --output-dir /kaggle/working/results/objective3_v2/circuit_diagnostics/v2.0.0/smoke
```

## 4. Run the complete preregistered sweep

```bash
!python scripts/run_objective3_v2_circuit_diagnostics.py \
  --seed 42 \
  --expressibility-samples 5000 \
  --entangling-samples 5000 \
  --gradient-samples 200 \
  --fidelity-bins 75 \
  --output-dir /kaggle/working/results/objective3_v2/circuit_diagnostics/v2.0.0
```

The complete run writes `results.json`, its SHA-256 sidecar, three figures, and
hash-verified resumable shards. A full Kaggle session restart erases
`/kaggle/working`; preserve completed outputs by saving a notebook version and
attaching that saved output before a later session.
