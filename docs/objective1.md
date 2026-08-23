# Objective 1: data-to-graph pipeline

This package implements the reproducible foundation for the thesis objective:

> Develop an end-to-end AI pipeline that includes image preprocessing, ROI
> segmentation, feature extraction (radiomic, handcrafted, clinical), and
> graph construction from 2D/3D medical images.

It provides code and tests, but it does **not** by itself prove that the ROI
segmenter is clinically accurate. A trained checkpoint and a held-out mask
evaluation are required before marking segmentation complete.

## Implemented components

- Canonical multi-dataset study manifest.
- Patient-level leakage detection and deterministic NIH validation splitting.
- Aspect-ratio-preserving CXR preprocessing and aligned mask transforms.
- CT HU windowing and spacing-aware 3D resampling.
- Compact 2D U-Net, segmentation loss, and relative-area fragment cleanup
  that preserves meaningful disconnected anatomy.
- Dice, IoU, HD95, mask failure rate, and mask plausibility checks.
- ROI-conditioned intensity, histogram, LBP, HOG, asymmetry, and shape features.
- Clinical metadata encoding with explicit missingness indicators.
- Optional PyRadiomics adapter for 2D and 3D ROIs.
- ROI-aware 2D patch graphs, 3D patch graphs, and heterogeneous multimodal graphs.
- Dependency-light `.npz` graph artifacts with optional PyG conversion.
- Manifest-to-features-and-graphs command-line pipeline.

## Data contract

The minimum manifest columns are:

| Column | Meaning |
|---|---|
| `dataset` | Dataset name, included in patient identity checks |
| `patient_id` | Stable de-identified patient identifier |
| `study_id` | Stable study identifier |
| `image_id` | Globally unique image identifier |
| `image_path` | Absolute path or path relative to `--data-root` |
| `modality` | `CXR` or `CT` |
| `view` | `PA`, `AP`, lateral, or CT descriptor |
| `split` | `train`, `val`, `test`, or `external` |

Optional columns include `mask_path`, `age`, `sex`, `indication`, and
`label_<name>`. Findings/Impression text that is a report-generation target
must not be included as a model input.

## Target-domain ROI annotation cohort

`scripts/select_roi_annotation_cohort.py` locks a private, patient-disjoint
200-case NIH development cohort: 120 adaptation-training, 40 target-validation,
and 40 locked target-test cases. The locked test is selected by deterministic
identifier hashing before active-QC scores are joined, and its manifest excludes
prediction columns. Adaptation and validation cases mix high-risk predictions
with representatives spanning the remaining risk distribution. The official NIH
test split is never used.

Manual masks are edited with `scripts/annotate_roi_masks.py`. The tool loads
pre-annotations only for adaptation-training and target-validation roles,
writes same-size binary PNG masks atomically, and keeps a resumable private
progress log. The locked-test role refuses pre-annotation files and requires an
explicit prediction-blind confirmation flag.

Before annotation, `scripts/audit_roi_projection.py` performs a resumable,
image-only eligibility audit. It never resolves pre-annotation paths or loads
risk metrics, and records frontal, lateral, other-ineligible, or uncertain
decisions for deterministic same-stratum replacement before masks are edited.
After the first pass, `--review-flagged` restricts the viewer to existing
non-eligible or uncertain decisions for a focused, still image-only second pass.
Confirmed non-frontal development cases are replaced using deterministic
same-split, same-stratum reserves from
`scripts/create_projection_replacement_reserves.py`. Each replacement retains
the rejected case's high-risk or representative selection basis. The locked
target test is not modified by this prediction-aware reserve procedure.
If ephemeral compute storage is lost after the private annotation bundle was
downloaded, `scripts/recover_projection_replacement_reserves.py` matches the
200 retained images back to the canonical NIH manifest using exact file-size
and SHA-256 fingerprints. It then excludes all recovered patients and selects
same-split, same-view/sex/finding reserves by a prediction-blind seeded hash.
Pre-annotations are generated only after reserve membership is fixed.
After image-only audit, `scripts/finalize_projection_replacements.py` selects
the lowest-ranked eligible reserve for each rejected development case. It
backs up the affected worklists and audits, leaves rejected source files as
unreferenced private evidence, proves that locked-test hashes are unchanged,
and writes a private rollback and replacement record before annotation begins.
Completed annotations are checked with `scripts/audit_roi_annotations.py` for
shape, binary values, foreground-area extremes, excessive connected regions,
border contact, and unusually large changes from the frozen pre-annotation.
Flags trigger focused radiologist re-review and never alter a mask automatically.
Passing the private QC table to `scripts/annotate_roi_masks.py --qc-audit`
loads only flagged cases, displays each flag, and records whether the reviewer
approved the saved mask unchanged or corrected it in a separate private log.
After post-review QC, `scripts/finalize_roi_annotation_set.py` verifies the
worklist, progress, QC, focused-review coverage, provenance, binary masks, and
source-image alignment; it then hashes the final mask set without copying it.
The output separates a private identifier-bearing manifest from a sanitized
aggregate summary suitable for later publication.

## Setup

Create a Python 3.10–3.12 environment for the full medical stack. PyRadiomics
and some medical-image libraries may not yet publish wheels for newer Python
versions.

```bash
python -m pip install -e ".[medical,graph,dev]"
python -m pytest
```

The core test suite can also run without PyRadiomics or PyG:

```bash
python -m unittest discover -s tests -v
```

## Build and validate an NIH manifest

```bash
python scripts/objective1.py build-nih-manifest \
  --metadata /data/nih/Data_Entry_2017.csv \
  --train-val-list /data/nih/train_val_list.txt \
  --test-list /data/nih/test_list.txt \
  --images-root /data/nih/images \
  --output artifacts/manifests/nih.csv

python scripts/objective1.py validate-manifest \
  --manifest artifacts/manifests/nih.csv \
  --require-files
```

The official NIH test list remains test. Validation patients are selected only
from the official training list.

## Run CXR feature and graph extraction

First populate `mask_path` with masks generated by a frozen segmenter. Then:

```bash
python scripts/objective1.py extract-cxr \
  --manifest artifacts/manifests/nih.csv \
  --config configs/objective1/default.yaml \
  --output-root artifacts/objective1/nih
```

Outputs are:

- `preprocessed/<image_id>.png`
- `masks/<image_id>.png`
- `graphs/<image_id>.npz`
- `metadata/<image_id>.json`
- `features.csv`

`--allow-full-image-roi` is strictly for software smoke tests. Results created
with that option are not valid ROI, radiomics, or thesis results.

## Segmentation experiment required next

1. Select and document an appropriately licensed lung-mask dataset.
2. Split it by patient, not image.
3. Train `UNet2D` using BCE plus Dice loss.
4. Select the threshold on validation masks.
5. Freeze the checkpoint and threshold.
6. Report Dice, IoU, HD95, and failure rate on held-out masks.
7. Manually audit at least 200 target-domain NIH/MIMIC images.
8. Add corrected target-domain masks if domain shift is material.
9. Record the mask checkpoint hash and configuration in every feature artifact.

Training and mask-generation commands:

```bash
python scripts/train_roi_segmentation.py \
  --manifest artifacts/manifests/lung_masks.csv \
  --data-root /data \
  --output-dir artifacts/segmentation/lung_union

python scripts/generate_roi_masks.py \
  --manifest artifacts/manifests/nih.csv \
  --checkpoint artifacts/segmentation/lung_union/best.pt \
  --data-root /data \
  --mask-dir /data/derived/nih_lung_masks \
  --output-manifest artifacts/manifests/nih_with_masks.csv \
  --batch-size 32 \
  --min-component-fraction 0.001 \
  --uncertainty-margin 0.10 \
  --expected-checkpoint-sha256 CHECKPOINT_SHA256 \
  --resume
```

The training script chooses the mask threshold using validation cases only.
The generation script stores the SHA-256 checkpoint identity in the manifest,
writes per-image QC and run-summary artifacts, checkpoints its manifest during
long runs, and resumes only when the checkpoint and postprocessing signature
match. It does not force every prediction to contain exactly two components.
The audit also records entropy, threshold-margin, boundary-entropy, and
foreground/background confidence proxies. These values are useful for active
learning and QC ranking, but they are not calibrated uncertainty estimates.

## Radiomics protocol required next

The default config leaves radiomics disabled so missing dependencies cannot be
silently confused with an empty feature set. After installing PyRadiomics:

1. Review `configs/objective1/radiomics.yaml` with the supervisor.
2. Fix pixel spacing and resampling policy per dataset.
3. Set `enable_radiomics: true` and the parameter file path.
4. Perturb masks, contrast, and resolution to estimate feature stability.
5. Remove near-constant and unstable features using training data only.
6. Fit scaling and feature selection on training data only.

## Completion criteria

Objective 1 should be called complete only when all of these are available:

- Valid manifests for the primary 2D and secondary 3D datasets.
- Frozen patient-level splits with no leakage.
- A held-out ROI segmentation report and failure analysis.
- Versioned preprocessing and radiomics configurations.
- Feature stability results.
- Saved 2D, 3D, and multimodal graph examples with schema documentation.
- An end-to-end extraction run on the chosen datasets.
- Tests and reproducibility commands passing in a clean environment.
