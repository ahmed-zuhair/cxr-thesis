# NIH Target-Domain ROI Segmentation Smoke Audit

This artifact records a 32-case target-domain transfer audit for
the frozen Objective 1 lung ROI segmenter.

## Sampling design

The sample contains 32 unique NIH development patients:

- train and validation only;
- no official NIH test cases;
- PA and AP views;
- female and male patients;
- No Finding and abnormal groups;
- two patients per 16-factor combination.

## Frozen model

- Architecture: compact 2D U-Net
- Checkpoint SHA-256: `94a20eee892f9240df4a4534ced85e55f3677975c627ee336db7eb51a98e5642`
- Validation-selected threshold: 0.55
- Postprocessing: relative component-area threshold of 0.1%
- Fixed two-component restriction: not used

## Automated QC

- Masks generated: 32
- Empty masks: 0
- Automatically implausible masks: 0
- Border-touching model-space masks: 0
- Cases outside training ROI-fraction range: 1
- Cases with more than three retained components: 1
- Automatic QC flags: 1

## Technical visual review

This was an AI-assisted technical review, not a clinical or
radiologist assessment.

- Clear major failures: cases 12 and 32
- Probable under-segmentation: cases 10, 16, and 26
- Additional focused review recommended: cases 1, 21, and 27

Standard PA cases generally transferred well. Material failures were
observed in portable/AP, rotated, severely abnormal, low-volume, and
heavily obscured radiographs.

## Decision

**Full NIH inference is not approved at this stage.**

The experiment shows that automated size/component checks alone are
insufficient. Target-domain adaptation, focused annotation, and a
stronger QC gate are required before masks are used for radiomics,
feature extraction, or graph construction.

## Important limitation

NIH ChestX-ray14 does not provide lung-mask ground truth for these
cases. Therefore Dice, IoU, and HD95 were not computed for this
target-domain smoke audit.

## Privacy boundary

Published files contain anonymized case indices and aggregate QC.
Patient IDs, image IDs, study IDs, private manifests, generated masks,
and clinical text are excluded.

The visual figures contain de-identified examples from the public
NIH ChestX-ray14 research dataset and remain subject to its original
terms.

## Files

- `metrics/nih_roi_smoke_reviewed_summary.json`
- `metrics/nih_roi_smoke_anonymized_per_case_qc.csv`
- `figures/visual_audit_cases_01_16.png`
- `figures/visual_audit_cases_17_32.png`
- `reproducibility/sampling_design.json`
