# Objective 5: External-Domain Harmonisation Protocol

This artifact locks the external-domain evaluation and adaptation
protocol before model inference or test evaluation.

## External datasets

- CheXpert v1.0 Small
- PadChest Small

## Shared label space

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Pneumothorax

## Cohort design

CheXpert:

- 30,000 adaptation-training patients
- 5,000 target-validation patients
- 200 official-validation locked-test patients

PadChest:

- 30,000 adaptation-training patients
- 5,000 target-validation patients
- 5,000 locked-test patients

All roles are patient-disjoint. Selection was label-blind,
prediction-blind, and risk-score-blind.

The PadChest eligibility correction excluded explicit pediatric cases
and metadata rows whose source image was unavailable. It did not use
labels, predictions, or model-derived risk scores.

## Experimental comparison

1. Frozen NIH DenseNet-121 zero-shot baseline
2. Harmonised-preprocessing baseline
3. Target-domain adapted DenseNet-121

Model selection is restricted to target-validation cohorts. Each
locked test may be evaluated only once after all adaptation decisions
are frozen.

## Important limitation

The locked test sets contain very few Pneumothorax-positive cases.
Pneumothorax results must therefore be reported with confidence
intervals and interpreted cautiously.

## Privacy

This public artifact contains no patient identifiers, image
identifiers, image paths, medical images, private manifests,
case-level predictions, or model explanations.
