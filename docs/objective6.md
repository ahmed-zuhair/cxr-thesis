# Objective 6: Clinical Report Generation

Objective 6 closes the clinical-report-generation component named in the thesis
title. It is a separate, preregistered experiment and does not reinterpret the
classification labels from Objectives 2 or 5 as generated prose.

## Data and language

PadChest is used because its metadata pairs chest radiographs with Spanish
radiology reports. The task therefore generates Spanish reports and evaluates
them against the original Spanish reference text. No machine-translated target
reports are used as training truth.

Eligibility is limited to adult frontal studies with a nonempty report and an
available image. One representative image is selected for each study/report.
All patients previously used by Objective 5 PadChest adaptation, validation, or
locked testing are excluded before Objective 6 cohort selection. The remaining
patients are divided into patient-disjoint training, validation, and a new
locked test cohort.

## Model and comparisons

The primary model combines:

1. the PadChest-adapted DenseNet-121 visual encoder from Objective 5;
2. a separate non-diagnostic clinical token containing age, sex, and view; and
3. an autoregressive Transformer report decoder.

Ground-truth disease labels are never passed to the decoder. An auxiliary
clinical-label loss may regularise the visual representation during training,
but its labels remain unavailable at inference. The preregistered comparisons
are a training-report retrieval baseline, an image-only generator, and the
image-plus-clinical generator. Vocabulary construction uses training reports
only.

## Evaluation and safety

Validation is used for architecture selection and stopping. The locked test is
evaluated once after every decision is frozen. Lexical metrics include BLEU-1
through BLEU-4, ROUGE-L, METEOR, and CIDEr. Because lexical overlap alone is not
clinical correctness, the primary interpretation also reports controlled
PadChest concept precision, recall, F1, and negation errors. Patient-clustered
bootstrap confidence intervals quantify uncertainty.

Raw reports, identifiers, manifests, generated case-level reports, medical
images, and private checkpoints remain in the private recovery repository.
Only aggregate metrics, configuration, hashes, and non-identifying figures are
eligible for public publication.

The first step is the read-only audit:

```bash
python scripts/audit_objective6_report_data.py \
  --metadata-csv /path/to/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv \
  --image-root /path/to/images-224 \
  --objective5-private-root /path/to/objective5/private \
  --output-dir /path/to/objective6_report_data_audit_v1.0.0
```

The audit performs no model inference or training and writes no reports or
identifiers. Its aggregate counts determine the final cohort sizes before the
selection protocol is locked and published.
