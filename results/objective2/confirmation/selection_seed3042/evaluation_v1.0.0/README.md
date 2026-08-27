# Objective 2 Independent Confirmation

This is the pre-specified, one-time independent confirmation of the frozen
original CNN and enhanced DenseNet-121 on 5,000 images from 568 complete
patients who did not appear in the original locked-test cohort.

| Model | Macro AUROC | Macro AUPRC | Macro F1 |
|---|---:|---:|---:|
| Original CNN | 0.667197 | 0.149377 | 0.208860 |
| Enhanced DenseNet-121 | 0.764389 | 0.239957 | 0.296831 |

Paired bootstrap DenseNet-minus-CNN mean differences were
0.097176 AUROC,
0.090979 AUPRC, and
0.088049 macro F1. All corresponding
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
