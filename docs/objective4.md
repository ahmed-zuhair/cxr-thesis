# Objective 4: quantitative explainability

Objective 4 explains the independently confirmed DenseNet-121 classifier with
Grad-CAM and Integrated Gradients. The protocol uses a deterministic,
patient-unique 240-case validation cohort: 20 label-positive cases for each of
the 12 classification targets. No locked-test manifest, prediction, or risk
score is used during cohort selection.

The public protocol records only aggregate counts, methods, metrics, and
cryptographic hashes. The cohort manifest and all case-level explanations stay
in the private recovery repository. Public reporting is limited to aggregate
faithfulness, stability, anatomical-concentration, and method-agreement
results; medical images and identifiers are not published.
