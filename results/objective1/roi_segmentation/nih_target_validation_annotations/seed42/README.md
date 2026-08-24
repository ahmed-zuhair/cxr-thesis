# NIH target-validation annotation-set lock (seed 42)

This directory records sanitized aggregate evidence for the 40-case NIH
target-domain validation annotation set used in Objective 1.

- A senior specialist radiologist (anonymous reviewer `RAD-01`) reviewed all
  40 lung ROI masks case by case.
- Twenty-three final masks differ from the model preannotations; 17 were
  accepted unchanged after review.
- Automated QC selected 12 cases for focused review, and all 12 were approved.
- All final masks passed binary-value, source-image shape, and structural
  integrity checks.
- The locked target-test split was not accessed.

The public JSON contains only aggregate counts and verification hashes. This
directory contains no patient/image identifiers, medical images, annotation
masks, predictions, review logs, or private manifests.
