# NIH adaptation annotation-set lock (seed 42)

This directory records sanitized aggregate evidence for the 120-case NIH
target-domain adaptation annotation set used in Objective 1.

- A senior specialist radiologist (anonymous reviewer `RAD-01`) reviewed all
  120 lung ROI masks case by case.
- Automated QC selected 27 cases for focused review. Six were corrected and 21
  were approved unchanged.
- All 120 final masks passed binary-value, source-image shape, and structural
  integrity checks.
- Twenty-seven conservative geometric warnings remain, but every warning has a
  resolved focused-review record.
- The locked NIH target test was not used.

The public JSON contains only aggregate counts and verification hashes. This
directory contains no patient/image identifiers, source medical images,
annotation masks, predictions, or private manifests. Identifier-bearing mask
hash records and review logs remain in the private research workspace.
