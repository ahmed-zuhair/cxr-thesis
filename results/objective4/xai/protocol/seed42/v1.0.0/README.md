# Objective 4 quantitative XAI protocol lock

- Model: independently confirmed DenseNet-121
- Cohort: 240 validation cases from 240 unique patients
- Allocation: 20 label-positive cases for each of 12 disease targets
- Methods: Grad-CAM and Integrated Gradients
- Public protocol SHA-256: `a596c284bc60ab3b7fc680e7f10760b32249963104e9ce81b98d0c7aa4d6f7b7`
- Private cohort SHA-256: `daa7eeda7104f64dcd353f45604310748ca2ff84ea9ffa7cb4110e7c8daa0d2a`
- DenseNet checkpoint SHA-256: `2b7fa0d2f3dee3c59c538be15dd0435c71ad26b411fc1312bd7e5fe99fbac55f`
- Protocol source commit: `9880bbca67deb05aebe454333830cac5a36d2950`

The cohort was selected deterministically before explanation generation.
Predictions and risk scores were not used for selection. The locked test
manifest was not opened or evaluated. The private cohort manifest, patient and
image identifiers, medical images, and case-level explanation maps are not
included in this public artifact.
