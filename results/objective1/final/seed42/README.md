# Objective 1 Complete Research Package

## Scope

Objective 1 implements an end-to-end pipeline for:

- CXR preprocessing
- Radiologist-reviewed lung ROI segmentation
- Domain adaptation
- Radiomic, handcrafted and clinical feature extraction
- Feature-stability filtering
- ROI-aware 2D patch graphs
- Heterogeneous image-radiomics-clinical study graphs

## Key result

On the prediction-blind 40-case locked NIH target test set, target
adaptation improved mean Dice from 0.8278 to 0.8877 and mean IoU from
0.7335 to 0.8032. Paired Dice and IoU improvements were statistically
significant. HD95 improved numerically but was not statistically
significant.

## Feature and graph result

The pipeline extracted 350 model features from 200 cases. A
validation-only perturbation audit retained 159 stable image features:
44 handcrafted and 115 radiomic. The final multimodal representation
contains one ROI node, one radiomics node and one clinical node per
patient, with no inter-patient edges.

## Annotation provenance

The private masks were reviewed case-by-case by one Senior Specialist
Radiologist, represented publicly by the anonymous code RAD-01.

## Privacy

This package contains no medical images, masks, patient/image
identifiers, private manifests, private feature rows, private graphs,
per-case metrics or private reviewer records.

## Scope limitation

The empirical cohort is 2D chest radiography. The repository includes
unit-tested 3D preprocessing and graph-construction capabilities, but
Objective 1 does not claim a full empirical 3D evaluation.

## Repositories

- Source code: https://github.com/ahmed-zuhair/cxr-thesis
- Checkpoints: https://huggingface.co/ahmed-zuhair/cxr-thesis-checkpoints
