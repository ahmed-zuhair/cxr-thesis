# CXR thesis code

Research code for explainable graph-based and quantum-classical medical-image
experiments.

The repository's reusable data foundation is documented in
[`docs/objective1.md`](docs/objective1.md). Existing root-level scripts are
retained for reproducibility while new components are added under
`src/cxr_thesis`.

## Objective 1 smoke test

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/objective1.py --help
```

Restricted clinical datasets must never be committed to this repository.

## Objective 2 classification baselines

Objective 2 adds dependency-light PyTorch baselines for multilabel CXR
classification: CNN, CBAM attention-CNN, compact vision transformer, GCN, and
GAT. See [`docs/objective2.md`](docs/objective2.md) for the locked comparison
protocol.
