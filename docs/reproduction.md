# Reproduction guide

This repository separates three reproducibility targets:

1. The post-competition training and evaluation workflow in `notebooks/`.
2. The archived inference source in `src/submission/`.
3. The exact competition prediction, which additionally requires excluded trained weights and organizer-authorized data.

## Local configuration

Create a virtual environment, install `requirements.txt`, copy `.env.example` to `.env`, and set:

```text
DATSCAN_DATA_ROOT=/absolute/path/to/local/datscan/data
```

The notebooks import `DATA_ROOT` from `src/config.py`. Launch Jupyter from the repository root or from `notebooks/`; both locations are supported.

## Expected private data layout

The configured data root must contain the organizer-provided NIfTI directory and label CSV using the names referenced in notebook 01. Generated project artifacts are written below `DATA_ROOT/latent_protocol_cv/` and are excluded from Git.

## Interactive quality-control gates

The workflow is not intended to run unattended from beginning to end. These gates deliberately stop execution:

- Notebook 02: inspect preprocessing figures, then set `VISUAL_AUDIT_APPROVED=True`.
- Notebook 03: inspect bilateral ROI localization, then set `ROI_AUDIT_APPROVED=True`.
- Notebook 06: inspect original-slice crops, then set `VISUAL_AUDIT_APPROVED=True`.
- Notebook 08: select a valid OOF reference file and set `APPROVE_REFERENCE=True` after confirming that every prediction is genuinely out of fold.

These controls prevent silent propagation of anatomical or validation errors. They also mean that a clean reproduction requires human review at specified stages.

## Missing public artifacts

Raw scans, restricted labels, model weights, cached arrays and OOF prediction files are not redistributed. Notebook outputs were cleared before publication. Metrics in `results/` come from archived metadata and recorded project results; they cannot all be regenerated from the public repository alone.

## Calibration interpretation

The repeated 5-fold × 10-repeat cross-fitted procedure estimates internal performance without transforming a patient's prediction using that patient's label. The deployed bias and temperature were then refit on the complete OOF predictions. Therefore:

- `0.362112` is the preferred repeated cross-fitted log-loss estimate.
- `0.361771` is the full-OOF optimized value and is slightly optimistic.

Notebook 05 demonstrates cross-fitted calibration mechanics for a later protocol experiment; it is not the provenance of the deployed submission constants.

## Current verification boundary

The public repository supports source inspection, environment setup and rerunning the documented workflow with authorized data. It does not yet provide a CI-verified end-to-end reproduction, checksummed weights or a public comparable table with uncertainty intervals for every validation scheme.
