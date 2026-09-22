# Model card

## Intended use

This model was built for a machine-learning competition involving binary classification of DaT-SPECT studies. It is suitable for methodological demonstration, reproducibility research and analysis of distribution shift.

It is **not intended** for diagnosis, treatment decisions, patient triage or autonomous clinical use.

## Final executable

The archived submission contained 21 files and approximately 82 MB of code, metadata and parameters. The inference path combined a classical ROI/Fourier stack, an adaptive ROI/asymmetry predictor, multiview EfficientNet-B0 features and a two-channel compact 3D CNN ensemble.

Although the submission comment described a 55/45 hybrid, the archived `main.py` executed the following calibrated logit blend:

```text
uncalibrated_logit = 0.634346 * previous_stack_logit
                   + 0.365654 * cnn3d_stratified_logit

final_logit = (uncalibrated_logit + 0.1311148068) / 0.7612935387
```

Five stratified and five grouped CNN checkpoints were loaded. The optimized final prediction used the stratified family; grouped predictions supported hybrid diagnostics and robustness experiments.

## Performance

| Measure | Value |
|---|---:|
| Final blend OOF log loss | 0.361771 |
| Final blend OOF ROC AUC | 0.919448 |
| Repeated cross-fitted log loss | 0.362112 |
| Final private log loss | 0.4471 |
| Final private ROC AUC | 0.8777 |
| Final rank | 254/1,009 |

The difference between OOF and private performance is material and should not be minimized. It suggests remaining distribution shift, optimistic model-selection effects, or both.

## Data and ethical limitations

- Training scans cannot be redistributed from this repository.
- The target represents the competition label and is not equivalent to a full neurological diagnosis.
- Scanner, center and acquisition-protocol effects may influence predictions.
- Demographic subgroup performance was not established.
- External clinical validity, especially in Nicaragua or Latin America, is unknown.

## Weights

The source repository excludes `.pt`, `.joblib` and `.npz` parameter files by default. Before publishing weights separately, verify the competition data and model-distribution terms. If distribution is permitted, use a versioned release or Git LFS and publish checksums; do not place the 75 MB submission archive directly in ordinary Git history.

