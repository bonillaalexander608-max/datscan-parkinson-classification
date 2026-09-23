# Model card

> **Target clarification:** this model predicts the competition label “pathologic DaT-SPECT pattern.” It does not diagnose Parkinson's disease or determine a specific neurological disorder.

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

## Calibration provenance

Model and calibration selection used a repeated cross-fitted constrained logit blend with 5 folds and 10 repeats. The repeated cross-fitted estimates were:

| Measure | Value |
|---|---:|
| Log loss | 0.362112 |
| ROC AUC | 0.916002 |
| Brier score | 0.111702 |

In cross-fitted calibration, each held-out fold is transformed using parameters estimated from the other folds. This prevents a patient's label from being used to transform that patient's cross-fitted prediction. After selection, the fixed bias and temperature shown above were refit on the complete OOF predictions and embedded in the archived inference source.

Consequently, the reported full-OOF result 0.361771 is an in-sample optimized quantity, whereas 0.362112 is the preferred repeated cross-fitted estimate. Notebook 05 demonstrates the fold-wise mechanics on a later CNN experiment; it did not generate the deployed constants. The repository preserves the executed constants and inference code, but it does not include the complete original optimization history.

## Performance

| Measure | Value |
|---|---:|
| Full-OOF optimized log loss | 0.361771 |
| Full-OOF optimized ROC AUC | 0.919448 |
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

## Known failure modes

- Signed left-right asymmetry depends on reliable NIfTI orientation metadata. Incorrect affines can silently reverse lateralized features.
- The three ensemble branches use different localization and preprocessing routines, so they may attend to different anatomy in difficult cases.
- Acquisition protocol, scanner geometry and field of view can act as shortcuts or confounders.
- Temperature below 1 sharpens probabilities and can amplify overconfidence under distribution shift.
- A malformed or empty volume can stop batch inference; the archived submission prioritized fail-fast validation over partial output.

## Ensemble weights

The deployed hierarchy in logit space was:

1. Classical Fourier variants: axial 0.14, axial plus sagittal 0.21, multiscale 0.65.
2. Classical stack versus adaptive ROI predictor: 0.50 / 0.50.
3. Resulting stack versus EfficientNet-B0: 0.625 / 0.375.
4. Previous full stack versus stratified 3D CNN: 0.634346 / 0.365654.

At the final top level, the effective contributions are approximately 0.1982 classical stack, 0.1982 adaptive ROI, 0.2379 EfficientNet-B0 and 0.3657 stratified 3D CNN. The separately computed 50/50 stratified/grouped CNN hybrid was retained for diagnostics but was not used in the executed final prediction.

## Weights

The source repository excludes `.pt`, `.joblib` and `.npz` parameter files by default. Before publishing weights separately, verify the competition data and model-distribution terms. If distribution is permitted, use a versioned release or Git LFS and publish checksums; do not place the 75 MB submission archive directly in ordinary Git history.

## Reproducibility scope

- The ordered notebooks document the training and evaluation workflow.
- The archived inference source documents the exact ensemble, blend weights and calibration constants.
- Exact submitted predictions require the excluded trained parameter files and authorized access to the original data.
- “Reproducible” in this repository therefore refers to the documented workflow, not to byte-for-byte reproduction of the final competition submission from the public files alone.
