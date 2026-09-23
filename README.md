# DaT-SPECT classification under acquisition-protocol shift

Documented training and evaluation workflow developed for a DrivenData competition to estimate the probability that a DaT-SPECT scan is pathologic. The project focuses on robustness under heterogeneous acquisition protocols rather than on leaderboard optimization alone. The source code and ordered experiments are reproducible with authorized access to the competition data; exact reproduction of the archived submission also requires the excluded trained weights.

> **Research-use disclaimer:** this repository is an educational competition project. It is not a medical device, does not diagnose Parkinson's disease, and has not been clinically or externally validated.

The prediction target is the competition label **pathologic DaT-SPECT pattern**. It is not a neurological diagnosis and must not be interpreted as one.

## Results

| Evaluation | Log loss | ROC AUC | Notes |
|---|---:|---:|---|
| Full-OOF optimized final blend | 0.3618 | 0.9194 | Optimistic in-sample fit reported by bundled metadata |
| Repeated cross-fitted blend estimate | 0.3621 | 0.9160 | 5 folds × 10 repeats; preferred internal estimate |
| Best result observed before final evaluation | 0.4103 | 0.8999 | Competition result during development |
| Final private evaluation | 0.4471 | 0.8777 | Rank 254/1,009 (top 25.2%) |

The private-score degradation is reported deliberately: it is the central generalization lesson of the project. The final executable combined a classical ROI/Fourier model, a multiview EfficientNet-B0 representation and a compact 3D CNN. Later protocol-shift experiments are retained separately and must not be confused with the submitted model.

## Baselines and component evidence

The project did not begin with the final ensemble. A CPU-compatible baseline used bilateral ROI and asymmetry features with Elastic Net logistic regression. The following result is retained as post-competition baseline evidence:

| Model | Validation | Log loss | ROC AUC | Interpretation |
|---|---|---:|---:|---|
| ROI/asymmetry + Elastic Net | Stratified 5-fold OOF | 0.4418 | 0.8780 | Simple engineered-feature baseline |

The archived submission metadata records the component and blend results below:

| Model | Validation recorded in submission metadata | Log loss | ROC AUC |
|---|---|---:|---:|
| Previous ROI/Fourier/EfficientNet stack | OOF metadata | 0.3647 | Not recorded |
| Stratified compact 3D CNN | OOF metadata | 0.4029 | Not recorded |
| Grouped compact 3D CNN | OOF metadata | 0.4152 | Not recorded |
| Stratified/grouped CNN hybrid | OOF metadata | 0.3804 | Not recorded |
| Repeated cross-fitted final blend | 5 folds × 10 repeats | 0.3621 | 0.9160 |
| Executed final blend | Full-OOF optimized metadata | 0.3618 | 0.9194 |

These rows document different stages and, in the case of the post-competition Elastic Net baseline, a different experiment. They are not presented as a single head-to-head benchmark. Their purpose is to show the progression from a simple baseline to the submitted ensemble without implying comparability that the available artifacts do not support. Machine-readable values are in [`results/component_metrics.csv`](results/component_metrics.csv).

## Dataset

- 1,362 labeled training studies.
- Binary target: normal versus pathologic DaT scan.
- Heterogeneous shapes, voxel spacing, field of view and acquisition protocols.
- The original NIfTI images and competition files are **not redistributed** here. Access them from the competition organizer and comply with its terms.

## Submitted ensemble

1. Classical ROI/asymmetry and Fourier-derived predictors.
2. Multiview EfficientNet-B0 image embeddings with a portable linear head.
3. Compact two-channel 3D CNN using the normalized volume and signed left-right difference.
4. Five stratified and five grouped CNN checkpoints were bundled for inference and diagnostics.
5. The executed final blend used 63.4346% of the previous stack and 36.5654% of the stratified CNN family in logit space.
6. An affine logit calibration (bias plus temperature scaling) was applied after blending. Its deployed constants were refit on the complete OOF predictions after model selection; repeated cross-fitting was used to estimate internal performance.

The original submission description summarized this as a 55/45 hybrid. Inspection of the archived `main.py` showed the exact executable weights above; this repository reports the executed implementation.

### Calibration provenance

The executed prediction used:

```text
uncalibrated_logit = 0.634346 * previous_stack_logit
                   + 0.365654 * cnn3d_stratified_logit

final_logit = (uncalibrated_logit + 0.1311148068) / 0.7612935387
```

Model and calibration selection used a repeated cross-fitted constrained logit procedure with 5 folds and 10 repeats. In that evaluation, parameters for each held-out fold were learned from the remaining folds, so a patient's label was not used to transform that patient's cross-fitted prediction. The resulting internal estimates were log loss 0.362112, ROC AUC 0.916002 and Brier score 0.111702.

The constants executed by `src/submission/main.py` were subsequently refit on all available OOF predictions. Therefore, the full-OOF value 0.361771 is an in-sample optimized quantity and is expected to be slightly optimistic; 0.362112 is the more defensible internal performance estimate. Notebook 05 demonstrates fold-wise temperature and Platt calibration for the later protocol study, but it is not the source of the deployed submission constants.

Because the original optimization history and trained parameter files are not distributed, the repository documents and audits the executed calibration but does not claim byte-for-byte reproducibility of the submitted predictions.

## Post-competition robustness study

The numbered notebooks extend the competition work with latent protocol clustering, protocol-aware preprocessing, calibration diagnostics and validation stress tests. These experiments were conducted after the archived submission and are documented as follow-up research, not as the source of the private leaderboard result.

## Repository structure

```text
.
├── notebooks/          # Ordered post-competition experiments
├── src/
│   ├── config.py       # Shared portable paths
│   └── submission/     # Exact inference source from the archived submission
├── results/            # Small, shareable result summaries
├── docs/               # Additional project documentation
├── scripts/            # Data-free repository validation
├── .env.example        # Local path configuration
└── requirements.txt
```

Some model-training notebooks require a CUDA-capable GPU; the ROI/Elastic Net workflow can run on CPU.

## Reproduction

```bash
git clone <YOUR-REPOSITORY-URL>
cd datscan-parkinson-classification
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, set `DATSCAN_DATA_ROOT`, and reproduce the notebooks in numerical order. Generated arrays, checkpoints, raw scans and submissions are ignored by Git. Trained weights are not committed in the source repository; see the model card for their role and release considerations.

The notebooks are intentionally interactive. Notebooks 02, 03 and 06 stop at visual quality-control gates until the user reviews the generated figures and changes the corresponding approval flag. Notebook 08 also requires an explicitly selected and approved OOF reference file; that restricted/generated artifact is not redistributed. See [`docs/reproduction.md`](docs/reproduction.md) before running the sequence.

Exact reproduction can vary with GPU hardware and library versions. Random seeds are fixed where possible, but some CUDA operations may remain nondeterministic.

Static checks that do not require competition data can be run with:

```bash
python scripts/validate_repository.py
```

The same check runs in GitHub Actions on every push and pull request.

### Reproducibility levels

- **Training/evaluation workflow:** reproducible with authorized access to the competition data and the documented environment.
- **Inference source:** preserved in `src/submission/` and auditable, including the executed blend and calibration constants.
- **Exact submitted predictions:** not reproducible from this repository alone because the trained `.pt`, `.joblib` and `.npz` files are excluded pending verification of redistribution terms.

## Validation design

The submitted stack used stratified and grouped CNN families, while the post-competition study compared stratified folds with folds informed by latent acquisition protocols. Protocol-aware evaluation is a stress test for center/protocol shift. All preprocessing learned from data should be fitted within the training portion of each fold.

## Limitations

- Competition labels are not a substitute for neurological diagnosis.
- Acquisition center and protocol can act as confounders.
- No external Nicaraguan or Latin American clinical cohort was available.
- Performance estimates do not establish clinical utility, safety or transportability.
- The final private score shows a meaningful generalization gap.
- Left-right features depend on trustworthy NIfTI orientation metadata. Incorrect or identity affines after DICOM conversion can invalidate signed asymmetry features.
- The post-competition protocol comparisons currently lack bootstrap confidence intervals in the public summary and should be treated as exploratory.

## Versión breve en español

Este repositorio documenta un flujo reproducible de entrenamiento y evaluación para clasificar la etiqueta de competencia de estudios DaT-SPECT. La reproducción exacta de la entrega requiere pesos que no se distribuyen en el repositorio. El modelo final obtuvo Log Loss 0.4471, AUC 0.8777 y la posición 254 de 1,009 participantes. El trabajo se presenta como un estudio de generalización y cambio de protocolo, no como una herramienta diagnóstica. Para una aplicación en Nicaragua sería indispensable realizar validación externa con datos locales y colaboración clínica.

## Citation

If this repository supports academic work, cite the repository and its archived release. See [`CITATION.cff`](CITATION.cff).

## License

Code is released under the MIT License. The dataset, competition materials and pretrained weights are excluded and remain subject to their respective terms.
