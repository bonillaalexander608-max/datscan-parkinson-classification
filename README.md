# DaT-SPECT classification under acquisition-protocol shift

Reproducible machine-learning pipeline developed for a DrivenData competition to estimate the probability that a DaT-SPECT scan is pathologic. The project focuses on robustness under heterogeneous acquisition protocols rather than on leaderboard optimization alone.

> **Research-use disclaimer:** this repository is an educational competition project. It is not a medical device, does not diagnose Parkinson's disease, and has not been clinically or externally validated.

## Results

| Evaluation | Log loss | ROC AUC | Notes |
|---|---:|---:|---|
| Final blend OOF | 0.3618 | 0.9194 | Bundled submission metadata |
| Repeated cross-fitted blend | 0.3621 | 0.9160 | Constrained logit blending |
| Best result observed before final evaluation | 0.4103 | 0.8999 | Competition result during development |
| Final private evaluation | 0.4471 | 0.8777 | Rank 254/1,009 (top 25.2%) |

The private-score degradation is reported deliberately: it is the central generalization lesson of the project. The final executable combined a classical ROI/Fourier model, a multiview EfficientNet-B0 representation and a compact 3D CNN. Later protocol-shift experiments are retained separately and must not be confused with the submitted model.

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
6. A cross-fitted bias/temperature calibration was applied after blending.

The original submission description summarized this as a 55/45 hybrid. Inspection of the archived `main.py` showed the exact executable weights above; this repository reports the executed implementation.

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

Exact reproduction can vary with GPU hardware and library versions. Random seeds are fixed where possible, but some CUDA operations may remain nondeterministic.

## Validation design

The submitted stack used stratified and grouped CNN families, while the post-competition study compared stratified folds with folds informed by latent acquisition protocols. Protocol-aware evaluation is a stress test for center/protocol shift. All preprocessing learned from data should be fitted within the training portion of each fold.

## Limitations

- Competition labels are not a substitute for neurological diagnosis.
- Acquisition center and protocol can act as confounders.
- No external Nicaraguan or Latin American clinical cohort was available.
- Performance estimates do not establish clinical utility, safety or transportability.
- The final private score shows a meaningful generalization gap.

## Versión breve en español

Este repositorio documenta un pipeline reproducible para clasificar estudios DaT-SPECT en una competencia de DrivenData. El modelo final obtuvo Log Loss 0.4471, AUC 0.8777 y la posición 254 de 1,009 participantes. El trabajo se presenta como un estudio de generalización y cambio de protocolo, no como una herramienta diagnóstica. Para una aplicación en Nicaragua sería indispensable realizar validación externa con datos locales y colaboración clínica.

## Citation

If this repository supports academic work, cite the repository and its archived release. See [`CITATION.cff`](CITATION.cff).

## License

Code is released under the MIT License. The dataset, competition materials and pretrained weights are excluded and remain subject to their respective terms.

