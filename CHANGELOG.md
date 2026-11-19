# Changelog

## Unreleased

- The Kaggle notebook also copies each model's best checkpoint (`*_best.pt`) to `reports/`, so the trained weights are kept in the notebook output; notebook stored without outputs.
- Results of the five-seed Kaggle run in `results/kaggle-2026-11-19/` and in the README.
- `write_normalisation` saves the training-fold channel mean and std with the lead order (`normalisation.json`); the main notebook now writes it to `reports/`, and `notebooks/ptbxl_normalisation_stats.ipynb` recomputes it on CPU for the released checkpoints.

## 0.2.0 (2026-11-18)

- Five seeds per model and an `analyse` command: macro AUROC per seed, seed-ensemble AUROC with patient-level bootstrap intervals, paired ResNet − CNN difference and selective-prediction curves (macro F1 at 100% to 50% coverage).
- GPU used automatically when available; per-seed checkpoints, metrics and test probabilities are saved.
- `evaluate` command wired to the existing evaluation code.
- Kaggle notebook (`notebooks/ptbxl_cnn_vs_resnet.ipynb`) for the full experiment.
- Renamed from CardioNet-AgentBench: the project has no agent, so the name now says what it is.
- README: PTB-XL 1.0.3 counts (21,799 records, 18,869 patients), corrected dataset citation, author metadata.

## 0.1.0 (2026-11-18)

First release: data preparation, 1D CNN and 1D ResNet, single-seed training.
