# CardioNet-AgentBench v0.1.0

**CPU-first, reproducible deep-learning benchmark for multi-label ECG classification using PTB-XL.**

This research software trains and evaluates 1D CNN and 1D ResNet models on the five PTB-XL diagnostic superclasses: `NORM`, `MI`, `STTC`, `CD`, and `HYP`.

> **Research-use only. Not a medical device.** Outputs are model predictions over dataset labels, not diagnoses, triage, treatment recommendations, or clinical decision support. Do not use for patient care. PTB-XL labels are dataset annotations and do not establish clinical ground truth for an individual.

## Research question

Can a compact 1D residual network improve multi-label ECG superclass classification over a 1D CNN baseline under a patient-level evaluation protocol, and how does confidence-threshold abstention change coverage and selective performance?

## Dataset

- PTB-XL: 21,837 ten-second 12-lead ECG records from 18,885 patients.
- Official source: https://physionet.org/content/ptb-xl/1.0.3/
- Paper: https://doi.org/10.1038/s	data? (See `CITATION.cff` and `docs/REFERENCES.md` for corrected bibliographic details.)
- Dataset is not bundled. The downloader retrieves the official PhysioNet v1.0.3 files through WFDB; allow roughly 3 GB expanded disk space. Access is governed by the dataset's CC BY 4.0 licence and attribution requirements.
- This implementation uses the 100 Hz waveform files by default and maps SCP statements to five diagnostic superclasses via `scp_statements.csv`.

## Quick start (macOS CPU)

```bash
cd CardioNet-AgentBench-v0.1.0
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

Download PTB-XL (large; allow time and disk space):

```bash
python -m cardionet.cli download --output data/raw
```

Prepare a patient-level split and cached arrays:

```bash
python -m cardionet.cli prepare --data-dir data/raw --output-dir data/processed --sampling-rate 100
```

Train a baseline (CPU):

```bash
python -m cardionet.cli train --config configs/default.yaml --model cnn --epochs 5
```

Train ResNet:

```bash
python -m cardionet.cli train --config configs/default.yaml --model resnet --epochs 10
```

Evaluate a checkpoint:

```bash
python -m cardionet.cli evaluate --config configs/default.yaml --checkpoint checkpoints/resnet_best.pt
```

All commands are CPU-compatible. Training time depends on CPU, disk, and PyTorch build. Start with a smaller epoch count to validate the pipeline.

## Methodology and safeguards

- Patient-level official folds: folds 1–8 train, 9 validation, 10 test. No patient should cross partitions; the preparation step checks this.
- Multi-label targets are the five superclass indicators; no forced single-class argmax.
- Training-only normalization statistics are applied to validation/test.
- Validation threshold selection is performed per label, then frozen for test evaluation.
- Reports include macro/micro F1, macro AUROC (where defined), per-label AUROC and average precision, and confusion-independent multilabel metrics.
- Optional abstention is a simple confidence rule over sigmoid scores. It is a research analysis, not a clinical safety guarantee.
- No external LLM/agent is used in v0.1.0. “AgentBench” denotes a reproducible benchmark scaffold, not autonomous clinical agents.

## Repository layout

```text
src/cardionet/       data, models, training, evaluation, CLI
configs/             experiment configuration
scripts/             convenience scripts
tests/               unit tests
docs/                protocol, model card, references
reports/             generated metrics (not committed by default)
```

## Hugging Face publication

See `docs/HUGGINGFACE.md`. Publish only model weights, configuration, model card, and a small synthetic example if desired. Do not upload PTB-XL waveform files or patient-identifying data. Check dataset terms before redistributing any derived artifacts. A Hugging Face repository is not clinical validation.

## Reproducibility

Record Python/PyTorch versions, random seed, config, git commit, dataset version and dataset checksum in each run's `run_metadata.json`. The default seed is 17.

## Citation

If you use this software, cite the repository and the PTB-XL dataset paper listed in `docs/REFERENCES.md`.
