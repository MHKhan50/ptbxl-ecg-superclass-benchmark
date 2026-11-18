---
license: mit
library_name: pytorch
tags:
- ecg
- time-series
- medical
- multi-label-classification
- research
---

# CardioNet ECG ResNet

## Model description
A compact 1D residual convolutional network for research benchmarking of five PTB-XL diagnostic superclass labels from 12-lead ECG waveforms.

## Intended use
Research and educational benchmarking only. Not for diagnosis, screening, triage, treatment decisions, or clinical deployment.

## Training data
PTB-XL v1.0.3, official folds 1–8 for training, fold 9 validation, fold 10 test. The dataset is not redistributed with this model.

## Labels
NORM, MI, STTC, CD, HYP. Multi-label sigmoid outputs; thresholds are selected on validation data.

## Evaluation
Populate this section with actual results from `checkpoints/resnet_metrics.json`. Do not publish placeholder or unverified metrics.

## Limitations
Single-dataset evaluation; no external or prospective validation. Dataset labels are not a substitute for clinical adjudication. Performance may vary across populations, devices and acquisition settings.

## Ethical and safety considerations
This model is not a medical device and must not be used to make decisions about an individual's health.
