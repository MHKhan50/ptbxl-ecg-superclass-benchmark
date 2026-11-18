# Experimental protocol

## Dataset and task
PTB-XL version 1.0.3. Predict five diagnostic superclass indicators (NORM, MI, STTC, CD, HYP) from 12-lead 10-second ECG waveforms. This is multi-label classification.

## Splits
Use the official stratified patient folds: 1–8 training, 9 validation, 10 test. Patient overlap is checked at preparation time. Fit normalization on training records only. Do not tune hyperparameters or thresholds on the test fold.

## Models
- ECGCNN: stack of Conv1D, batch normalization, ReLU and max pooling.
- ECGResNet: residual Conv1D blocks with adaptive average pooling.
- Loss: binary cross entropy with logits.
- Optimizer: AdamW.
- CPU-first deterministic seed; exact bitwise determinism is not guaranteed across hardware/library versions.

## Metrics
Macro/micro F1, macro precision/recall, per-label AUROC and average precision. Thresholds are selected using validation data only. Report class-wise metrics; macro summaries can obscure minority-class performance.

## Abstention
A record-level mean max(p, 1-p) heuristic is reported at a fixed 0.8 threshold. This is not calibrated clinical uncertainty and does not guarantee safety. It is an exploratory selective-prediction statistic only.

## Limitations
PTB-XL is a curated research dataset; label prevalence, acquisition devices, patient population and annotation processes may limit generalization. No external validation, prospective validation, clinical utility study or regulatory assessment is performed in v0.1.0.
