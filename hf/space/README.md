---
title: ECG Superclass Check
emoji: 🫀
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: 12-lead ECG superclass scores with abstention (research only)
models:
- MHkhan50/ptbxl-ecg-resnet
---

# ECG Superclass Check

Research demo of a compact 1D ResNet trained on PTB-XL 1.0.3. It scores a 10-second 12-lead ECG for five diagnostic superclasses (NORM, MI, STTC, CD, HYP) with per-label thresholds chosen on the validation fold, and sets a record aside when the model's confidence is below 0.8.

**Not a medical device.** Outputs are model scores against dataset labels, not diagnoses.

Model and evaluation: [MHkhan50/ptbxl-ecg-resnet](https://huggingface.co/MHkhan50/ptbxl-ecg-resnet) · Code: [github.com/MHKhan50/ptbxl-ecg-superclass-benchmark](https://github.com/MHKhan50/ptbxl-ecg-superclass-benchmark)
