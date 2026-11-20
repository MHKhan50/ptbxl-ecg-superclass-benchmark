---
license: cc-by-4.0
library_name: pytorch
pipeline_tag: other
tags:
- ecg
- ptb-xl
- medical-signal
- multi-label-classification
- selective-prediction
- research-only
datasets:
- PTB-XL
metrics:
- roc_auc
- f1
---

# PTB-XL ECG ResNet (5 diagnostic superclasses)

A compact 1D residual network (about 0.19 M parameters) that scores a 10-second, 12-lead ECG at 100 Hz for the five PTB-XL diagnostic superclasses: **NORM** (normal ECG), **MI** (myocardial infarction), **STTC** (ST/T change), **CD** (conduction disturbance) and **HYP** (hypertrophy).

> **Research use only. Not a medical device.** Outputs are model scores against PTB-XL labels. They are not diagnoses, triage or treatment advice and must not be used for patient care.

- Code, training and full evaluation: [github.com/MHKhan50/ptbxl-ecg-superclass-benchmark](https://github.com/MHKhan50/ptbxl-ecg-superclass-benchmark)
- Demo: [ECG Superclass Check](https://huggingface.co/spaces/MHkhan50/ecg-superclass-check)

## Which checkpoint and why

Five seeds were trained (17, 23, 31, 47, 59). This release is **seed 23**, the seed with the **lowest validation loss** (0.2662). It was not chosen on test results; its test macro AUROC (0.925) equals the five-seed mean (0.925 ± 0.003).

## Results on the official test fold (fold 10, 2,158 records)

| | NORM | MI | STTC | CD | HYP | Macro |
|---|---|---|---|---|---|---|
| AUROC | 0.941 | 0.927 | 0.931 | 0.920 | 0.904 | **0.925** |

Macro F1 at validation-chosen thresholds: 0.745. At a confidence threshold of 0.8 the model labels 84.5% of records and sets the rest aside.

For context (from the repository, seed ensembles with 1,000 patient-level bootstrap resamples): 1D ResNet 0.932 [0.924, 0.939] against 1D CNN 0.923 [0.915, 0.930]; difference +0.009 [+0.007, +0.011].

## Files

- `resnet_s23_best.pt`: PyTorch checkpoint, a dictionary with `model_state_dict`, `model_name`, `config` and `labels`. It loads with `torch.load(..., weights_only=True)`.
- `config.json`: architecture arguments, lead order, training-fold normalisation (per-lead mean and std in mV), per-label thresholds, abstention rule and test metrics.
- `model.py`: the network definition.

## Use

```python
import json, numpy as np, torch
from huggingface_hub import hf_hub_download
import importlib.util

repo = "MHkhan50/ptbxl-ecg-resnet"
cfg = json.load(open(hf_hub_download(repo, "config.json")))
spec = importlib.util.spec_from_file_location("model", hf_hub_download(repo, "model.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
net = m.ECGResNet(num_labels=5, channels=(32, 64, 128), dropout=0.2)
net.load_state_dict(torch.load(hf_hub_download(repo, "resnet_s23_best.pt"), weights_only=True)["model_state_dict"])
net.eval()

ecg = np.zeros((12, 1000), dtype=np.float32)          # 12 leads in cfg["input"]["leads"] order, mV, 100 Hz, 10 s
mean = np.array(cfg["input"]["mean"], dtype=np.float32)[:, None]
std = np.array(cfg["input"]["std"], dtype=np.float32)[:, None]
p = torch.sigmoid(net(torch.from_numpy((ecg - mean) / std)[None])).detach().numpy()[0]
labels = [k for k, pk in zip(cfg["labels"], p) if pk >= cfg["thresholds"][k]]
confidence = float(np.maximum(p, 1 - p).mean())     # below 0.8: set aside for expert review
```

## Training data and procedure

- **Data:** PTB-XL 1.0.3 (Wagner et al., 2020, *Scientific Data* 7:154), 100 Hz signals, records with at least one diagnostic superclass (21,388). Official patient-wise folds: 1–8 train (17,084), 9 validation (2,146), 10 test (2,158); no patient in two partitions.
- **Training:** AdamW, learning rate 1e-3, weight decay 1e-4, batch size 32, binary cross-entropy, early stopping on validation loss (patience 5, at most 40 epochs; this seed stopped after 14). Kaggle T4 GPU.
- **Thresholds:** per label, chosen on the validation fold and frozen.

## Limitations

- One public dataset from one German centre (recordings 1989–1996); no external validation. Performance on other devices, populations or recording conditions is unknown.
- Superclass labels only; no rhythm or fine-grained statements. Labels are the dataset's, not re-read by cardiologists for this work.
- Only 100 Hz, 10-second, 12-lead input. Missing or swapped leads and different units silently change the output.
- HYP is the weakest label (average precision 0.68).

## Licence and citation

Weights are derived from PTB-XL and released under CC BY 4.0, like the dataset; code is MIT. Please cite PTB-XL:

Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W. and Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data* 7, 154. https://doi.org/10.1038/s41597-020-0495-6

Author: Muzammal Hussain (ORCID 0009-0001-2570-5686).
