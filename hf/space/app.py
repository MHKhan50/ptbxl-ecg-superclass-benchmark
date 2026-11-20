"""ECG Superclass Check: research demo of a 1D ResNet trained on PTB-XL (not a medical device)."""
import json
import os
import tempfile

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

from model import ECGResNet

REPO = "MHkhan50/ptbxl-ecg-resnet"
CFG = json.load(open(hf_hub_download(REPO, "config.json")))
CKPT = torch.load(hf_hub_download(REPO, "resnet_s23_best.pt"), map_location="cpu", weights_only=True)
MODEL = ECGResNet(**{k: tuple(v) if isinstance(v, list) else v for k, v in CFG["model_kwargs"].items()})
MODEL.load_state_dict(CKPT["model_state_dict"])
MODEL.eval()

LEADS = CFG["input"]["leads"]
MEAN = np.asarray(CFG["input"]["mean"], dtype=np.float32)[:, None]
STD = np.asarray(CFG["input"]["std"], dtype=np.float32)[:, None]
LABELS = CFG["labels"]
THRESH = np.asarray([CFG["thresholds"][k] for k in LABELS])
B, Y, R, G = "#1F4E9C", "#E3A800", "#C8102E", "#5FA65A"
EXAMPLES = {"PTB-XL record 00001": "00001", "PTB-XL record 00002": "00002", "PTB-XL record 00003": "00003"}


def to_100hz(sig: np.ndarray, fs: float) -> np.ndarray:
    """sig: (12, n) in mV. Returns the first 10 s at 100 Hz, shape (12, 1000)."""
    if fs != 100:
        sig = resample_poly(sig, 100, int(round(fs)), axis=1)
    if sig.shape[1] < 1000:
        raise gr.Error("The ECG must be at least 10 seconds long.")
    return sig[:, :1000].astype(np.float32)


def read_upload(files):
    paths = [f if isinstance(f, str) else f.name for f in (files or [])]
    if not paths:
        raise gr.Error("Upload an ECG or pick an example.")
    hea = [p for p in paths if p.endswith(".hea")]
    if hea:
        import wfdb
        folder = tempfile.mkdtemp()
        for p in paths:
            os.symlink(p, os.path.join(folder, os.path.basename(p)))
        rec = wfdb.rdrecord(os.path.join(folder, os.path.basename(hea[0])[:-4]))
        names = [n.upper() for n in rec.sig_name]
        missing = [l for l in LEADS if l not in names]
        if missing:
            raise gr.Error("Missing leads: " + ", ".join(missing))
        sig = np.stack([rec.p_signal[:, names.index(l)] for l in LEADS])
        return to_100hz(np.nan_to_num(sig), rec.fs)
    p = paths[0]
    arr = np.load(p) if p.endswith(".npy") else np.loadtxt(p, delimiter=",", skiprows=1 if not _numeric_first_line(p) else 0)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or 12 not in arr.shape:
        raise gr.Error("Expected a 12-lead array: 12 columns (or rows) in the order " + ", ".join(LEADS) + ".")
    sig = arr if arr.shape[0] == 12 else arr.T
    return to_100hz(sig, 100)


def _numeric_first_line(path):
    with open(path) as fh:
        first = fh.readline()
    try:
        [float(v) for v in first.strip().split(",")]
        return True
    except ValueError:
        return False


def load_example(name):
    import wfdb
    rid = EXAMPLES[name]
    rec = wfdb.rdrecord(rid + "_lr", pn_dir="ptb-xl/1.0.3/records100/00000")
    names = [n.upper() for n in rec.sig_name]
    return to_100hz(np.stack([rec.p_signal[:, names.index(l)] for l in LEADS]), rec.fs)


def plot(sig):
    fig, axes = plt.subplots(6, 2, figsize=(11, 8), sharex=True)
    t = np.arange(sig.shape[1]) / 100
    for i, ax in enumerate(axes.T.flat):
        ax.plot(t, sig[i], color=B, lw=0.8)
        ax.set_ylabel(LEADS[i], rotation=0, labelpad=18, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("seconds")
    fig.tight_layout()
    return fig


def run(sig):
    x = torch.from_numpy((sig - MEAN) / STD).unsqueeze(0)
    with torch.no_grad():
        p = torch.sigmoid(MODEL(x)).squeeze(0).numpy()
    conf = float(np.maximum(p, 1 - p).mean())
    rows = [[k, CFG["label_names"][k], round(float(pi), 3), float(t), "yes" if pi >= t else "no"]
            for k, pi, t in zip(LABELS, p, THRESH)]
    found = [k for k, pi, t in zip(LABELS, p, THRESH) if pi >= t]
    if conf < CFG["abstention"]["threshold"]:
        verdict = (f"**Low confidence ({conf:.2f} < {CFG['abstention']['threshold']}).** The model would set this "
                   "record aside for expert review rather than label it.")
    else:
        verdict = f"**Confidence {conf:.2f}.** Labels above threshold: **{', '.join(found) if found else 'none'}**."
    return rows, verdict, plot(sig)


def predict_upload(files):
    return run(read_upload(files))


def predict_example(name):
    return run(load_example(name))


with gr.Blocks(title="ECG Superclass Check") as demo:
    gr.Markdown(
        "# ECG Superclass Check\n"
        "A 1D ResNet trained on PTB-XL scores a 10-second 12-lead ECG for five diagnostic superclasses "
        "and sets low-confidence records aside.\n\n"
        "> **Research demo only. Not a medical device.** Outputs are model scores against PTB-XL labels, not a "
        "diagnosis. Do not use for patient care.\n\n"
        "Input: WFDB (`.hea` + `.dat`, any sampling rate), or CSV/NPY with 12 leads in the order "
        + ", ".join(LEADS) + ", in mV at 100 Hz. The first 10 seconds are used.")
    with gr.Row():
        with gr.Column():
            files = gr.File(label="ECG files", file_count="multiple")
            btn = gr.Button("Score uploaded ECG", variant="primary")
        with gr.Column():
            ex = gr.Dropdown(list(EXAMPLES), value=list(EXAMPLES)[0], label="Or load a PTB-XL record from PhysioNet (may be from the training folds)")
            exbtn = gr.Button("Score example")
    verdict = gr.Markdown()
    table = gr.Dataframe(headers=["Label", "Meaning", "Probability", "Threshold", "Above threshold"], interactive=False)
    fig = gr.Plot()
    btn.click(predict_upload, files, [table, verdict, fig])
    exbtn.click(predict_example, ex, [table, verdict, fig])
    gr.Markdown(
        f"Model: [{REPO}](https://huggingface.co/{REPO}) · test fold macro AUROC "
        f"{CFG['test_fold_10']['macro_auroc']:.3f} · thresholds chosen on the validation fold · "
        "code: [ptbxl-ecg-superclass-benchmark](https://github.com/MHKhan50/ptbxl-ecg-superclass-benchmark) · "
        "data: PTB-XL 1.0.3 (Wagner et al., 2020; CC BY 4.0).")

if __name__ == "__main__":
    demo.launch()
