"""Compare trained models over seeds: test AUROC with patient-level bootstrap intervals, and
selective prediction (how macro F1 changes when low-confidence records are set aside)."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
BLUE, YELLOW, RED, GREEN = "#1F4E9C", "#E3A800", "#C8102E", "#5FA65A"


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1])]))


def confidence(p: np.ndarray) -> np.ndarray:
    """Record-level confidence: mean distance of each label probability from 0.5, scaled to [0.5, 1]."""
    return np.maximum(p, 1.0 - p).mean(axis=1)


def selective_f1(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, coverage: float) -> float:
    """Macro F1 on the most confident ``coverage`` share of records (validation thresholds)."""
    keep = max(1, int(round(coverage * len(y))))
    order = np.argsort(-confidence(p), kind="stable")[:keep]
    pred = (p[order] >= thresholds).astype(int)
    return float(f1_score(y[order].astype(int), pred, average="macro", zero_division=0))


def patient_bootstrap(groups: np.ndarray, stat, n_boot: int = 1000, seed: int = 7) -> List[float]:
    """Percentile 95% interval, resampling patients (all records of a patient move together)."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.where(inverse == k)[0] for k in range(len(unique))]
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([members[k] for k in pick])
        try:
            values.append(stat(idx))
        except ValueError:
            continue
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def load_runs(checkpoint_dir: str | Path) -> Dict[str, Dict[int, dict]]:
    runs: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for path in sorted(Path(checkpoint_dir).glob("*_s*_predictions.npz")):
        match = re.match(r"(.+)_s(\d+)_predictions\.npz", path.name)
        if match:
            data = np.load(path)
            runs[match.group(1)][int(match.group(2))] = {k: data[k] for k in data.files}
    if not runs:
        raise FileNotFoundError("No *_s<seed>_predictions.npz files in " + str(checkpoint_dir))
    return runs


def analyse(checkpoint_dir: str | Path, output: str | Path, processed_dir: str | Path) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(checkpoint_dir)
    archive = np.load(Path(processed_dir) / "ptbxl_100hz.npz")
    patients = archive["patients"]

    per_model = {}
    ensembles = {}
    for model, seeds in runs.items():
        first = seeds[min(seeds)]
        y, test_index = first["test_y"].astype(int), first["test_index"]
        groups = patients[test_index]
        seed_auroc = {s: macro_auroc(r["test_y"], r["test_p"]) for s, r in seeds.items()}
        seed_label_auroc = {s: [float(roc_auc_score(r["test_y"][:, i], r["test_p"][:, i])) for i in range(5)]
                            for s, r in seeds.items()}
        seed_selective = {s: {str(c): selective_f1(r["test_y"], r["test_p"], r["thresholds"], c) for c in COVERAGES}
                          for s, r in seeds.items()}
        # Seed ensemble: mean probability over seeds, mean validation thresholds.
        p_mean = np.mean([r["test_p"] for r in seeds.values()], axis=0)
        th_mean = np.mean([r["thresholds"] for r in seeds.values()], axis=0)
        ensembles[model] = (y, p_mean, groups)
        values = np.array(list(seed_auroc.values()))
        per_model[model] = {
            "seeds": sorted(seeds),
            "macro_auroc_per_seed": seed_auroc,
            "macro_auroc_mean": float(values.mean()),
            "macro_auroc_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "per_label_auroc_mean": dict(zip(LABELS, np.mean(list(seed_label_auroc.values()), axis=0).round(4).tolist())),
            "ensemble_macro_auroc": macro_auroc(y, p_mean),
            "ensemble_macro_auroc_ci95": patient_bootstrap(groups, lambda idx: macro_auroc(y[idx], p_mean[idx])),
            "selective_macro_f1_mean": {str(c): float(np.mean([v[str(c)] for v in seed_selective.values()])) for c in COVERAGES},
            "ensemble_selective_macro_f1": {str(c): selective_f1(y, p_mean, th_mean, c) for c in COVERAGES},
        }

    comparison = {}
    if {"cnn", "resnet"} <= set(ensembles):
        y, p_res, groups = ensembles["resnet"]
        _, p_cnn, _ = ensembles["cnn"]
        diff = macro_auroc(y, p_res) - macro_auroc(y, p_cnn)
        comparison["resnet_minus_cnn_ensemble_macro_auroc"] = {
            "value": diff,
            "ci95": patient_bootstrap(groups, lambda idx: macro_auroc(y[idx], p_res[idx]) - macro_auroc(y[idx], p_cnn[idx])),
        }
        common = sorted(set(runs["cnn"]) & set(runs["resnet"]))
        comparison["seeds_where_resnet_beats_cnn"] = int(sum(
            per_model["resnet"]["macro_auroc_per_seed"][s] > per_model["cnn"]["macro_auroc_per_seed"][s] for s in common))
        comparison["paired_seeds"] = len(common)

    summary = {"models": per_model, "comparison": comparison, "test_records": int(len(ensembles[next(iter(ensembles))][0])),
               "bootstrap": "1000 patient-level resamples, percentile 95% interval"}
    (out / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, out / "analysis.md")
    _figures(summary, out)
    return {"summary": summary}


def _write_markdown(summary: dict, path: Path) -> None:
    lines = ["| Model | Seeds | Macro AUROC, mean ± SD over seeds | Seed-ensemble macro AUROC [95% CI] | Macro F1 at 100% / 80% / 50% coverage |",
             "|---|---|---|---|---|"]
    for model, m in summary["models"].items():
        lo, hi = m["ensemble_macro_auroc_ci95"]
        sel = m["ensemble_selective_macro_f1"]
        lines.append("| {} | {} | {:.4f} ± {:.4f} | {:.4f} [{:.4f}, {:.4f}] | {:.3f} / {:.3f} / {:.3f} |".format(
            model, len(m["seeds"]), m["macro_auroc_mean"], m["macro_auroc_sd"], m["ensemble_macro_auroc"], lo, hi,
            sel["1.0"], sel["0.8"], sel["0.5"]))
    c = summary["comparison"].get("resnet_minus_cnn_ensemble_macro_auroc")
    if c:
        lines += ["", "ResNet minus CNN (seed ensembles): {:+.4f} [{:+.4f}, {:+.4f}]; ResNet higher in {} of {} seeds.".format(
            c["value"], c["ci95"][0], c["ci95"][1], summary["comparison"]["seeds_where_resnet_beats_cnn"],
            summary["comparison"]["paired_seeds"])]
    path.write_text("\n".join(lines) + "\n")


def _figures(summary: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colours = {"cnn": YELLOW, "resnet": BLUE}
    names = {"cnn": "1D CNN", "resnet": "1D ResNet"}
    models = [m for m in ("cnn", "resnet") if m in summary["models"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    width = 0.38
    for k, model in enumerate(models):
        vals = [summary["models"][model]["per_label_auroc_mean"][label] for label in LABELS]
        axes[0].bar([i + (k - 0.5) * width for i in range(5)], vals, width * 0.92, color=colours[model], label=names[model])
        sel = summary["models"][model]["ensemble_selective_macro_f1"]
        axes[1].plot([100 * c for c in COVERAGES], [sel[str(c)] for c in COVERAGES], marker="o", lw=2,
                     color=colours[model], label=names[model])
    axes[0].set_xticks(range(5)); axes[0].set_xticklabels(LABELS); axes[0].set_ylim(0.75, 1.0)
    axes[0].set_ylabel("Test AUROC (mean over seeds)")
    axes[1].set_xlabel("Records kept, most confident first (%)"); axes[1].set_ylabel("Macro F1 on kept records")
    axes[1].invert_xaxis()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_auroc_selective.png", dpi=200)
    fig.savefig(out / "fig_auroc_selective.pdf")
    plt.close(fig)
