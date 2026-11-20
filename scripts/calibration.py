"""Per-label expected calibration error and Brier score from saved test predictions.

Usage: python scripts/calibration.py <dir with *_s<seed>_predictions.npz> <output.json>
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cardionet.metrics import expected_calibration_error  # noqa: E402

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


def main(pred_dir: str, output: str) -> None:
    runs = {}
    for path in sorted(glob.glob(str(Path(pred_dir) / "*_predictions.npz"))):
        match = re.search(r"(cnn|resnet)_s(\d+)_predictions\.npz$", path)
        if match:
            runs.setdefault(match.group(1), []).append(np.load(path))
    result = {"method": "ECE with 15 equal-width bins per label and Brier score, test fold 10", "models": {}}
    for model, preds in runs.items():
        ece = np.array([[expected_calibration_error(d["test_y"][:, i], d["test_p"][:, i]) for i in range(5)] for d in preds])
        brier = np.array([[np.mean((d["test_p"][:, i] - d["test_y"][:, i]) ** 2) for i in range(5)] for d in preds])
        y = preds[0]["test_y"]
        p_mean = np.mean([d["test_p"] for d in preds], axis=0)
        ens = [expected_calibration_error(y[:, i], p_mean[:, i]) for i in range(5)]
        result["models"][model] = {
            "seeds": len(preds),
            "ece_per_label_mean": dict(zip(LABELS, ece.mean(0).round(4).tolist())),
            "ece_per_label_sd": dict(zip(LABELS, ece.std(0, ddof=1).round(4).tolist())),
            "macro_ece_mean": float(ece.mean(1).mean()),
            "macro_ece_sd": float(ece.mean(1).std(ddof=1)),
            "brier_per_label_mean": dict(zip(LABELS, brier.mean(0).round(4).tolist())),
            "macro_brier_mean": float(brier.mean()),
            "ensemble_ece_per_label": dict(zip(LABELS, np.round(ens, 4).tolist())),
            "ensemble_macro_ece": float(np.mean(ens)),
            "test_prevalence": dict(zip(LABELS, y.mean(0).round(4).tolist())),
        }
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
