"""Multi-label evaluation metrics and confidence-based abstention analysis."""
import numpy as np
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             precision_score, recall_score)

def evaluate_multilabel(y_true, probabilities, thresholds=None):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    thresholds = np.full(y_true.shape[1], 0.5) if thresholds is None else np.asarray(thresholds)
    pred = (probabilities >= thresholds).astype(int)
    per_auc, per_ap = {}, {}
    for i in range(y_true.shape[1]):
        name = ["NORM", "MI", "STTC", "CD", "HYP"][i]
        try:
            per_auc[name] = float(roc_auc_score(y_true[:, i], probabilities[:, i]))
        except ValueError:
            per_auc[name] = None
        try:
            per_ap[name] = float(average_precision_score(y_true[:, i], probabilities[:, i]))
        except ValueError:
            per_ap[name] = None
    aucs = [v for v in per_auc.values() if v is not None]
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "per_label_auroc": per_auc, "per_label_average_precision": per_ap,
        "thresholds": thresholds.tolist()
    }

def choose_thresholds(y_true, probabilities):
    """Choose per-label F1-optimal thresholds on validation data only."""
    thresholds = []
    for i in range(y_true.shape[1]):
        candidates = np.linspace(0.1, 0.9, 17)
        scores = [f1_score(y_true[:, i], probabilities[:, i] >= t, zero_division=0)
                  for t in candidates]
        thresholds.append(float(candidates[int(np.argmax(scores))]))
    return np.asarray(thresholds)

def abstention_summary(probabilities, confidence_threshold=0.8):
    """Report record-level coverage; this heuristic is not a clinical safety guarantee."""
    p = np.asarray(probabilities)
    confidence = np.maximum(p, 1.0 - p).mean(axis=1)
    accepted = confidence >= confidence_threshold
    return {"confidence_threshold": float(confidence_threshold),
            "coverage": float(accepted.mean()),
            "accepted_records": int(accepted.sum()),
            "abstained_records": int((~accepted).sum())}
