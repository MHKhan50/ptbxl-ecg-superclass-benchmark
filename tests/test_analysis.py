import numpy as np

from cardionet.analysis import confidence, macro_auroc, patient_bootstrap, selective_f1


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 5)) < 0.3).astype(int)
    p = np.clip(y * 0.6 + rng.random((n, 5)) * 0.4, 0, 1)
    return y, p


def test_confidence_is_between_half_and_one():
    _, p = _toy()
    c = confidence(p)
    assert c.min() >= 0.5 and c.max() <= 1.0


def test_full_coverage_equals_plain_macro_f1():
    from sklearn.metrics import f1_score
    y, p = _toy()
    th = np.full(5, 0.5)
    assert abs(selective_f1(y, p, th, 1.0) - f1_score(y, (p >= th).astype(int), average="macro", zero_division=0)) < 1e-12


def test_patient_bootstrap_interval_contains_estimate():
    y, p = _toy()
    groups = np.arange(len(y)) // 2
    lo, hi = patient_bootstrap(groups, lambda idx: macro_auroc(y[idx], p[idx]), n_boot=200)
    assert lo <= macro_auroc(y, p) <= hi
