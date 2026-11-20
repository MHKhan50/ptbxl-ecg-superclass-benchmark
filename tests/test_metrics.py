import numpy as np
from cardionet.metrics import evaluate_multilabel, choose_thresholds, abstention_summary

def test_metrics_are_finite_and_multilabel():
    y = np.array([[1,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1],[0,1,1,0,0]])
    p = np.array([[.9,.1,.8,.1,.2],[.2,.8,.2,.9,.1],[.8,.2,.3,.1,.8],[.1,.9,.7,.2,.1]])
    result = evaluate_multilabel(y, p)
    assert 0 <= result["macro_f1"] <= 1
    assert len(result["per_label_auroc"]) == 5

def test_thresholds_have_one_per_label():
    y = np.tile([[1,0,1,0,0],[0,1,0,1,1]], (10,1))
    p = y * .8 + .1
    assert choose_thresholds(y,p).shape == (5,)

def test_abstention_coverage_bounds():
    result = abstention_summary(np.array([[.99,.01,.99,.01,.99],[.5]*5]), .8)
    assert 0 <= result["coverage"] <= 1


def test_expected_calibration_error():
    import numpy as np
    from cardionet.metrics import expected_calibration_error
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert abs(expected_calibration_error(y, np.full(4, 0.5)) - 0.0) < 1e-12
    assert abs(expected_calibration_error(np.ones(4), np.full(4, 0.5)) - 0.5) < 1e-12
