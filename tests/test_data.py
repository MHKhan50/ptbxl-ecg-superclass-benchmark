import numpy as np
import pandas as pd
from cardionet.data import map_superclasses, LABELS

def test_superclass_mapping():
    statements = pd.DataFrame({
        "diagnostic_class": ["NORM", "MI", "CD", "OTHER"],
        "diagnostic": [1, 1, 1, 1]
    }, index=["NORM", "IMI", "LBBB", "X"])
    got = map_superclasses({"NORM": 1, "IMI": 1, "LBBB": 1, "X": 1}, statements)
    assert got.tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]
    assert len(LABELS) == 5


def test_write_normalisation(tmp_path):
    import json
    from cardionet.data import write_normalisation
    mean = np.arange(12, dtype=np.float64).reshape(1, 12, 1)
    std = np.ones((1, 12, 1))
    np.savez_compressed(tmp_path / "ptbxl_100hz.npz", mean=mean, std=std)
    stats = write_normalisation(tmp_path, None, tmp_path / "out" / "normalisation.json")
    saved = json.loads((tmp_path / "out" / "normalisation.json").read_text())
    assert saved == stats
    assert saved["mean"] == [float(i) for i in range(12)] and len(saved["std"]) == 12
