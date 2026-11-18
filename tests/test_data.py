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
