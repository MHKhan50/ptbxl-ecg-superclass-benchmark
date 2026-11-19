"""PTB-XL download, preprocessing and dataset utilities."""
from __future__ import annotations
import ast
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

PHYSIONET_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-1.0.3.zip"
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_ptbxl(output_dir: str | Path) -> Path:
    """Download PTB-XL v1.0.3 from PhysioNet using WFDB's database downloader.

    The official database is approximately 1.7 GB compressed / 3.0 GB expanded.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = next(iter(out.glob("**/ptbxl_database.csv")), None)
    if existing is not None:
        return existing.parent
    try:
        import wfdb
        wfdb.dl_database("ptb-xl/1.0.3", dl_dir=str(out), keep_subdirs=True)
    except Exception as exc:
        raise RuntimeError(
            "PTB-XL download failed. Check internet access and retry. "
            "Official dataset: https://physionet.org/content/ptb-xl/1.0.3/"
        ) from exc
    candidates = list(out.glob("**/ptbxl_database.csv"))
    if not candidates:
        raise FileNotFoundError("Download finished but ptbxl_database.csv was not found.")
    return candidates[0].parent

def _scp_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    return ast.literal_eval(value) if isinstance(value, str) else {}

def map_superclasses(scp_codes: dict, statements: pd.DataFrame) -> np.ndarray:
    """Map weighted diagnostic SCP codes to five superclass multi-hot targets."""
    targets = np.zeros(len(LABELS), dtype=np.float32)
    for code in scp_codes:
        if code in statements.index:
            row = statements.loc[code]
            superclass = row.get("diagnostic_class")
            if bool(row.get("diagnostic", 0)) and superclass in LABELS:
                targets[LABELS.index(superclass)] = 1.0
    return targets

def prepare_ptbxl(data_dir: str | Path, output_dir: str | Path, sampling_rate: int = 100,
                  seed: int = 17) -> dict:
    """Read PTB-XL, use official patient-level folds, and save compressed NumPy arrays."""
    import wfdb
    root, out = Path(data_dir), Path(output_dir)
    db_path = next(iter(root.glob("**/ptbxl_database.csv")), None)
    scp_path = next(iter(root.glob("**/scp_statements.csv")), None)
    if db_path is None or scp_path is None:
        raise FileNotFoundError("Expected ptbxl_database.csv and scp_statements.csv under data-dir.")
    db = pd.read_csv(db_path, index_col="ecg_id")
    statements = pd.read_csv(scp_path, index_col=0)
    db["scp_codes"] = db["scp_codes"].apply(_scp_dict)
    db["targets"] = db["scp_codes"].apply(lambda x: map_superclasses(x, statements))
    db = db[db["targets"].apply(lambda x: float(np.sum(x)) > 0)].copy()
    out.mkdir(parents=True, exist_ok=True)

    # Use disk-backed arrays so CPU-only laptops do not need the full waveform
    # tensor plus intermediate copies resident in RAM.
    key = "filename_lr" if sampling_rate == 100 else "filename_hr"
    n_records = len(db)
    n_samples = 1000 if sampling_rate == 100 else 5000
    x_path = out / "signals_unscaled.npy"
    x = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32,
                                  shape=(n_records, 12, n_samples))
    y = np.zeros((n_records, len(LABELS)), dtype=np.float32)
    folds = np.zeros(n_records, dtype=np.int16)
    patients = np.zeros(n_records, dtype=np.int64)
    for i, (_, row) in enumerate(db.iterrows()):
        record = root / str(row[key])
        signal, _ = wfdb.rdsamp(str(record))
        if signal.shape[0] != n_samples or signal.shape[1] != 12:
            raise ValueError(f"Unexpected ECG shape for {record}: {signal.shape}")
        x[i] = signal.astype(np.float32).T
        y[i] = row["targets"]
        folds[i] = int(row["strat_fold"])
        patients[i] = int(row["patient_id"])
    x.flush()
    train_mask = np.isin(folds, [1,2,3,4,5,6,7,8])
    val_mask, test_mask = folds == 9, folds == 10
    # Assert no patient overlap across official folds.
    patient_sets = [set(patients[m].tolist()) for m in (train_mask, val_mask, test_mask)]
    if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Patient overlap detected between official folds.")
    # Compute channel-wise training statistics in chunks.
    sums = np.zeros(12, dtype=np.float64)
    sums_sq = np.zeros(12, dtype=np.float64)
    count = 0
    for start_idx in np.where(train_mask)[0]:
        sample = np.asarray(x[start_idx], dtype=np.float64)
        sums += sample.sum(axis=1)
        sums_sq += np.square(sample).sum(axis=1)
        count += sample.shape[1]
    mean = (sums / count).reshape(1, 12, 1)
    variance = np.maximum(sums_sq / count - np.square(sums / count), 0.0)
    std = np.sqrt(variance).reshape(1, 12, 1)
    std[std < 1e-6] = 1.0
    normalized_path = out / "signals_normalized.npy"
    xn = np.lib.format.open_memmap(normalized_path, mode="w+", dtype=np.float32,
                                   shape=x.shape)
    for i in range(n_records):
        xn[i] = (np.asarray(x[i]) - mean[0]) / std[0]
    xn.flush()
    del x
    x_path.unlink(missing_ok=True)
    np.savez_compressed(out / "ptbxl_100hz.npz", y=y, folds=folds,
                        patients=patients, mean=mean, std=std)
    # Keep the large normalized waveform matrix as a separate mmap-able .npy.
    metadata = {"signals_file": "signals_normalized.npy"}
    metadata = {"signals_file": "signals_normalized.npy", "records_after_label_filter": int(n_records), "shape": [n_records, 12, n_samples],
                "labels": LABELS, "sampling_rate": sampling_rate,
                "train_records": int(train_mask.sum()), "validation_records": int(val_mask.sum()),
                "test_records": int(test_mask.sum()), "seed": seed,
                "patient_overlap": False, "dataset": "PTB-XL 1.0.3"}
    (out / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata

class ECGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.x, self.y, self.indices = x, y, indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        import torch
        i = self.indices[index]
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


def write_normalisation(processed_dir: str | Path, data_dir: str | Path | None, output: str | Path) -> dict:
    """Write the training-fold channel mean and std (and the lead order) needed to scale a new ECG."""
    arrays = np.load(Path(processed_dir) / "ptbxl_100hz.npz")
    leads = None
    if data_dir is not None:
        import wfdb
        root = Path(data_dir)
        db_path = next(iter(root.glob("**/ptbxl_database.csv")))
        first = pd.read_csv(db_path)["filename_lr"].iloc[0]
        leads = [str(name) for name in wfdb.rdheader(str(db_path.parent / first)).sig_name]
    stats = {
        "dataset": "PTB-XL 1.0.3",
        "sampling_rate": 100,
        "samples": 1000,
        "folds": [1, 2, 3, 4, 5, 6, 7, 8],
        "units": "mV",
        "leads": leads,
        "mean": [float(v) for v in arrays["mean"].reshape(-1)],
        "std": [float(v) for v in arrays["std"].reshape(-1)],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(stats, indent=2))
    return stats
