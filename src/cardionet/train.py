"""Training and evaluation orchestration."""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from .data import ECGDataset
from .metrics import evaluate_multilabel, choose_thresholds, abstention_summary
from .models import build_model

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _predict(model, loader, device):
    model.eval(); probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)

def run_training(config_path, model_name="resnet", epochs=None, checkpoint_dir="checkpoints"):
    config = yaml.safe_load(Path(config_path).read_text())
    seed = int(config["training"]["seed"]); set_seed(seed)
    processed = Path(config["data"]["processed_dir"])
    archive = np.load(processed / "ptbxl_100hz.npz")
    x = np.load(processed / "signals_normalized.npy", mmap_mode="r")
    y, folds = archive["y"], archive["folds"]
    train_idx = np.where(np.isin(folds, config["data"]["folds_train"]))[0]
    val_idx = np.where(folds == config["data"]["fold_validation"])[0]
    test_idx = np.where(folds == config["data"]["fold_test"])[0]
    tr = DataLoader(ECGDataset(x, y, train_idx), batch_size=config["training"]["batch_size"],
                    shuffle=True, num_workers=config["training"]["num_workers"])
    va = DataLoader(ECGDataset(x, y, val_idx), batch_size=config["training"]["batch_size"])
    te = DataLoader(ECGDataset(x, y, test_idx), batch_size=config["training"]["batch_size"])
    device = torch.device("cpu")
    model = build_model(model_name, dropout=config["model"]["dropout"],
                        channels=tuple(config["model"]["channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    best_loss, stale = float("inf"), 0
    epochs = int(epochs or config["training"]["epochs"])
    out = Path(checkpoint_dir); out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb); loss.backward()
            optimizer.step(); losses.append(float(loss.item()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for xb, yb in va:
                val_losses.append(float(criterion(model(xb), yb).item()))
        val_loss = float(np.mean(val_losses))
        print(f"epoch={epoch}/{epochs} train_loss={np.mean(losses):.5f} val_loss={val_loss:.5f}")
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "model_name": model_name,
                        "config": config, "labels": ["NORM","MI","STTC","CD","HYP"]},
                       out / f"{model_name}_best.pt")
        else:
            stale += 1
            if stale >= config["training"]["patience"]:
                break
    checkpoint = torch.load(out / f"{model_name}_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    vy, vp = _predict(model, va, device)
    thresholds = choose_thresholds(vy, vp)
    ty, tp = _predict(model, te, device)
    metrics = evaluate_multilabel(ty, tp, thresholds)
    metrics["abstention"] = abstention_summary(tp, 0.8)
    metrics["metadata"] = {"model": model_name, "seed": seed, "epochs_requested": epochs,
                           "elapsed_seconds": round(time.time()-start, 2),
                           "validation_thresholds_selected": True,
                           "test_fold": config["data"]["fold_test"]}
    (out / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def run_evaluation(config_path, checkpoint_path):
    """Re-evaluate a saved checkpoint on the untouched official test fold."""
    config = yaml.safe_load(Path(config_path).read_text())
    processed = Path(config["data"]["processed_dir"])
    archive = np.load(processed / "ptbxl_100hz.npz")
    x = np.load(processed / "signals_normalized.npy", mmap_mode="r")
    y, folds = archive["y"], archive["folds"]
    test_idx = np.where(folds == config["data"]["fold_test"])[0]
    loader = DataLoader(ECGDataset(x, y, test_idx), batch_size=config["training"]["batch_size"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_name"], channels=tuple(config["model"]["channels"]),
                        dropout=config["model"]["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    labels, probabilities = _predict(model, loader, torch.device("cpu"))
    metrics_path = Path(checkpoint_path).with_name(Path(checkpoint_path).stem.replace("_best", "") + "_metrics.json")
    thresholds = np.full(y.shape[1], float(config["training"].get("threshold", 0.5)))
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text())
        thresholds = np.asarray(saved.get("thresholds", thresholds))
    result = evaluate_multilabel(labels, probabilities, thresholds)
    result["abstention"] = abstention_summary(probabilities, 0.8)
    print(json.dumps(result, indent=2))
    return result
