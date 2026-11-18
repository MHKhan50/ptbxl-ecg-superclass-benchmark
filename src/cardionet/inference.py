import torch
from .models import build_model

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(ckpt["model_name"], channels=tuple(cfg["model"]["channels"]),
                        dropout=cfg["model"]["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def predict(model, signal):
    """signal shape: (12, samples), already normalized with training statistics."""
    x = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return torch.sigmoid(model(x)).squeeze(0).numpy()
