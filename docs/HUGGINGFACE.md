# Hugging Face publication checklist

Target repositories (replace namespace if needed):
- Model: `MHKhan50/cardionet-ecg-resnet`
- Dataset card: `MHKhan50/cardionet-ptbxl-benchmark` (metadata/code only unless redistribution is explicitly permitted)
- Demo: Hugging Face Space `cardionet-ecg-demo`

## Prepare a model export
After a successful training run:

```bash
python - <<'PY'
import torch
ckpt = torch.load("checkpoints/resnet_best.pt", map_location="cpu", weights_only=False)
torch.save(ckpt, "cardionet_resnet.pt")
PY
```

Install tools and authenticate:

```bash
pip install -e ".[huggingface]"
hf auth login
```

Create a model repository and upload model card + weights:

```bash
hf repo create MHKhan50/cardionet-ecg-resnet --repo-type model
hf upload MHKhan50/cardionet-ecg-resnet ./cardionet_resnet.pt
hf upload MHKhan50/cardionet-ecg-resnet ./docs/MODEL_CARD.md README.md
hf upload MHKhan50/cardionet-ecg-resnet ./README.md project-README.md
```

Check current Hugging Face CLI syntax before publishing: https://huggingface.co/docs/huggingface_hub/guides/cli

## Before upload
- [ ] Training completed and metrics reviewed.
- [ ] No raw PTB-XL signals or patient-level data included.
- [ ] Model card includes dataset version, split, metrics, intended use and limitations.
- [ ] Verify dataset licence/terms and any redistribution restrictions.
- [ ] Include exact training configuration and software versions.
- [ ] Do not claim clinical readiness or diagnostic accuracy in practice.
