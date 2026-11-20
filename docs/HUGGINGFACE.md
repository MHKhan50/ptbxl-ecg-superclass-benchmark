# Hugging Face release

- Model: [MHkhan50/ptbxl-ecg-resnet](https://huggingface.co/MHkhan50/ptbxl-ecg-resnet): ResNet seed 23 (lowest validation loss of five seeds), `config.json` with lead order, training-fold normalisation, thresholds and abstention rule.
- Demo: [MHkhan50/ecg-superclass-check](https://huggingface.co/spaces/MHkhan50/ecg-superclass-check).

Sources are in `hf/model/` and `hf/space/`. The normalisation statistics come from `notebooks/ptbxl_normalisation_stats.ipynb` (copy in `results/kaggle-2026-11-19/normalisation.json`). Upload:

```bash
pip install huggingface_hub
hf auth login                      # write token, entered in the terminal only
python scripts/upload_hf.py --weights path/to/resnet_s23_best.pt
```

No PTB-XL signals are uploaded. The demo's example records are read from PhysioNet at run time.
