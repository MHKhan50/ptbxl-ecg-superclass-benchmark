"""Command-line interface."""
import argparse, json
from pathlib import Path
from .data import download_ptbxl, prepare_ptbxl
from .train import run_training

def main():
    parser = argparse.ArgumentParser(prog="cardionet")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download"); d.add_argument("--output", default="data/raw")
    p = sub.add_parser("prepare"); p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--output-dir", default="data/processed"); p.add_argument("--sampling-rate", type=int, choices=[100, 500], default=100)
    t = sub.add_parser("train"); t.add_argument("--config", default="configs/default.yaml")
    t.add_argument("--model", choices=["cnn", "resnet"], default="resnet"); t.add_argument("--epochs", type=int)
    t.add_argument("--checkpoint-dir", default="checkpoints")
    e = sub.add_parser("evaluate"); e.add_argument("--config", default="configs/default.yaml")
    e.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "download":
        print(download_ptbxl(args.output))
    elif args.command == "prepare":
        print(json.dumps(prepare_ptbxl(args.data_dir, args.output_dir, args.sampling_rate), indent=2))
    elif args.command == "train":
        run_training(args.config, args.model, args.epochs, args.checkpoint_dir)
    elif args.command == "evaluate":
        raise NotImplementedError("v0.1.0 evaluation is performed after training and saved as *_metrics.json.")
if __name__ == "__main__":
    main()
