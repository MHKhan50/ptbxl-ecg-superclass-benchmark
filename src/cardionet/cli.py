"""Command-line interface."""
import argparse, json
from pathlib import Path
from .data import download_ptbxl, prepare_ptbxl
from .train import run_evaluation, run_training

def main():
    parser = argparse.ArgumentParser(prog="cardionet")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download"); d.add_argument("--output", default="data/raw")
    p = sub.add_parser("prepare"); p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--output-dir", default="data/processed"); p.add_argument("--sampling-rate", type=int, choices=[100, 500], default=100)
    t = sub.add_parser("train"); t.add_argument("--config", default="configs/default.yaml")
    t.add_argument("--model", choices=["cnn", "resnet"], default="resnet"); t.add_argument("--epochs", type=int)
    t.add_argument("--checkpoint-dir", default="checkpoints"); t.add_argument("--seed", type=int)
    a = sub.add_parser("analyse"); a.add_argument("--checkpoint-dir", default="checkpoints")
    a.add_argument("--output", default="reports"); a.add_argument("--processed-dir", default="data/processed")
    e = sub.add_parser("evaluate"); e.add_argument("--config", default="configs/default.yaml")
    e.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "download":
        print(download_ptbxl(args.output))
    elif args.command == "prepare":
        print(json.dumps(prepare_ptbxl(args.data_dir, args.output_dir, args.sampling_rate), indent=2))
    elif args.command == "train":
        run_training(args.config, args.model, args.epochs, args.checkpoint_dir, args.seed)
    elif args.command == "evaluate":
        run_evaluation(args.config, args.checkpoint)
    elif args.command == "analyse":
        from .analysis import analyse
        print(json.dumps(analyse(args.checkpoint_dir, args.output, args.processed_dir)["summary"], indent=2))
if __name__ == "__main__":
    main()
