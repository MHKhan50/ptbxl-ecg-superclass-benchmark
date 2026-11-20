#!/usr/bin/env python3
"""Upload the released checkpoint (model repo) and the demo (Space). Log in first with `hf auth login`."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="path to resnet_s23_best.pt")
    parser.add_argument("--model-repo", default="MHkhan50/ptbxl-ecg-resnet")
    parser.add_argument("--space", default="MHkhan50/ecg-superclass-check")
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.model_repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=args.model_repo, folder_path=str(ROOT / "hf" / "model"),
                      commit_message="Model card, config and network definition")
    api.upload_file(path_or_fileobj=args.weights, path_in_repo="resnet_s23_best.pt", repo_id=args.model_repo,
                    commit_message="ResNet seed 23 checkpoint (lowest validation loss of five seeds)")
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(ROOT / "hf" / "space"),
                      commit_message="ECG Superclass Check demo")
    print("https://huggingface.co/" + args.model_repo)
    print("https://huggingface.co/spaces/" + args.space)


if __name__ == "__main__":
    main()
