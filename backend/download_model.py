"""
download_model.py

Pre-downloads the model weights so the server starts instantly.
Run this once before starting the backend.

Usage:
    python download_model.py
    python download_model.py --model Salesforce/codet5p-220m
"""

import argparse
import sys

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def download(model_name: str):
    print(f"Downloading tokenizer: {model_name}")
    AutoTokenizer.from_pretrained(model_name)

    print(f"Downloading model weights: {model_name}")
    print("(This may take a few minutes on first run...)")
    AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"\nDone! Model cached locally.")
    print("You can now start the server with: python main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Salesforce/codet5-base",
        help="Hugging Face model ID to download",
    )
    args = parser.parse_args()
    download(args.model)
