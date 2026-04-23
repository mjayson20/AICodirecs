"""
evaluate.py

Evaluate base vs fine-tuned model on the eval split.
Metrics: exact match, token-level accuracy, BLEU-4.

Usage:
    # Evaluate base model
    python evaluate.py --dataset data/dataset_eval.jsonl --model_name Salesforce/codet5-base

    # Evaluate fine-tuned model
    python evaluate.py --dataset data/dataset_eval.jsonl --model_name ./checkpoints/final
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def exact_match(pred: str, target: str) -> bool:
    return pred.strip() == target.strip()


def token_accuracy(pred_tokens: list[int], target_tokens: list[int]) -> float:
    """Fraction of target tokens correctly predicted (positional)."""
    if not target_tokens:
        return 0.0
    correct = sum(p == t for p, t in zip(pred_tokens, target_tokens))
    return correct / len(target_tokens)


def bleu_score(pred: str, target: str) -> float:
    """Simple BLEU-4 approximation using token n-grams."""
    from collections import Counter
    import math

    def ngrams(tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    pred_tokens = pred.strip().split()
    ref_tokens = target.strip().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    score = 1.0
    for n in range(1, 5):
        pred_ng = Counter(ngrams(pred_tokens, n))
        ref_ng = Counter(ngrams(ref_tokens, n))
        if not pred_ng:
            return 0.0
        clipped = sum(min(c, ref_ng[ng]) for ng, c in pred_ng.items())
        precision = clipped / sum(pred_ng.values())
        if precision == 0:
            return 0.0
        score *= precision

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
    return bp * (score ** 0.25)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model_name: str, dataset_path: str, num_samples: int, device: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    records = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if num_samples > 0:
        records = records[:num_samples]

    print(f"Evaluating on {len(records)} samples...")

    em_scores = []
    tok_acc_scores = []
    bleu_scores = []

    for record in tqdm(records):
        inp = record.get("input", "").strip()
        target = record.get("output", "").strip()

        if not inp or not target:
            continue

        inputs = tokenizer(
            inp,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LENGTH,
                num_beams=2,
                early_stopping=True,
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        em_scores.append(int(exact_match(pred, target)))
        tok_acc_scores.append(
            token_accuracy(
                tokenizer.encode(pred, add_special_tokens=False),
                tokenizer.encode(target, add_special_tokens=False),
            )
        )
        bleu_scores.append(bleu_score(pred, target))

    n = len(em_scores)
    print(f"\n{'='*50}")
    print(f"Model          : {model_name}")
    print(f"Samples        : {n}")
    print(f"Exact Match    : {sum(em_scores)/n*100:.2f}%")
    print(f"Token Accuracy : {sum(tok_acc_scores)/n*100:.2f}%")
    print(f"BLEU-4         : {sum(bleu_scores)/n:.4f}")
    print(f"{'='*50}\n")

    return {
        "model": model_name,
        "samples": n,
        "exact_match": sum(em_scores) / n,
        "token_accuracy": sum(tok_acc_scores) / n,
        "bleu4": sum(bleu_scores) / n,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset_eval.jsonl")
    parser.add_argument("--model_name", default="Salesforce/codet5-base")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to fine-tuned model to compare against base",
    )
    args = parser.parse_args()

    base_results = evaluate(args.model_name, args.dataset, args.num_samples, args.device)

    if args.compare:
        ft_results = evaluate(args.compare, args.dataset, args.num_samples, args.device)

        print("\nComparison (fine-tuned vs base):")
        for metric in ["exact_match", "token_accuracy", "bleu4"]:
            base_val = base_results[metric]
            ft_val = ft_results[metric]
            delta = ft_val - base_val
            sign = "+" if delta >= 0 else ""
            print(f"  {metric:20s}: {base_val:.4f} → {ft_val:.4f}  ({sign}{delta:.4f})")
