"""
preprocess.py

Converts raw collected functions into input/output pairs for fine-tuning.

Strategy:
  - Input  : function signature + first N lines (partial function)
  - Output : remaining lines (completion)

Usage:
    python preprocess.py \
        --input  ../model/data/raw.jsonl \
        --output ../model/data/dataset.jsonl \
        --split_ratio 0.9
"""

import argparse
import json
import random
import re
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def remove_comments(source: str) -> str:
    """Remove inline # comments but keep docstrings."""
    lines = []
    for line in source.splitlines():
        # Remove inline comments (but not inside strings — simple heuristic)
        stripped = re.sub(r"\s*#.*$", "", line)
        lines.append(stripped)
    return "\n".join(lines)


def normalize_whitespace(source: str) -> str:
    """Collapse multiple blank lines into one."""
    return re.sub(r"\n{3,}", "\n\n", source).strip()


def is_valid_function(source: str) -> bool:
    """Basic quality filter."""
    lines = source.splitlines()
    if len(lines) < 4:
        return False
    # Must have a def line
    if not any(line.strip().startswith(("def ", "async def ")) for line in lines):
        return False
    # Must have a body beyond just `pass` or `...`
    body_lines = [l.strip() for l in lines[1:] if l.strip() not in ("", "pass", "...")]
    if len(body_lines) < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def make_pairs(source: str, min_prefix_lines: int = 2) -> list[dict]:
    """
    Generate multiple (input, output) pairs from a single function by
    splitting at different points. This augments the dataset.
    """
    lines = source.splitlines()
    pairs = []

    # We generate 2–3 split points per function
    total = len(lines)
    split_points = []

    # Always include a split at ~30% and ~60% of the function
    for frac in [0.3, 0.6]:
        idx = max(min_prefix_lines, int(total * frac))
        if idx < total - 1:
            split_points.append(idx)

    for split_idx in split_points:
        prefix = "\n".join(lines[:split_idx])
        completion = "\n".join(lines[split_idx:])

        # Skip if completion is trivially short
        if len(completion.strip()) < 10:
            continue

        pairs.append(
            {
                "input": prefix,
                "output": completion,
            }
        )

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess(input_path: str, output_path: str, split_ratio: float, strip_comments: bool):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = output_path.parent / (output_path.stem + "_train.jsonl")
    eval_path = output_path.parent / (output_path.stem + "_eval.jsonl")

    raw_records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    print(f"Loaded {len(raw_records)} raw functions")

    all_pairs = []
    skipped = 0

    for record in tqdm(raw_records, desc="Processing"):
        source = record.get("source", "")

        if strip_comments:
            source = remove_comments(source)

        source = normalize_whitespace(source)

        if not is_valid_function(source):
            skipped += 1
            continue

        pairs = make_pairs(source)
        for pair in pairs:
            pair["repo"] = record.get("repo", "")
            pair["name"] = record.get("name", "")
            all_pairs.append(pair)

    print(f"Generated {len(all_pairs)} pairs (skipped {skipped} invalid functions)")

    # Shuffle and split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * split_ratio)
    train_pairs = all_pairs[:split_idx]
    eval_pairs = all_pairs[split_idx:]

    # Write combined output
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # Write train/eval splits
    with open(train_path, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nOutput:")
    print(f"  Combined : {output_path}  ({len(all_pairs)} pairs)")
    print(f"  Train    : {train_path}  ({len(train_pairs)} pairs)")
    print(f"  Eval     : {eval_path}  ({len(eval_pairs)} pairs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../model/data/raw.jsonl")
    parser.add_argument("--output", default="../model/data/dataset.jsonl")
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument(
        "--strip_comments",
        action="store_true",
        help="Remove inline comments from functions",
    )
    args = parser.parse_args()

    preprocess(args.input, args.output, args.split_ratio, args.strip_comments)
