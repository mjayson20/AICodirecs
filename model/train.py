"""
train.py

Fine-tune CodeT5-base (or CodeT5+) on code completion using QLoRA.
Designed to run on Google Colab free T4 GPU or locally on CPU (slow).

Usage:
    python train.py \
        --dataset data/dataset_train.jsonl \
        --eval_dataset data/dataset_eval.jsonl \
        --output ./checkpoints \
        --model_name Salesforce/codet5-base \
        --epochs 3 \
        --batch_size 4
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

set_seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(records: list[dict], tokenizer, max_input: int, max_target: int) -> Dataset:
    inputs, targets = [], []
    for r in records:
        inp = r.get("input", "").strip()
        out = r.get("output", "").strip()
        if inp and out:
            inputs.append(inp)
            targets.append(out)

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input"],
            max_length=max_input,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=batch["output"],
            max_length=max_target,
            truncation=True,
            padding="max_length",
        )
        # Replace padding token id in labels with -100 so loss ignores them
        label_ids = labels["input_ids"]
        label_ids = [
            [(l if l != tokenizer.pad_token_id else -100) for l in lbl]
            for lbl in label_ids
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    raw = Dataset.from_dict({"input": inputs, "output": targets})
    tokenized = raw.map(tokenize, batched=True, remove_columns=["input", "output"])
    return tokenized


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, use_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_4bit and torch.cuda.is_available():
        print("Loading model in 4-bit (QLoRA) mode...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("Loading model in full precision (CPU mode)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


def apply_lora(model, task_type=TaskType.SEQ_2_SEQ_LM):
    lora_config = LoraConfig(
        task_type=task_type,
        r=16,                    # LoRA rank
        lora_alpha=32,
        target_modules=["q", "v"],  # CodeT5 attention projections
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    print(f"Model  : {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output : {args.output}")

    use_4bit = torch.cuda.is_available() and not args.no_quantize
    model, tokenizer = load_model_and_tokenizer(args.model_name, use_4bit)
    model = apply_lora(model)

    # Load datasets
    train_records = load_jsonl(args.dataset)
    print(f"Train samples: {len(train_records)}")

    train_dataset = build_dataset(train_records, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    eval_dataset = None
    if args.eval_dataset and Path(args.eval_dataset).exists():
        eval_records = load_jsonl(args.eval_dataset)
        eval_dataset = build_dataset(eval_records, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
        print(f"Eval samples : {len(eval_records)}")

    # Training arguments
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=3e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_dataset is not None,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        report_to="none",  # No wandb / tensorboard required
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nModel saved to {final_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset_train.jsonl")
    parser.add_argument("--eval_dataset", default="data/dataset_eval.jsonl")
    parser.add_argument("--output", default="./checkpoints")
    parser.add_argument("--model_name", default="Salesforce/codet5-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable 4-bit quantization (use for CPU-only training)",
    )
    args = parser.parse_args()
    train(args)
