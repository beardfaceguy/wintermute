#!/usr/bin/env python3
"""
Lightweight GPT-2 LM fine-tune on tiny text for HF export validation.

Args:
  --train: path to train.txt
  --val: path to val.txt
  --out: output directory
  --epochs: number of epochs (default 1)
"""
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_dataset(
        "text",
        data_files={"train": args.train, "validation": args.val},
    )
    # Drop empty/whitespace-only lines to avoid zero-length batches
    data = data.filter(lambda ex: bool(ex["text"] and ex["text"].strip()))

    block_size = 256

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            return_special_tokens_mask=True,
        )

    tok = data.map(tokenize_fn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args_train = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        learning_rate=5e-4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=100,
        fp16=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=tok["train"],
        eval_dataset=tok["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()
