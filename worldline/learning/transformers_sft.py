"""Optional Hugging Face SFT backend with assistant-only next-token loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldline.datasets import _jsonl


def encode_record(tokenizer, record: dict, max_length: int) -> dict:
    """Use the model's chat template, preserve tool calls, mask user/tool tokens.

    Image training requires a model-specific processor/backend. Refuse image
    samples rather than silently flattening them into text or discarding them.
    """
    messages = []
    for message in record["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("multimodal SFT requires a processor-aware external trainer")
            message = dict(message, content="\n".join(part["text"] for part in content))
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported chat role")
        messages.append(message)
    tools = record.get("tools") or None
    tokens, labels = [], []
    for index, message in enumerate(messages):
        current = tokenizer.apply_chat_template(
            messages[: index + 1], tools=tools, tokenize=True, add_generation_prompt=False
        )
        if list(current[: len(tokens)]) != tokens:
            raise ValueError("chat template rewrites previous turns; use a model-specific training backend")
        new_tokens = list(current[len(tokens) :])
        tokens.extend(new_tokens)
        labels.extend(new_tokens if message["role"] == "assistant" else [-100] * len(new_tokens))
    tokens, labels = tokens[:max_length], labels[:max_length]
    if not any(label != -100 for label in labels):
        raise ValueError("record has no assistant supervision after truncation")
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens), "labels": labels}


def run(config: dict) -> None:
    # These imports belong only inside the optional trainer process.
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(config.get("seed", 0))
    model_name = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("model tokenizer must provide a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = config.get("max_length", 4096)

    def dataset(path):
        encoded = [encode_record(tokenizer, row, max_length) for _, row in _jsonl(Path(path))]
        return Dataset.from_list(encoded) if encoded else None

    train_data = dataset(config["train_file"])
    eval_data = dataset(config["validation_file"]) if config.get("validation_file") else None
    if train_data is None:
        raise ValueError("empty training data")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)

    def collate(rows):
        width = max(len(row["input_ids"]) for row in rows)
        result = {}
        for key, padding in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            result[key] = torch.tensor(
                [row[key] + [padding] * (width - len(row[key])) for row in rows], dtype=torch.long
            )
        return result

    arguments = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config.get("epochs", 1),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config.get("learning_rate", 2e-5),
        seed=config.get("seed", 0),
        eval_strategy="epoch" if eval_data is not None else "no",
        save_strategy="epoch",
        report_to="none",
        use_cpu=config.get("use_cpu", False),
    )
    trainer = Trainer(
        model=model, args=arguments, train_dataset=train_data, eval_dataset=eval_data, data_collator=collate
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
