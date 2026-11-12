"""Portable LLaMA-Factory and verl data packages; no trainer imports."""

from __future__ import annotations

import hashlib
import itertools
import json
import base64
import io
from pathlib import Path

from worldline.datasets import _jsonl, _record_split, validate_datasets
from worldline.media import MediaStore
from worldline.run import write_json


def _content(content, media, images):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("message content must be text or multimodal parts")
    parts = []
    for part in content:
        if part.get("type") == "text":
            parts.append(part["text"])
        elif part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                raise ValueError("image handoff requires embedded verified image bytes")
            prefix, encoded = url.split(";base64,", 1)
            from PIL import Image

            try:
                Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).verify()
            except Exception as error:
                raise ValueError("invalid training image bytes") from error
            reference = media.put(encoded, prefix[5:], "data_url")["__worldline_media__"]
            images.append(reference["path"])
            parts.append("<image>")
        else:
            raise ValueError("unsupported multimodal content type")
    return "\n".join(parts)


def normalized_messages(row, media):
    images, messages = [], []
    pending_calls = set()
    for raw in row["messages"]:
        message = dict(raw)
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role")
        message["content"] = _content(message.get("content"), media, images)
        calls = []
        for call in message.get("tool_calls") or []:
            if "function" in call:
                call = dict(call)
            else:
                call = {
                    "id": call.get("tool_call_id"),
                    "type": "function",
                    "function": {"name": call["function_name"], "arguments": json.dumps(call.get("arguments", {}))},
                }
            if (
                role != "assistant"
                or not isinstance(call.get("id"), str)
                or not call["id"]
                or call["id"] in pending_calls
            ):
                raise ValueError("missing or duplicate tool call id")
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                json.loads(arguments)
            else:
                call["function"] = {**call["function"], "arguments": json.dumps(arguments)}
            pending_calls.add(call["id"])
            calls.append(call)
        if calls:
            message["tool_calls"] = calls
        if role == "tool":
            ident = message.get("tool_call_id")
            if ident not in pending_calls:
                raise ValueError("tool response has no matching pending call")
            pending_calls.remove(ident)
        messages.append(message)
    if pending_calls:
        raise ValueError("trajectory ends with unanswered tool calls")
    if not messages or not any(message["role"] == "assistant" for message in messages):
        raise ValueError("SFT trajectory must contain assistant supervision")
    return messages, images


def llama_factory_record(row, media):
    messages, images = normalized_messages(row, media)
    conversations, system, dropped_thoughts = [], [], 0
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "system":
            if conversations:
                raise ValueError("mid-episode system messages require a custom trainer")
            system.append(content)
            continue
        if message.get("tool_calls"):
            dropped_thoughts += int(bool(content))
            content = json.dumps(
                [
                    {"name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"])}
                    for call in message["tool_calls"]
                ],
                ensure_ascii=False,
            )
            tag = "function_call"
        else:
            tag = {"user": "human", "assistant": "gpt", "tool": "observation"}[role]
        if conversations and tag in {"human", "observation"} and conversations[-1]["from"] in {"human", "observation"}:
            conversations[-1]["value"] += "\n" + content
        else:
            conversations.append({"from": tag, "value": content})
    trailing = 0
    while conversations and conversations[-1]["from"] in {"human", "observation"}:
        conversations.pop()
        trailing += 1
    images = images[: sum(message["value"].count("<image>") for message in conversations)]
    for index, message in enumerate(conversations):
        valid = {"human", "observation"} if index % 2 == 0 else {"gpt", "function_call"}
        if message["from"] not in valid:
            raise ValueError("LLaMA-Factory requires alternating input/output roles")
    if len(conversations) % 2:
        raise ValueError("incomplete LLaMA-Factory SFT conversation")
    return {
        "id": row["id"],
        "conversations": conversations,
        "system": "\n".join(system),
        "tools": json.dumps(row.get("tools") or [], ensure_ascii=False),
        "images": images,
        "worldline_metadata": {**row.get("metadata", {}), "trailing_inputs_omitted": trailing},
    }, dropped_thoughts


class ParquetStream:
    def __init__(self, path, schema, batch_size=128):
        import pyarrow.parquet as pq

        self.path, self.schema, self.batch_size, self.rows = path, schema, batch_size, []
        self.writer = pq.ParquetWriter(path, schema)

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self):
        import pyarrow as pa

        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()


def export_handoff(
    train_file: Path,
    output_dir: Path,
    *,
    framework: str,
    validation_file: Path | None = None,
    model: str = "REPLACE_WITH_MODEL",
    template: str = "qwen",
):
    if framework not in {"llama-factory", "verl"}:
        raise ValueError("unsupported training framework")
    inputs = [train_file.resolve()] + ([validation_file.resolve()] if validation_file else [])
    validation = validate_datasets(inputs)
    if not validation["valid"]:
        raise ValueError(f"training handoff validation failed: {validation['errors']}")
    for _, row in _jsonl(inputs[0]):
        if _record_split(row) in {"test", "evaluation", "held_out", "validation"}:
            raise ValueError("held-out records cannot be placed in the training package")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("handoff output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir()
    media = MediaStore(output_dir)
    counts, rejected, thoughts = {}, {}, 0
    dataset_info = {}
    for split, path in zip(("train", "validation"), inputs):
        count, failures = 0, 0
        writers = []
        seen_prompts = set()
        if framework == "llama-factory":
            stream = (output_dir / "data" / f"{split}.jsonl").open("w")
        else:
            import pyarrow as pa

            image_type = pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
            sft_schema = pa.schema(
                [
                    ("id", pa.string()),
                    ("messages", pa.string()),
                    ("tools", pa.string()),
                    ("images", image_type),
                    ("metadata_json", pa.string()),
                ]
            )
            rl_schema = pa.schema(
                [
                    ("data_source", pa.string()),
                    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
                    ("images", image_type),
                    ("ability", pa.string()),
                    ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
                    (
                        "extra_info",
                        pa.struct(
                            [
                                ("task_id", pa.string()),
                                ("task_version", pa.string()),
                                ("seed", pa.int64()),
                                ("split_group", pa.string()),
                                ("split", pa.string()),
                                ("tools_json", pa.string()),
                            ]
                        ),
                    ),
                ]
            )
            writers = [
                ParquetStream(output_dir / "data" / f"{split}.parquet", sft_schema),
                ParquetStream(output_dir / "data" / f"{split}-rl.parquet", rl_schema),
            ]
        rejected_stream = (output_dir / f"rejected-{split}.jsonl").open("w")
        try:
            for _, row in _jsonl(path):
                try:
                    if framework == "llama-factory":
                        converted, lost = llama_factory_record(row, media)
                        thoughts += lost
                        stream.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    else:
                        messages, images = normalized_messages(row, media)
                        image_rows = [{"bytes": (output_dir / image).read_bytes(), "path": None} for image in images]
                        tools = json.dumps(row.get("tools") or [], ensure_ascii=False)
                        metadata = row.get("metadata", {})
                        prompt = [
                            {"role": message["role"], "content": message["content"]}
                            for message in itertools.takewhile(lambda m: m["role"] != "assistant", messages)
                        ]
                        info = {
                            "task_id": metadata["task_id"],
                            "task_version": metadata.get("task_version", "unversioned"),
                            "seed": metadata.get("seed", 0),
                            "split_group": metadata.get("split_group", metadata["task_id"]),
                            "split": split,
                            "tools_json": tools,
                        }
                        writers[0].append(
                            {
                                "id": row["id"],
                                "messages": json.dumps(messages, ensure_ascii=False),
                                "tools": tools,
                                "images": image_rows,
                                "metadata_json": json.dumps(metadata),
                            }
                        )
                        key = hashlib.sha256(json.dumps((prompt, info), sort_keys=True).encode()).hexdigest()
                        if key not in seen_prompts:
                            initial_images = image_rows[: sum(m["content"].count("<image>") for m in prompt)]
                            writers[1].append(
                                {
                                    "data_source": "worldline",
                                    "prompt": prompt,
                                    "images": initial_images,
                                    "ability": "stateful-environment",
                                    "reward_model": {"style": "rule", "ground_truth": json.dumps(info)},
                                    "extra_info": info,
                                }
                            )
                            seen_prompts.add(key)
                    count += 1
                except (ValueError, KeyError, TypeError) as error:
                    failures += 1
                    rejected_stream.write(json.dumps({"id": row.get("id"), "error": str(error)}) + "\n")
        finally:
            rejected_stream.close()
            if framework == "llama-factory":
                stream.close()
            for writer in writers:
                writer.close()
        counts[split], rejected[split] = count, failures
        dataset_info[f"worldline_{split}"] = {
            "file_name": f"data/{split}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools", "images": "images"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
            },
        }
    if not counts.get("train"):
        raise ValueError("no valid training samples; inspect rejected-train.jsonl")
    if framework == "llama-factory":
        write_json(output_dir / "dataset_info.json", dataset_info)
        config = f"model_name_or_path: {json.dumps(model)}\ntemplate: {json.dumps(template)}\nstage: sft\ndo_train: true\nfinetuning_type: lora\ndataset_dir: .\ndataset: worldline_train\noutput_dir: models/student\noverwrite_output_dir: false\nper_device_train_batch_size: 1\ngradient_accumulation_steps: 8\nlearning_rate: 0.00002\nnum_train_epochs: 1\ncutoff_len: 4096\n"
        if counts.get("validation"):
            config += "eval_dataset: worldline_validation\neval_strategy: epoch\n"
        (output_dir / "llama_factory_sft.yaml").write_text(config)
    else:
        (output_dir / "adapters").mkdir()
        (output_dir / "adapters/verl_sft_dataset.py").write_text(
            Path(__file__).with_name("verl_dataset.py").read_text()
        )
        (output_dir / "verl_data.yaml").write_text(
            "data:\n  train_files: data/train.parquet\n  messages_key: messages\n  tools_key: tools\n  image_key: images\n  custom_cls:\n    path: adapters/verl_sft_dataset.py\n    name: WorldlineSFTDataset\n"
        )
    (output_dir / "README.md").write_text(
        "# Worldline training handoff\n\nRun training from this directory so relative image/data paths resolve.\n\n"
        + (
            "Install LLaMA-Factory separately, choose model/template, then run `llamafactory-cli train llama_factory_sft.yaml`.\n"
            "Tool decisions are function_call outputs; simultaneous assistant prose is retained in original Worldline traces and counted as omitted from function-only supervision.\n"
            if framework == "llama-factory"
            else "Install verl separately. SFT Parquet uses the included thin MultiTurnSFTDataset adapter to decode JSON messages and tool schemas. Merge verl_data.yaml into your version's trainer configuration; supply tokenizer/model and multimodal processor when images exist.\n"
            "*-rl.parquet contains initial prompts only, never teacher actions as the prompt. ground_truth is a task identity, not a recorded score. Online RUA/CUA RL additionally requires an environment AgentLoop and verifier reward bridge; recorded rollout rewards cannot score newly generated responses.\n"
        )
    )
    manifest = {
        "schema_version": "worldline.handoff/0.1",
        "framework": framework,
        "records": counts,
        "rejected": rejected,
        "function_only_prose_omissions": thoughts,
        "media_files": media.files_written,
        "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs],
        "files": {
            str(path.relative_to(output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output_dir.rglob("*")
            if path.is_file()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
