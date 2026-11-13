"""Run this standalone file in the user's actual LLaMA-Factory/verl environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def check_package(package, *, framework, model, template="qwen", expected_version=None, samples=2):
    package = Path(package).resolve()
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest["framework"] != framework or manifest.get("function_only_prose_omissions"):
        raise ValueError("framework mismatch or lossy supervision; inspect handoff manifest")
    import hashlib

    for name, expected in manifest["files"].items():
        path = (package / name).resolve()
        if not path.is_relative_to(package):
            raise ValueError("package hash/path validation failed")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("package hash/path validation failed")
    distribution = "llamafactory" if framework == "llama-factory" else "verl"
    version = importlib.metadata.version(distribution)
    if expected_version and version != expected_version:
        raise ValueError(f"expected {distribution} {expected_version}, found {version}")
    if framework == "llama-factory":
        if manifest.get("tool_prose_policy") == "preserve":
            from importlib import util

            entry = package / "adapters/llama_factory_entry.py"
            spec = util.spec_from_file_location("worldline_lf_entry", entry)
            entry_module = util.module_from_spec(spec)
            spec.loader.exec_module(entry_module)
            template = entry_module.register_template(template)
        from llamafactory.hparams import get_train_args
        from llamafactory.model import load_tokenizer
        from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
        from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq

        model_args, data_args, training_args, _, _ = get_train_args(
            {
                "model_name_or_path": str(model),
                "template": template,
                "stage": "sft",
                "do_train": True,
                "dataset_dir": str(package),
                "dataset": "worldline_train",
                "output_dir": str(package / "check-output"),
                "max_samples": samples,
                "overwrite_cache": True,
                "report_to": "none",
            }
        )
        module = load_tokenizer(model_args)
        chat_template = get_template_and_fix_tokenizer(module["tokenizer"], data_args)
        dataset = get_dataset(chat_template, model_args, data_args, training_args, stage="sft", **module)[
            "train_dataset"
        ]
        collator = MultiModalDataCollatorForSeq2Seq(
            template=chat_template, tokenizer=module["tokenizer"], processor=module.get("processor")
        )
        batch = collator([dataset[i] for i in range(min(samples, len(dataset)))])
        labels = batch["labels"]
        supervised = int((labels != -100).sum())
        visual = any("pixel" in key or "image" in key for key in batch)
    else:
        from importlib import util
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer, AutoProcessor

        path = package / "adapters/verl_sft_dataset.py"
        spec = util.spec_from_file_location("worldline_external_dataset", path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        processor = None
        if manifest.get("training_image_records", manifest.get("media_files")):
            processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        dataset = module.WorldlineSFTDataset(
            parquet_files=[str(package / "data/train.parquet")],
            tokenizer=tokenizer,
            processor=processor,
            config=OmegaConf.create(
                {"messages_key": "messages", "tools_key": "tools", "image_key": "images", "max_length": 4096}
            ),
        )
        rows = [dataset[index] for index in range(min(samples, len(dataset)))]
        supervised = sum(int(row["loss_mask"].sum()) for row in rows)
        visual = any("multi_modal_inputs" in row or any("pixel" in key for key in row) for row in rows)
    if supervised < 1:
        raise ValueError("actual trainer produced no supervised assistant tokens")
    if manifest.get("training_image_records", manifest.get("media_files")) and not visual:
        raise ValueError("actual trainer did not produce multimodal inputs")
    return {
        "framework": framework,
        "version": version,
        "supervised_tokens": supervised,
        "multimodal_inputs": visual,
        "passed": True,
        "weight_update_tested": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--framework", choices=("llama-factory", "verl"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--expected-version")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_package(
        args.package,
        framework=args.framework,
        model=args.model,
        template=args.template,
        expected_version=args.expected_version,
    )
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
