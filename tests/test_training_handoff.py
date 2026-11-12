from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from worldline.learning.handoff import export_handoff


def sample():
    return {
        "id": "teacher-1",
        "metadata": {"task_id": "task", "task_version": "v1", "seed": 7, "split": "train"},
        "messages": [
            {"role": "user", "content": "fill the form"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": "act", "arguments": '{"action": {"op": "click"}}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call", "content": "submitted"},
            {"role": "assistant", "content": "done"},
        ],
    }


class TrainingHandoffTest(unittest.TestCase):
    def test_llama_factory_roles_tools_and_portable_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data.jsonl"
            data.write_text(json.dumps(sample()) + "\n")
            manifest = export_handoff(data, root / "lf", framework="llama-factory")
            row = json.loads((root / "lf/data/train.jsonl").read_text())
            self.assertEqual(
                [m["from"] for m in row["conversations"]], ["human", "function_call", "observation", "gpt"]
            )
            self.assertEqual(sum(m["value"].count("<image>") for m in row["conversations"]), len(row["images"]))
            self.assertTrue((root / "lf" / row["images"][0]).is_file())
            self.assertEqual(manifest["records"], {"train": 1})
            self.assertTrue((root / "lf/dataset_info.json").is_file())
            self.assertTrue((root / "lf/llama_factory_sft.yaml").is_file())

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "install handoff extra")
    def test_verl_sft_roundtrip_and_rl_prompt_has_no_teacher_actions(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data.jsonl"
            data.write_text(json.dumps(sample()) + "\n")
            export_handoff(data, root / "verl", framework="verl")
            sft = pq.read_table(root / "verl/data/train.parquet").to_pylist()[0]
            rl = pq.read_table(root / "verl/data/train-rl.parquet").to_pylist()[0]
            self.assertEqual(json.loads(sft["messages"])[2]["tool_calls"][0]["function"]["name"], "act")
            self.assertTrue(all(m["role"] != "assistant" for m in rl["prompt"]))
            self.assertEqual(rl["extra_info"]["seed"], 7)
            self.assertNotIn("reward", json.loads(rl["reward_model"]["ground_truth"]))
            self.assertTrue((root / "verl/adapters/verl_sft_dataset.py").is_file())

    def test_bad_tool_sequences_are_rejected_instead_of_silently_trained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = sample()
            row["messages"][3]["tool_call_id"] = "wrong"
            data = root / "data.jsonl"
            data.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "no valid training"):
                export_handoff(data, root / "lf", framework="llama-factory")
