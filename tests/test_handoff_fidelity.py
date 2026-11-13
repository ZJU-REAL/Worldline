from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldline.learning.handoff import export_handoff
from worldline.learning.check_package import check_package


class HandoffFidelityTest(unittest.TestCase):
    def test_default_preservation_and_explicit_loss_policies(self):
        row = {
            "id": "sample",
            "metadata": {"task_id": "task", "split": "train"},
            "messages": [
                {"role": "user", "content": "goal"},
                {
                    "role": "assistant",
                    "content": "plan before action",
                    "tool_calls": [{"id": "call", "type": "function", "function": {"name": "act", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call", "content": "feedback"},
                {"role": "assistant", "content": "done"},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data.jsonl"
            data.write_text(json.dumps(row) + "\n")
            preserved = export_handoff(data, root / "preserved", framework="llama-factory")
            self.assertEqual(preserved["function_only_prose_omissions"], 0)
            serialized = json.loads((root / "preserved/data/train.jsonl").read_text())
            calls = json.loads(serialized["conversations"][1]["value"])
            self.assertEqual(calls[0]["_worldline_assistant_prose"], "plan before action")
            self.assertIn("worldline__qwen", (root / "preserved/llama_factory_sft.yaml").read_text())
            with self.assertRaisesRegex(ValueError, "no valid training"):
                export_handoff(data, root / "strict", framework="llama-factory", tool_prose_policy="reject")
            result = export_handoff(data, root / "explicit", framework="llama-factory", tool_prose_policy="omit")
            self.assertEqual(result["function_only_prose_omissions"], 1)
            canonical = json.loads((root / "explicit/data/train-worldline.jsonl").read_text())
            self.assertEqual(canonical["messages"], row["messages"])
            with self.assertRaisesRegex(ValueError, "lossy supervision"):
                check_package(root / "explicit", framework="llama-factory", model="unused")

    def test_package_tampering_is_rejected_before_loading_a_trainer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "id": "x",
                        "metadata": {"task_id": "task", "split": "train"},
                        "messages": [{"role": "user", "content": "goal"}, {"role": "assistant", "content": "answer"}],
                    }
                )
                + "\n"
            )
            export_handoff(data, root / "package", framework="llama-factory")
            (root / "package/data/train.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash/path"):
                check_package(root / "package", framework="llama-factory", model="unused")
