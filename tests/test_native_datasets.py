from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldline.datasets import export_dataset, split_dataset, validate_datasets


def native_result(root, name, *, version="v1", split="train", outcome="success", image=False):
    path = root / name
    (path / "agent").mkdir(parents=True)
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}] if image else "observe"
    (path / "agent/trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "worldline.messages/0.1",
                "messages": [
                    {"role": "user", "content": "reach"},
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": "act"},
                ],
            }
        )
    )
    (path / "task.json").write_text(json.dumps({"seed": 7, "split": split, "metadata": {"split_group": "scene-a"}}))
    (path / "result.normalized.json").write_text(
        json.dumps(
            {
                "task_id": "reach",
                "task_version": version,
                "trial_id": name,
                "outcome": outcome,
                "reward": 1 if outcome == "success" else 0,
                "source": str(path),
                "artifacts": [{"kind": "agent_trace", "path": "agent/trajectory.json"}],
            }
        )
    )


class NativeDatasetTest(unittest.TestCase):
    def test_preserves_multimodal_messages_and_excludes_held_out(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native_result(root, "training", image=True)
            native_result(root, "heldout", split="test")
            result = export_dataset([root], root / "sft.jsonl", format="sft")
            self.assertEqual(result["records"], 1)
            self.assertEqual(result["load_stats"]["held_out"], 1)
            row = json.loads((root / "sft.jsonl").read_text())
            self.assertEqual(row["messages"][1]["content"][0]["type"], "image_url")

    def test_preferences_do_not_mix_task_versions_or_initial_prompts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native_result(root, "success")
            native_result(root, "failed-v2", version="v2", outcome="failure")
            native_result(root, "failed-prompt", outcome="failure", image=True)
            output = root / "pref.jsonl"
            self.assertEqual(export_dataset([root], output, format="preference")["records"], 0)
            native_result(root, "failed-validation", split="validation", outcome="failure")
            self.assertEqual(export_dataset([root], output, format="preference")["records"], 0)
            native_result(root, "failed-v1", outcome="failure")
            self.assertEqual(export_dataset([root], output, format="preference")["records"], 1)

    def test_scene_variants_share_split_and_conflicting_declarations_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [
                {"id": str(i), "task_id": f"variant-{i}", "split_group": "same-demo", "messages": []} for i in range(20)
            ]
            data = root / "data.jsonl"
            data.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            split = split_dataset(data, root / "splits")
            self.assertEqual(sorted(split["records"].values()), [0, 0, 20])
            records[0]["split"] = "train"
            records[1]["split"] = "test"
            data.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            with self.assertRaisesRegex(ValueError, "conflicting"):
                split_dataset(data, root / "conflict")

    def test_validator_detects_shared_source_with_different_task_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / "train.jsonl", root / "test.jsonl"]
            for index, path in enumerate(paths):
                path.write_text(
                    json.dumps({"id": str(index), "task_id": f"variant-{index}", "split_group": "same-demo"}) + "\n"
                )
            result = validate_datasets(paths)
            self.assertFalse(result["valid"])
            self.assertTrue(result["group_overlaps"])
