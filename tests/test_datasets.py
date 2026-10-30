from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldline.datasets import atif_messages, export_dataset, split_dataset, validate_datasets


def make_result(root: Path, name: str, task: str, outcome: str, reward: float) -> None:
    trial = root / "trials" / name
    agent = trial / "agent"
    agent.mkdir(parents=True)
    trajectory = {
        "schema_version": "ATIF-v1.8",
        "session_id": name,
        "agent": {"name": "test", "version": "1"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "move the block"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "I will inspect it.",
                "tool_calls": [{"tool_call_id": "call-1", "function_name": "observe", "arguments": {}}],
                "observation": {"results": [{"source_call_id": "call-1", "content": "image.png"}]},
            },
        ],
    }
    (agent / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    run = root / "runs" / name
    run.mkdir(parents=True)
    result = {
        "schema_version": "worldline.result/0.1",
        "backend": "harbor",
        "trial_id": name,
        "task_id": task,
        "outcome": outcome,
        "reward": reward,
        "metrics": {"progress": reward},
        "usage": {"n_input_tokens": 10},
        "source": str(trial),
        "artifacts": [{"kind": "agent_trace", "path": "agent/trajectory.json", "size_bytes": 1}],
    }
    (run / "result.normalized.json").write_text(json.dumps(result), encoding="utf-8")


class DatasetExportTest(unittest.TestCase):
    def test_preserves_tool_calls_and_observations(self) -> None:
        messages = atif_messages(
            {
                "steps": [
                    {
                        "source": "agent",
                        "message": "look",
                        "tool_calls": [{"tool_call_id": "x", "function_name": "observe", "arguments": {}}],
                        "observation": {"results": [{"source_call_id": "x", "content": "seen"}]},
                    }
                ]
            }
        )
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[1], {"role": "tool", "content": "seen", "tool_call_id": "x"})

    def test_exports_sft_rl_and_preference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_result(root, "success", "task-a", "success", 1.0)
            make_result(root, "failure", "task-a", "failure", 0.0)

            sft = export_dataset([root / "runs"], root / "sft.jsonl", format="sft")
            rl = export_dataset([root / "runs"], root / "rl.jsonl", format="rl")
            pref = export_dataset([root / "runs"], root / "pref.jsonl", format="preference")

            self.assertEqual(sft["records"], 1)
            self.assertEqual(rl["records"], 2)
            self.assertEqual(pref["records"], 1)
            sft_record = json.loads((root / "sft.jsonl").read_text(encoding="utf-8"))
            pref_record = json.loads((root / "pref.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(sft_record["metadata"]["outcome"], "success")
            self.assertEqual(pref_record["chosen_reward"], 1.0)
            self.assertEqual(pref_record["rejected_reward"], 0.0)
            self.assertTrue((root / "sft.jsonl.manifest.json").is_file())

    def test_split_keeps_every_task_in_one_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "data.jsonl"
            with dataset.open("w", encoding="utf-8") as stream:
                for index in range(20):
                    stream.write(json.dumps({"id": str(index), "task_id": f"task-{index // 2}", "messages": []}) + "\n")
            output = root / "split"

            manifest = split_dataset(dataset, output, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2)
            validation = validate_datasets([output / "train.jsonl", output / "validation.jsonl", output / "test.jsonl"])

            self.assertEqual(sum(manifest["records"].values()), 20)
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["overlaps"], [])

    def test_validation_finds_duplicate_ids_and_task_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left = root / "left.jsonl"
            right = root / "right.jsonl"
            left.write_text(json.dumps({"id": "same", "task_id": "task-a", "messages": []}) + "\n", encoding="utf-8")
            right.write_text(json.dumps({"id": "same", "task_id": "task-a", "messages": []}) + "\n", encoding="utf-8")

            validation = validate_datasets([left, right])

            self.assertFalse(validation["valid"])
            self.assertTrue(any("duplicate id" in error for error in validation["errors"]))
            self.assertEqual(validation["overlaps"][0]["tasks"], ["task-a"])


if __name__ == "__main__":
    unittest.main()
