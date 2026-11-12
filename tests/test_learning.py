from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.learning.launch import train
from worldline.learning.transformers_sft import encode_record


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [ord(character) for message in messages for character in f"{message['role']}:{message['content']};"]


class LearningTest(unittest.TestCase):
    def test_sft_loss_masks_observation_and_keeps_assistant(self):
        row = {
            "messages": [
                {"role": "user", "content": "move"},
                {"role": "assistant", "content": "act"},
                {"role": "tool", "content": "observed"},
            ]
        }
        result = encode_record(Tokenizer(), row, 1000)
        user_length = len("user:move;")
        assistant_length = len("assistant:act;")
        self.assertTrue(all(label == -100 for label in result["labels"][:user_length]))
        self.assertTrue(all(label != -100 for label in result["labels"][user_length : user_length + assistant_length]))
        self.assertTrue(all(label == -100 for label in result["labels"][user_length + assistant_length :]))
        with self.assertRaisesRegex(ValueError, "no assistant supervision"):
            encode_record(Tokenizer(), row, 2)

    def test_multimodal_sft_does_not_silently_drop_images(self):
        row = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]}]}
        with self.assertRaisesRegex(ValueError, "processor-aware"):
            encode_record(Tokenizer(), row, 1000)

    def make_config(self, root):
        data = root / "train.jsonl"
        data.write_text(
            json.dumps({"id": "1", "task_id": "a", "messages": [{"role": "assistant", "content": "act"}]}) + "\n"
        )
        config = root / "train.toml"
        config.write_text(
            'schema_version = "worldline.training/0.1"\n[trainer]\nbackend = "command"\n'
            'train_file = "train.jsonl"\noutput_dir = "model"\ncommand = ["trainer", "{train_file}"]\n'
        )
        return config, data

    def test_training_plan_hashes_inputs_and_real_launch_records_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, data = self.make_config(root)
            with patch("worldline.learning.launch.subprocess.run") as run:
                manifest, path = train(config, dry_run=True)
                self.assertEqual(manifest["status"], "planned")
                self.assertEqual(manifest["command"], ["trainer", str(data)])
                self.assertEqual(len(manifest["inputs"][0]["sha256"]), 64)
                self.assertTrue(path.is_file())
                run.assert_not_called()
            with patch("worldline.learning.launch.subprocess.run", return_value=subprocess.CompletedProcess([], 7)):
                manifest, path = train(config)
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(json.loads(path.read_text())["exit_code"], 7)

    def test_rejects_held_out_and_empty_training_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, data = self.make_config(root)
            row = json.loads(data.read_text())
            row["metadata"] = {"split": "test"}
            data.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "held-out"):
                train(config, dry_run=True)
            data.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                train(config, dry_run=True)
