from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import tempfile
import tracemalloc
import unittest
from pathlib import Path

from worldline.datasets import export_dataset, split_dataset, validate_datasets
from worldline.distill import build_distillation_bundle
from worldline.learning.handoff import export_handoff
from worldline.learning.formats.verl import ParquetStream
from worldline.media import MediaStore, MARKER, MessageHistory
from test_datasets import make_result
from test_training_handoff import sample


class VisualStreamingTest(unittest.TestCase):
    def test_jsonl_record_budget_is_enforced_before_parsing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "large.jsonl"
            path.write_text(
                json.dumps({"id": "row", "task_id": "task", "messages": [{"role": "assistant", "content": "x" * 2048}]})
                + "\n"
            )
            validation = validate_datasets([path], max_record_bytes=1024)
            self.assertFalse(validation["valid"])
            self.assertTrue(any("byte budget" in error for error in validation["errors"]))

    def test_undeclared_splits_require_explicit_review_and_conflicting_splits_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_result(root, "teacher", "task", "success", 1)
            result_path = root / "runs/teacher/result.normalized.json"
            result = json.loads(result_path.read_text())
            result.pop("task_metadata")
            result_path.write_text(json.dumps(result))
            manifest = export_dataset([root / "runs"], root / "strict.jsonl", format="sft")
            self.assertEqual(manifest["records"], 0)
            self.assertEqual(manifest["load_stats"]["undeclared_split"], 1)
            self.assertEqual(
                export_dataset([root / "runs"], root / "reviewed.jsonl", format="sft", allow_unspecified_split=True)[
                    "records"
                ],
                1,
            )
            result["task_metadata"] = {"split": "train"}
            result_path.write_text(json.dumps(result))
            (result_path.parent / "task.json").write_text(json.dumps({"split": "test"}))
            manifest = export_dataset([root / "runs"], root / "conflict.jsonl", format="sft")
            self.assertEqual(manifest["records"], 0)
            self.assertEqual(manifest["load_stats"]["conflicting_split"], 1)

    def test_development_data_stays_in_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "source.jsonl"
            path.write_text(json.dumps({"id": "dev", "task_id": "task", "split": "development", "messages": []}) + "\n")
            manifest = split_dataset(path, root / "splits", train_ratio=1, validation_ratio=0, test_ratio=0)
            self.assertEqual(manifest["records"]["validation"], 1)
            self.assertEqual(manifest["records"]["train"], 0)

    def test_context_budget_hydrates_only_latest_images_without_losing_trace_history(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MediaStore(Path(temp))
            history = MessageHistory(store)
            image = sample()["messages"][1]["content"][0]
            history.append({"role": "system", "content": "fixed task instructions"})
            for index in range(10):
                history.append({"role": "user", "content": [{"type": "text", "text": str(index)}, image]})
                history.append({"role": "assistant", "content": "feedback"})
            context = history.model_context(max_images=2)
            frames = [
                part
                for msg in context
                if isinstance(msg.get("content"), list)
                for part in msg["content"]
                if part["type"] == "image_url"
            ]
            self.assertEqual(len(frames), 2)
            self.assertTrue(all(isinstance(part["image_url"]["url"], str) for part in frames))
            self.assertEqual(len(context), len(history))
            self.assertEqual(context[0]["content"], "fixed task instructions")
            self.assertEqual(
                sum(MARKER in msg["content"][-1]["image_url"]["url"] for msg in history if msg["role"] == "user"), 10
            )

    def test_native_export_memory_does_not_expand_episode_image_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_result(root, "teacher", "task", "success", 1)
            origin = root / "trials/teacher"
            store = MediaStore(origin)
            frames = [
                store.put(base64.b64encode(bytes([index]) * (1024 * 1024)).decode(), "image/png", "data_url")
                for index in range(10)
            ]
            messages = [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": frames[index % 10]}}]}
                for index in range(100)
            ]
            messages.append({"role": "assistant", "content": "done"})
            (origin / "agent/trajectory.json").write_text(
                json.dumps({"schema_version": "worldline.messages/0.2", "messages": messages})
            )
            tracemalloc.start()
            manifest = export_dataset([root / "runs"], root / "dataset/sft.jsonl", format="sft", media_mode="reference")
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.assertEqual(manifest["records"], 1)
            self.assertEqual(manifest["media"]["files"], 10)
            self.assertLess(peak, 4 * 1024 * 1024)
            self.assertLess((root / "dataset/sft.jsonl").stat().st_size, 100 * 1024)
            row = json.loads((root / "dataset/sft.jsonl").read_text())
            self.assertIn(MARKER, row["messages"][0]["content"][0]["image_url"]["url"])

    def test_reference_bundle_can_move_and_handoff_actual_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_result(root, "teacher", "task", "success", 1)
            origin = root / "trials/teacher"
            store = MediaStore(origin)
            row = sample()
            trajectory = {"schema_version": "worldline.messages/0.2", "messages": store.externalize(row["messages"])}
            (origin / "agent/trajectory.json").write_text(json.dumps(trajectory))
            result_path = root / "runs/teacher/result.normalized.json"
            result = json.loads(result_path.read_text())
            result["task_metadata"] = {"split": "train"}
            result_path.write_text(json.dumps(result))
            bundle = build_distillation_bundle([root / "runs"], root / "bundle")
            self.assertEqual(bundle["datasets"]["sft"]["export"]["media"]["files"], 1)
            shutil.move(root / "bundle", root / "moved")
            data = root / "moved/splits/sft/train.jsonl"
            manifest = export_handoff(data, root / "lf", framework="llama-factory")
            self.assertEqual(manifest["records"]["train"], 1)
            converted = json.loads((root / "lf/data/train.jsonl").read_text())
            original = base64.b64decode(row["messages"][1]["content"][0]["image_url"]["url"].split(",", 1)[1])
            self.assertEqual((root / "lf" / converted["images"][0]).read_bytes(), original)
            if importlib.util.find_spec("pyarrow"):
                import pyarrow.parquet as pq

                export_handoff(data, root / "verl", framework="verl")
                self.assertEqual(
                    pq.read_table(root / "verl/data/train.parquet").to_pylist()[0]["images"][0]["bytes"], original
                )

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "install handoff extra")
    def test_parquet_buffer_is_limited_by_bytes_and_rejects_oversized_row(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.parquet"
            writer = ParquetStream(
                path,
                pa.schema([("image", pa.binary())]),
                max_batch_bytes=3 * 1024 * 1024,
                max_record_bytes=4 * 1024 * 1024,
            )
            for _ in range(10):
                writer.append({"image": b"x" * (1024 * 1024)})
            with self.assertRaisesRegex(ValueError, "byte budget"):
                writer.append({"image": b"x" * (5 * 1024 * 1024)})
            writer.close()
            parquet = pq.ParquetFile(path)
            self.assertEqual(parquet.metadata.num_rows, 10)
            self.assertGreater(parquet.metadata.num_row_groups, 1)
            self.assertLessEqual(writer.peak_buffered_bytes, 3 * 1024 * 1024)

    def test_episode_media_budget_rejects_without_publishing_training_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "sample.jsonl"
            path.write_text(json.dumps(sample()) + "\n")
            with self.assertRaisesRegex(ValueError, "no valid training"):
                export_handoff(path, root / "lf", framework="llama-factory", max_media_bytes=1)
            self.assertFalse((root / "lf").exists())
