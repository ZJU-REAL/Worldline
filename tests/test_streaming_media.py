from __future__ import annotations

import base64
import json
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import patch

from worldline.datasets import split_dataset
from worldline.distill import build_distillation_bundle
from worldline.media import MediaStore, hydrate_media
from worldline.trace import TraceEvent, TraceWriter
from test_datasets import make_result


class StreamingMediaTest(unittest.TestCase):
    def test_image_is_saved_once_and_hydration_checks_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = MediaStore(root)
            payload = {"url": "data:image/png;base64," + base64.b64encode(b"same-image").decode()}
            trace = TraceWriter(root / "trace.jsonl", sync_every=32, media_store=store)
            for _ in range(20):
                trace.append(TraceEvent("test", "observation", payload))
            trace.close()
            self.assertEqual(store.files_written, 1)
            saved = json.loads((root / "trace.jsonl").read_text().splitlines()[0])["payload"]
            self.assertEqual(hydrate_media(saved, root), payload)
            next((root / "media").iterdir()).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                hydrate_media(saved, root)

    def test_distillation_reads_each_trajectory_once_for_all_views(self):
        from worldline import datasets

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_result(root, "good", "task", "success", 1)
            make_result(root, "bad", "task", "failure", 0)
            original = datasets._load_object
            read = []

            def load(path):
                if path.name == "trajectory.json":
                    read.append(path)
                return original(path)

            with patch("worldline.datasets._load_object", side_effect=load):
                bundle = build_distillation_bundle([root / "runs"], root / "bundle")
            self.assertEqual(len(read), 2)
            self.assertEqual(bundle["datasets"]["preference"]["export"]["records"], 1)

    def test_split_memory_does_not_scale_with_conversation_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "source.jsonl"
            with path.open("w") as stream:
                for index in range(400):
                    stream.write(
                        json.dumps(
                            {
                                "id": str(index),
                                "task_id": str(index),
                                "messages": [{"role": "assistant", "content": "x" * 32768}],
                            }
                        )
                        + "\n"
                    )
            tracemalloc.start()
            split_dataset(path, root / "split")
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.assertLess(peak, 2 * 1024 * 1024)
