from __future__ import annotations

import base64
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from worldline.adapters.atif import atif_messages
from worldline.datasets import export_dataset, split_dataset
from worldline.distill import build_distillation_bundle
from worldline.results import normalize_harbor_trial
from worldline.adapters.rua.catalog import write_catalog, catalog_task_metadata
from worldline.batch import evaluate_many_harbor
from worldline.eval import _ingest_completed_trials, evaluate_harbor
from worldline.learning.handoff import export_handoff
from worldline.media import MediaStore, media_bytes
from worldline.storage import publish_bytes
from worldline.trace import TraceWriter
from test_datasets import make_result
from test_eval import make_task
from test_training_handoff import sample


class DataIntegrityTest(unittest.TestCase):
    def test_portable_catalog_split_is_carried_into_the_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = make_task(root)
            (task / "task.toml").write_text('[task]\nname = "example/task"\n')
            (root / "portable-catalog.json").write_text(json.dumps({"datasets": [{"id": root.name, "split": "test"}]}))
            index = root / "index.json"
            write_catalog(root, index)
            batch = evaluate_many_harbor(
                [task], agent="oracle", runs_dir=root / "runs", dry_run=True, task_metadata=catalog_task_metadata(index)
            )
            self.assertEqual(batch["runs"][0]["run"]["metadata"]["split"], "test")
            saved = json.loads((Path(batch["runs"][0]["run_dir"]) / "task.json").read_text())
            self.assertEqual(saved["split"], "test")
            (task / "task.toml").write_text('[task]\nname = "example/task"\n[metadata]\nsplit = "test"\n')
            data = json.loads(index.read_text())
            data["tasks"][0]["split"] = "train"
            index.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "cannot override"):
                catalog_task_metadata(index)

    def test_harbor_basename_alias_is_accepted_but_version_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "verifier").mkdir()
            (root / "verifier/reward.json").write_text('{"success": 1}')
            (root / "result.json").write_text('{"task_name": "task", "task_checksum": "abc"}')
            snapshot = {"id": "efe/task", "version": "sha256:abc", "seed": 7, "split": "train", "metadata": {}}
            result = normalize_harbor_trial(root, task_snapshot=snapshot)
            self.assertEqual(result.task_id, "efe/task")
            snapshot["version"] = "sha256:different"
            with self.assertRaisesRegex(ValueError, "version differs"):
                normalize_harbor_trial(root, task_snapshot=snapshot)

    def test_failed_distillation_is_separate_and_success_paths_are_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_result(root, "good", "task", "success", 1)
            with patch(
                "worldline.distill.validate_datasets", return_value={"valid": False, "errors": ["injected failure"]}
            ):
                with self.assertRaises(ValueError) as error:
                    build_distillation_bundle([root / "runs"], root / "bundle")
            self.assertFalse((root / "bundle").exists())
            failed = next(root.glob(".bundle.failed-*"))
            self.assertFalse((failed / "manifest.json").exists())
            self.assertIn(str(failed), error.exception.__notes__[0])
            manifest = build_distillation_bundle([root / "runs"], root / "bundle")
            self.assertTrue(Path(manifest["datasets"]["sft"]["raw"]).is_file())
            self.assertEqual(manifest["output_dir"], str(root / "bundle"))

    def test_multitrial_test_split_is_preserved_in_each_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("a", "b"):
                trial = root / "harbor" / name
                (trial / "verifier").mkdir(parents=True)
                (trial / "agent").mkdir()
                (trial / "result.json").write_text(json.dumps({"id": name, "task_name": "test-task"}))
                (trial / "verifier/reward.json").write_text('{"success": 1}')
                (trial / "agent/trajectory.json").write_text(
                    json.dumps({"steps": [{"source": "agent", "message": "done"}]})
                )
            snapshot = {
                "id": "test-task",
                "version": "v3",
                "split": "test",
                "seed": 19,
                "metadata": {"split_group": "family"},
            }
            run = root / "run"
            trace = TraceWriter(run / "trace.jsonl")
            self.assertEqual(_ingest_completed_trials(root / "harbor", run, trace, "run", task_snapshot=snapshot), 2)
            trace.close()
            results = list(run.rglob("result.normalized.json"))
            self.assertTrue(all(json.loads(p.read_text())["task_metadata"]["seed"] == 19 for p in results))
            self.assertTrue(all(json.loads(p.read_text())["task_version"] == "v3" for p in results))
            exported = export_dataset([run], root / "sft.jsonl", format="sft")
            self.assertEqual(exported["records"], 0)
            self.assertEqual(exported["load_stats"]["held_out"], 2)

    def test_atif_image_is_restored_as_actual_training_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_result(root, "good", "task", "success", 1)
            trajectory = root / "trials/good/agent/trajectory.json"
            image = sample()["messages"][1]["content"][0]["image_url"]["url"]
            (trajectory.parent / "screen.png").write_bytes(base64.b64decode(image.split(",", 1)[1]))
            content = {"type": "image", "source": {"media_type": "image/png", "path": "screen.png"}}
            data = json.loads(trajectory.read_text())
            data["steps"][0]["message"] = [{"type": "text", "text": "inspect"}, content]
            trajectory.write_text(json.dumps(data))
            output = root / "sft.jsonl"
            self.assertEqual(export_dataset([root / "runs"], output, format="sft")["records"], 1)
            self.assertEqual(json.loads(output.read_text())["messages"][0]["content"][1]["image_url"]["url"], image)
            export_handoff(output, root / "package", framework="llama-factory", tool_prose_policy="omit")
            row = json.loads((root / "package/data/train.jsonl").read_text())
            self.assertEqual(
                (root / "package" / row["images"][0]).read_bytes(), (trajectory.parent / "screen.png").read_bytes()
            )

    def test_external_or_missing_atif_media_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for location in ("https://example.org/image.png", "../../../private.png", "missing.png"):
                with self.subTest(location=location), self.assertRaises(ValueError):
                    atif_messages(
                        {
                            "steps": [
                                {
                                    "source": "user",
                                    "message": [
                                        {"type": "image", "source": {"media_type": "image/png", "path": location}}
                                    ],
                                }
                            ]
                        },
                        artifact_root=root,
                        trajectory_path=root / "agent/trajectory.json",
                    )

    def test_hash_split_labels_are_enforced_by_training_handoff(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = sample()
            row["metadata"]["split"] = "unspecified"
            source = root / "data.jsonl"
            source.write_text(json.dumps(row) + "\n")
            split_dataset(source, root / "split", train_ratio=0, validation_ratio=0, test_ratio=1)
            test = root / "split/test.jsonl"
            self.assertEqual(json.loads(test.read_text())["split"], "test")
            with self.assertRaisesRegex(ValueError, "held-out"):
                export_handoff(test, root / "bad", framework="llama-factory")
            self.assertFalse((root / "bad").exists())

    def test_failed_publication_never_leaves_partial_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "record.json"
            path.write_bytes(b"original")
            with patch("worldline.storage.os.fsync", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    publish_bytes(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_concurrent_media_publication_deduplicates_complete_blobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stores = [MediaStore(root) for _ in range(8)]
            encoded = base64.b64encode(b"payload" * 10000).decode()
            with ThreadPoolExecutor(max_workers=8) as pool:
                refs = list(pool.map(lambda store: store.put(encoded, "image/png", "base64"), stores))
            self.assertEqual(sum(store.files_written for store in stores), 1)
            self.assertTrue(all(media_bytes(ref, root) == b"payload" * 10000 for ref in refs))

    def test_harbor_launch_error_records_terminal_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("worldline.eval.shutil.which", return_value="exists"):
                with patch("worldline.eval.subprocess.run", side_effect=OSError("exec failed")):
                    with self.assertRaises(OSError):
                        evaluate_harbor(make_task(root), agent="oracle", runs_dir=root / "runs")
            record = json.loads(next((root / "runs").glob("*/run.json")).read_text())
            self.assertEqual(record["status"], "infra_error")
            self.assertIsNotNone(record["finished_at_ns"])
