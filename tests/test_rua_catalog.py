from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldline.adapters.rua.catalog import catalog_task_paths, write_catalog


class RUACatalogTest(unittest.TestCase):
    def test_old_efe_mode_and_seed_declarations_remain_readable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = root / "tasks/pool/reach-standard-realtime"
            task.mkdir(parents=True)
            (task / "task.toml").write_text('[task]\nname = "efe/reach"\n[metadata]\ninstance_seed = 60417\n')
            (task / "instruction.md").write_text("reach")
            (task / "DIGEST").write_text("abc")
            catalog = write_catalog(root, root / "index.json", mode="standard-realtime")
            self.assertEqual(catalog["count"], 1)
            self.assertEqual(catalog["tasks"][0]["seed"], 60417)

    def test_reads_portable_split_and_detects_changed_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = root / "tasks/pool/reach-privileged"
            task.mkdir(parents=True)
            (task / "task.toml").write_text('[task]\nname = "rua/reach"\n[metadata]\nmode = "privileged"\n')
            (task / "instruction.md").write_text("Reach the target")
            (task / "DIGEST").write_text("abc")
            (root / "portable-catalog.json").write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "id": "pool",
                                "split": "train",
                                "ready_for_collection": True,
                                "carrier_image": "private-company-registry",
                                "request": "private-request",
                                "eligible_tasks": ["reach-privileged"],
                            }
                        ]
                    }
                )
            )
            output = root / "index.json"
            catalog = write_catalog(root, output, split="train")
            self.assertEqual(catalog["count"], 1)
            self.assertEqual(catalog["tasks"][0]["split"], "train")
            self.assertNotIn("private-company", output.read_text())
            self.assertEqual(catalog_task_paths(output), [task])
            (task / "DIGEST").write_text("changed")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                catalog_task_paths(output)

    def test_rejects_paths_outside_catalog_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "index.json"
            output.write_text(
                json.dumps(
                    {
                        "schema_version": "worldline.catalog/0.1",
                        "root": str(root),
                        "tasks": [{"id": "x", "path": "../escape", "version": "x"}],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                catalog_task_paths(output)
