from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldline.trace import TraceEvent, TraceWriter


class TraceWriterTest(unittest.TestCase):
    def test_writes_jsonl_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            TraceWriter(path).append(TraceEvent(run_id="run-1", kind="action", payload={"name": "move"}))

            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["run_id"], "run-1")
            self.assertEqual(record["kind"], "action")
            self.assertEqual(record["payload"], {"name": "move"})


if __name__ == "__main__":
    unittest.main()
