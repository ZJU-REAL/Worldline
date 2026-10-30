from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worldline.results import ingest_harbor_trial, normalize_harbor_trial


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class HarborResultTest(unittest.TestCase):
    def test_normalizes_success_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trial = Path(temp_dir) / "task__trial"
            write_json(
                trial / "result.json",
                {
                    "id": "trial-1",
                    "task_name": "embodied-first-exam/task",
                    "agent_result": {"n_input_tokens": 12, "n_output_tokens": 3},
                },
            )
            write_json(trial / "verifier/reward.json", {"success": 1, "progress": 0.8, "sim_steps": 42})
            (trial / "verifier/reward.txt").write_text("1\n", encoding="utf-8")
            (trial / "verifier/replay.mp4").write_bytes(b"video")
            (trial / "agent").mkdir()
            (trial / "agent/trajectory.json").write_text("{}", encoding="utf-8")

            result = normalize_harbor_trial(trial)

            self.assertEqual(result.outcome, "success")
            self.assertEqual(result.reward, 1.0)
            self.assertEqual(result.metrics["sim_steps"], 42.0)
            self.assertEqual(result.usage["n_input_tokens"], 12.0)
            self.assertEqual({artifact.kind for artifact in result.artifacts}, {"video", "agent_trace"})

    def test_missing_verdict_is_infrastructure_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trial = Path(temp_dir) / "broken"
            write_json(trial / "result.json", {"exception_info": {"exception_type": "EnvironmentStartError"}})

            result = normalize_harbor_trial(trial)

            self.assertEqual(result.outcome, "infra_error")
            self.assertIsNone(result.reward)
            self.assertIn("reward.json", result.error or "")

    def test_ingest_writes_normalized_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trial = root / "failed"
            write_json(trial / "result.json", {"id": "trial-2", "task_name": "example/task"})
            write_json(trial / "verifier/reward.json", {"success": 0, "progress": 0.25})
            (trial / "verifier/reward.txt").write_text("0\n", encoding="utf-8")

            result, run_dir = ingest_harbor_trial(trial, root / "runs")
            saved = json.loads((run_dir / "result.normalized.json").read_text(encoding="utf-8"))

            self.assertEqual(result.outcome, "failure")
            self.assertEqual(saved["schema_version"], "worldline.result/0.1")
            self.assertEqual(saved["metrics"]["progress"], 0.25)
            self.assertTrue((run_dir / "trace.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
