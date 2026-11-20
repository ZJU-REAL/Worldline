"""Model-first evaluation boundaries, service ownership and protocol compatibility."""

import contextlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from worldline.cli import main
from worldline.evaluation import run_evaluation as platform_run_evaluation
from worldline.harnesses import HarnessPlugin, harness_plugins
from worldline.harnesses.json_agent import JSONActionAgent, parse_action
from worldline.models.config import ModelConfig
from worldline.models.serving import ModelService, probe_model
from worldline.errors import BudgetExceeded, InvalidAction
from worldline.contracts import Task


def run_evaluation(*args, **kwargs):
    kwargs.setdefault("harness", "json")
    return platform_run_evaluation(*args, **kwargs)


@contextlib.contextmanager
def model_endpoint(respond):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"fixture-model"}]}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            code, message = respond(body)
            response = {
                "choices": [{"message": {"role": "assistant", "content": message}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            }
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ModelConfig("fixture-model", base_url=f"http://127.0.0.1:{server.server_port}/v1"), calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ModelEntryTest(unittest.TestCase):
    def test_default_cli_plan_needs_no_endpoint_or_model_library(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["run", "--model", "org/open-model", "--runs-dir", root, "--dry-run"]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "planned")
            self.assertEqual(result["harness"], "codex")
            self.assertEqual(result["model"]["backend"], "vllm")
            self.assertEqual(result["tasks"][0]["split"], "test")

    def test_harness_protocol_and_task_mismatch_fail_before_serving(self):
        with patch("worldline.evaluation.ModelService") as service:
            with self.assertRaisesRegex(ValueError, "requires"):
                run_evaluation(ModelConfig("model", base_url="https://service.invalid/v1"), harness="harbor:codex")
            with self.assertRaisesRegex(ValueError, "other task format"):
                run_evaluation(
                    ModelConfig("model", base_url="https://service.invalid/v1", protocol="responses"),
                    harness="harbor:codex",
                )
            with self.assertRaisesRegex(ValueError, "image-capable"):
                run_evaluation(ModelConfig("model", base_url="https://service.invalid/v1"), suite="cua-smoke")
            service.assert_not_called()

    def test_model_config_rejects_secret_urls_and_server_overrides(self):
        for values in (
            {"base_url": "https://user:secret@service/v1"},
            {"server_args": ("--port=123",)},
            {"server_args": ("--api-key=secret",)},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ModelConfig("model", **values)
        with self.assertRaisesRegex(ValueError, "requires --base-url"):
            ModelConfig("model", backend="endpoint")

    def test_local_weights_mutation_changes_identity(self):
        with tempfile.TemporaryDirectory() as root:
            weights = Path(root) / "test.gguf"
            weights.write_bytes(b"weights-a")
            config = ModelConfig(str(weights))
            initial = config.identity()
            weights.write_bytes(b"weights-b")
            self.assertNotEqual(initial["weights_sha256"], config.identity()["weights_sha256"])

    def test_duplicate_harness_cannot_override_builtin(self):
        class Entry:
            name = "json"

            def load(self):
                return lambda: HarnessPlugin("json", "1", "native", ("chat",), lambda **_: None)

        with (
            patch("worldline.harnesses.entry_points", return_value=[Entry()]),
            self.assertRaisesRegex(ValueError, "duplicate"),
        ):
            harness_plugins()

    def test_external_endpoint_is_not_owned_and_probe_cost_is_separate(self):
        with model_endpoint(lambda _: (200, "OK")) as (config, calls), tempfile.TemporaryDirectory() as root:
            with ModelService(config, Path(root)) as service:
                probe = probe_model(service.model)
                self.assertFalse(probe["scored"])
                self.assertEqual(probe["usage"]["total_tokens"], 11)
                self.assertIsNone(service.process)
            self.assertEqual(len(calls), 1)

    def test_bad_endpoint_does_not_start_trial(self):
        with model_endpoint(lambda _: (401, "denied")) as (config, _), tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "probe failed"):
                run_evaluation(config, runs_dir=Path(root), batch_id="auth-failure")
            self.assertFalse((Path(root) / "batches").exists())
            result = json.loads((Path(root) / "evaluations/auth-failure/result.json").read_text())
            self.assertEqual(result["status"], "failed")

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "requires RUA extra")
    def test_real_environment_success_report_resume_and_artifact_integrity(self):
        def respond(body):
            if len(body["messages"]) == 1:
                return 200, "OK"
            obs = json.loads(body["messages"][-1]["content"])
            x, y = obs["target_position"]
            if (
                math.hypot(obs["tip_position"][0] - x, obs["tip_position"][1] - y) < 0.015
                and math.hypot(*obs["joint_velocities"]) < 0.1
            ):
                return 200, '{"tool":"finish","arguments":{}}'
            elbow = math.acos((x * x + y * y - 0.4**2 - 0.3**2) / (2 * 0.4 * 0.3))
            shoulder = math.atan2(y, x) - math.atan2(0.3 * math.sin(elbow), 0.4 + 0.3 * math.cos(elbow))
            return 200, json.dumps({"tool": "act", "arguments": {"action": {"joint_targets": [shoulder, elbow]}}})

        with model_endpoint(respond) as (config, calls), tempfile.TemporaryDirectory() as root:
            result = run_evaluation(config, runs_dir=Path(root), batch_id="e2e", max_workers=2)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["summary"]["outcomes"]["success"], 1)
            scored_requests = result["summary"]["usage_totals"]["model_requests"]
            self.assertEqual(len(calls), scored_requests + 1)  # separate protocol probe
            count = len(calls)
            resumed = run_evaluation(config, runs_dir=Path(root), batch_id="e2e", max_workers=2, resume=True)
            self.assertTrue(resumed["resumed"])
            self.assertEqual(len(calls), count)  # no model loading or inference on completed resume
            manifest = json.loads((Path(root) / "batches/e2e/manifest.json").read_text())
            artifact = Path(manifest["runs"][0]["run_dir"]) / "result.normalized.json"
            artifact.write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
                run_evaluation(config, runs_dir=Path(root), batch_id="e2e", max_workers=2, resume=True)

    def test_guard_reaps_server_descendants_after_controller_pipe_closes(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "pid"
            code = (
                "import subprocess,time,pathlib; p=subprocess.Popen(['sleep','60']); pathlib.Path("
                + repr(str(marker))
                + ").write_text(str(p.pid)); time.sleep(60)"
            )
            guard = subprocess.Popen(
                [sys.executable, "-m", "worldline.runtime.command_guard"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                guard.stdin.write(json.dumps([sys.executable, "-c", code]).encode() + b"\n")
                guard.stdin.flush()
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists())
                child_pid = int(marker.read_text())
                guard.stdin.close()
                guard.wait(timeout=5)
                # A killed orphan may briefly be a zombie under a container init.
                proc = Path(f"/proc/{child_pid}/stat")
                self.assertTrue(not proc.exists() or proc.read_text().split()[2] == "Z")
            finally:
                if not guard.stdin.closed:
                    guard.stdin.close()
                if guard.poll() is None:
                    guard.kill()
                    guard.wait(timeout=5)

    def test_owned_startup_timeout_closes_listener_and_process(self):
        with tempfile.TemporaryDirectory() as root:
            code = "import time; time.sleep(60)"
            config = ModelConfig(
                "model", backend="command", server_command=(sys.executable, "-c", code), startup_timeout_sec=0.15
            )
            service = ModelService(config, Path(root))
            with self.assertRaisesRegex(ValueError, "readiness"):
                service.__enter__()
            self.assertIsNotNone(service.process.poll())
            self.assertTrue(service.log.closed)

    def test_json_harness_reports_invalid_action_in_same_context(self):
        events, messages = [], []

        class View:
            def observe(self):
                return {"actual": 42}

        agent = JSONActionAgent(
            model="model",
            base_url="http://localhost:1/v1",
            spec={"action_schema": {}},
            initial_observation={},
            emit=lambda kind, data: events.append((kind, data)),
            messages=messages,
            usage={},
            max_turns=3,
            deadline=time.monotonic() + 10,
        )
        replies = ["not JSON", '{"tool":"observe","arguments":{}}', '{"tool":"finish","arguments":{}}']
        with patch.object(agent, "_completion", side_effect=[{"role": "assistant", "content": r} for r in replies]):
            agent.run(Task("task", "Do it", "env", "1"), View())
        self.assertIn("error", messages[4]["content"])
        self.assertEqual(json.loads(messages[6]["content"])["actual"], 42)
        self.assertNotIn("tools", agent._tool_request())
        for text in ('{"tool":"act","arguments":{"action":NaN}}', '{"tool":"finish","arguments":{},"extra":1}'):
            with self.assertRaises(InvalidAction):
                parse_action(text)

    def test_invalid_json_consumes_turns_without_infrastructure_retry(self):
        agent = JSONActionAgent(
            model="model",
            base_url="http://localhost:1/v1",
            spec={"action_schema": {}},
            initial_observation={},
            emit=lambda *_: None,
            messages=[],
            usage={},
            max_turns=2,
            deadline=time.monotonic() + 10,
        )
        with patch.object(
            agent, "_completion", return_value={"role": "assistant", "content": "nonsense"}
        ) as completion:
            with self.assertRaises(BudgetExceeded):
                agent.run(Task("task", "Do it", "env", "1"), object())
            self.assertEqual(completion.call_count, 2)

    def test_lazy_core_and_model_registration(self):
        code = "import sys; import worldline,worldline.models,worldline.harnesses; assert not {'torch','vllm','transformers','harbor','mujoco','browsergym'} & set(sys.modules)"
        subprocess.run([sys.executable, "-c", code], check=True, timeout=10)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "requires RUA extra")
    def test_owned_service_shared_by_workers_and_agent_failure_is_not_retried(self):
        from worldline.suites import suite_tasks

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            marker = directory / "starts"
            script = directory / "server.py"
            script.write_text("""import json,sys,pathlib
from http.server import BaseHTTPRequestHandler,HTTPServer
pathlib.Path(sys.argv[3]).open("a").write("started\\n")
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({"data":[{"id":sys.argv[2]}]}).encode())
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({"choices":[{"message":{"role":"assistant","content":'{"tool":"finish","arguments":{}}'}}],"usage":{"total_tokens":5}}).encode())
    def log_message(self,*args): pass
HTTPServer(("127.0.0.1",int(sys.argv[1])),Handler).serve_forever()
""")
            config = ModelConfig(
                "owned-fixture",
                backend="command",
                server_command=(sys.executable, str(script), "{port}", "{name}", str(marker)),
                startup_timeout_sec=5,
            )
            task = suite_tasks("rua-smoke")[0][0]
            result = run_evaluation(
                config, tasks=[task, task], runs_dir=directory / "runs", batch_id="shared", max_workers=2, cpu_budget=2
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["summary"]["outcomes"], {"success": 0, "failure": 2, "infra_error": 0})
            self.assertEqual(marker.read_text().splitlines(), ["started"])
            manifest = json.loads((directory / "runs/batches/shared/manifest.json").read_text())
            self.assertTrue(all(len(row["attempts"]) == 1 for row in manifest["runs"]))
            run = json.loads((Path(manifest["runs"][0]["run_dir"]) / "run.json").read_text())
            self.assertEqual(run["metadata"]["agent_runtime_config"]["settings"]["base_url"], "owned://command")
            run_evaluation(
                config,
                tasks=[task, task],
                runs_dir=directory / "runs",
                batch_id="shared",
                max_workers=2,
                cpu_budget=2,
                resume=True,
            )
            self.assertEqual(marker.read_text().splitlines(), ["started"])

    def test_codex_dispatch_preserves_native_config_and_keeps_key_out_of_artifacts(self):
        from worldline.models.config import ResolvedModel
        import os

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            task = directory / "task"
            task.mkdir()
            (task / "task.toml").write_text('[task]\nname="test"\n')
            (task / "instruction.md").write_text("Do it")
            config = ModelConfig("served-alias", base_url="https://model.example/v1", protocol="responses")
            model = ResolvedModel(
                "served-alias", config.base_url, config.api_key_env, None, "responses", False, config.identity()
            )
            from types import SimpleNamespace

            service = SimpleNamespace(model=model, readiness={})
            with (
                patch("worldline.evaluation.ModelService") as cls,
                patch("worldline.evaluation.probe_model", return_value={}),
                patch("worldline.evaluation.shutil.which", return_value="harbor"),
                patch("worldline.batch.evaluate_many_harbor", return_value={"runs": [], "statuses": {}}) as run,
                patch.dict(os.environ, {"WORLDLINE_API_KEY": "private-test-secret"}),
            ):
                cls.return_value.__enter__.return_value = service
                run_evaluation(config, harness="harbor:codex", tasks=[task], runs_dir=directory / "runs")
            args = run.call_args.kwargs
            self.assertEqual(args["model"], "openai/served-alias")
            self.assertEqual(args["env"]["OPENAI_API_KEY"], "private-test-secret")
            self.assertIn('config_source={"openai_base_url": "https://model.example/v1"}', args["extra_args"])
            for artifact in (directory / "runs").rglob("*.json"):
                self.assertNotIn("private-test-secret", artifact.read_text())
