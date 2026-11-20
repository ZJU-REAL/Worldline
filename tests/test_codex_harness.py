"""Codex protocol boundaries and optional real-CLI/container integration."""

import importlib.util
import json
import math
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from worldline.adapters.codex_responses import (
    ChatResponsesAdapter,
    ResponsesModelAdapter,
    messages_for_chat,
    public_tools,
    response_for,
    sse_events,
)
from worldline.errors import InvalidAction, BudgetExceeded, ModelAPIError
from worldline.harnesses.codex_gateway import CodexGateway
from worldline.models.config import ModelConfig


def adapter(**options):
    return ChatResponsesAdapter(
        model="fixture",
        base_url="http://localhost:1/v1",
        spec={"action_schema": {"type": "object"}},
        initial_observation={},
        emit=lambda *_: None,
        messages=[],
        usage={},
        max_turns=5,
        deadline=time.monotonic() + 10,
        **options,
    )


class CodexProtocolTest(unittest.TestCase):
    def test_owner_cancellation_interrupts_model_retry_backoff(self):
        stopped = threading.Event()
        agent = adapter()
        agent.cancel_check = stopped.is_set
        timer = threading.Timer(0.1, stopped.set)
        timer.start()
        started = time.monotonic()
        try:
            with patch(
                "worldline.agents.tool_agent.http_completion",
                side_effect=ModelAPIError("temporary", status=503, retry_after=2),
            ):
                with self.assertRaises(BudgetExceeded):
                    agent._completion()
            self.assertLess(time.monotonic() - started, 0.7)
        finally:
            timer.cancel()

    def test_namespace_tools_and_multimodal_history_translate_without_private_tools(self):
        body = {
            "instructions": "rules",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "state"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "namespace": "mcp__worldline",
                    "name": "act",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "actual feedback"},
            ],
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__worldline",
                    "tools": [{"type": "function", "name": "act", "parameters": {}}],
                },
                {"type": "function", "name": "shell", "parameters": {}},
            ],
        }
        messages = messages_for_chat(body)
        self.assertEqual(messages[-1], {"role": "tool", "tool_call_id": "c1", "content": "actual feedback"})
        self.assertEqual(messages[1]["content"][1]["type"], "image_url")
        self.assertEqual([t["function"]["name"] for t in public_tools(body)], ["mcp__worldline__act"])
        body["input"][1]["name"] = "verify"
        with self.assertRaises(InvalidAction):
            messages_for_chat(body)

    def test_function_sse_has_matching_ids_and_complete_usage(self):
        response = response_for(
            {"tool_calls": [{"id": "c1", "function": {"name": "mcp__worldline__observe", "arguments": "{}"}}]},
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )
        events = [json.loads(chunk.decode().split("data: ")[1].strip()) for chunk in sse_events(response)]
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(events[-1]["response"]["usage"]["total_tokens"], 12)
        self.assertEqual(events[2]["item"]["call_id"], "c1")
        self.assertEqual([event["sequence_number"] for event in events], list(range(len(events))))

    def test_json_adapter_uses_content_then_returns_namespace_call(self):
        agent = adapter()
        body = {
            "input": [{"role": "user", "content": "Do it"}],
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__worldline",
                    "tools": [{"type": "function", "name": "finish", "parameters": {}}],
                }
            ],
        }
        with patch.object(
            agent,
            "_completion",
            return_value={"role": "assistant", "content": '{"tool":"mcp__worldline__finish","arguments":{}}'},
        ):
            response = agent.decide(body)
        self.assertEqual(response["output"][0]["name"], "finish")
        self.assertEqual(response["output"][0]["namespace"], "mcp__worldline")
        self.assertEqual(agent._tool_request(), {})
        self.assertIn("Available tools", agent.messages[0]["content"])

    def test_invalid_model_decision_remains_agent_output(self):
        agent = adapter()
        with patch.object(agent, "_completion", return_value={"role": "assistant", "content": "not JSON"}):
            response = agent.decide({"input": "Do it", "tools": []})
        self.assertEqual(response["output"][0]["type"], "message")
        self.assertEqual(response["output"][0]["content"][0]["text"], "not JSON")

    def test_public_mcp_requires_authentication_and_never_exposes_verifier(self):
        gateway = CodexGateway(adapter(), object(), bind="127.0.0.1")
        try:
            url = f"http://127.0.0.1:{gateway.port}/mcp"

            def call(body, token):
                request = urllib.request.Request(
                    url, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + token}
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    return json.load(response)

            with self.assertRaises(urllib.error.HTTPError) as error:
                call({"method": "tools/list", "id": 1}, "wrong")
            self.assertEqual(error.exception.code, 401)
            listed = call({"method": "tools/list", "id": 2}, gateway.token)
            self.assertEqual({t["name"] for t in listed["result"]["tools"]}, {"observe", "act", "reset", "finish"})
            result = call(
                {"method": "tools/call", "id": 3, "params": {"name": "verify", "arguments": {}}}, gateway.token
            )
            self.assertTrue(result["result"]["isError"])
            self.assertIsNone(gateway.failure)
        finally:
            gateway.close()

    def test_responses_provider_preserves_calls_and_images(self):
        agent = ResponsesModelAdapter(
            model="model",
            base_url="http://localhost:1/v1",
            spec={},
            initial_observation={},
            emit=lambda *_: None,
            messages=[],
            usage={},
            max_turns=2,
            deadline=time.monotonic() + 10,
        )
        agent.current_tools = [{"type": "function", "function": {"name": "mcp__worldline__observe", "parameters": {}}}]
        body = agent._request_body(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}],
            {},
            0,
        )
        self.assertEqual(body["input"][0]["content"][0]["type"], "input_image")
        message = agent._response_message(
            {
                "output": [
                    {"type": "function_call", "call_id": "c1", "name": "mcp__worldline__observe", "arguments": "{}"}
                ]
            }
        )
        self.assertEqual(message["tool_calls"][0]["id"], "c1")


@unittest.skipUnless(
    os.environ.get("WORLDLINE_TEST_CODEX") == "1" and importlib.util.find_spec("mujoco"),
    "set WORLDLINE_TEST_CODEX=1 with Docker/Codex/RUA dependencies",
)
class RealCodexIntegrationTest(unittest.TestCase):
    def test_real_codex_operates_mujoco_for_two_model_aliases(self):
        from worldline.evaluation import run_evaluation

        names = ("fixture-qwen-compatible", "fixture-other-compatible")
        calls = {name: [] for name in names}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": name} for name in names]}).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model = body["model"]
                if len(body["messages"]) == 1 and body["messages"][0]["content"] == "Reply OK.":
                    content = "OK"
                else:
                    calls[model].append(body)
                    tools = (
                        [t["function"] for t in body["tools"]]
                        if body.get("tools")
                        else json.loads(body["messages"][0]["content"].split("Available tools: ", 1)[1])
                    )
                    count = len(calls[model])
                    name = "observe" if count == 1 else "act" if count <= 11 else "finish"
                    tool = next(t["name"] for t in tools if t["name"].endswith("__" + name) or t["name"] == name)
                    x, y = 0.35, -0.3
                    elbow = math.acos((x * x + y * y - 0.4**2 - 0.3**2) / (2 * 0.4 * 0.3))
                    shoulder = math.atan2(y, x) - math.atan2(0.3 * math.sin(elbow), 0.4 + 0.3 * math.cos(elbow))
                    content = json.dumps(
                        {
                            "tool": tool,
                            "arguments": {"action": {"joint_targets": [shoulder, elbow]}} if name == "act" else {},
                        }
                    )
                self.send_response(200)
                self.end_headers()
                message = {"role": "assistant", "content": content}
                if body.get("tools"):
                    decision = json.loads(content)
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"call-{len(calls[model])}",
                                "function": {"name": decision["tool"], "arguments": json.dumps(decision["arguments"])},
                            }
                        ],
                    }
                self.wfile.write(
                    json.dumps(
                        {
                            "choices": [{"message": message}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        }
                    ).encode()
                )

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                for name in names:
                    mode = "json" if name == names[0] else "tools"
                    config = ModelConfig(name, base_url=f"http://127.0.0.1:{server.server_port}/v1")
                    result = run_evaluation(
                        config,
                        harness="codex",
                        runs_dir=Path(root),
                        batch_id=name,
                        cpu_budget=2,
                        agent_options={"codex": {"decision_mode": mode}},
                    )
                    if result["summary"]["outcomes"]["success"] != 1:
                        manifest = json.loads((Path(root) / "batches" / name / "manifest.json").read_text())
                        trace = Path(manifest["runs"][0]["run_dir"]) / "trace.jsonl"
                        self.fail(trace.read_text()[-18000:])
                    self.assertEqual(result["status"], "completed")
                    self.assertGreaterEqual(len(calls[name]), 12)
                    before = len(calls[name])
                    with patch(
                        "worldline.harnesses.codex_image.prepare_image",
                        side_effect=AssertionError("completed resume must not start a container"),
                    ):
                        resumed = run_evaluation(
                            config,
                            harness="codex",
                            runs_dir=Path(root),
                            batch_id=name,
                            cpu_budget=2,
                            agent_options={"codex": {"decision_mode": mode}},
                            resume=True,
                        )
                    self.assertTrue(resumed["resumed"])
                    self.assertEqual(len(calls[name]), before)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
