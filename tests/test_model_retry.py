from __future__ import annotations

import json
import threading
import time
import unittest
import io
from types import SimpleNamespace
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from worldline.agents.tool_agent import OpenAICompatibleAgent
from worldline.errors import ModelAPIError


class ModelRetryTest(unittest.TestCase):
    def test_nonfinite_usage_does_not_poison_the_result_or_trace(self):
        events = []
        agent = self.agent(SimpleNamespace(server_port=0), events)
        payload = {
            "usage": {"prompt_tokens": 3, "completion_tokens": float("nan")},
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
        }
        with patch(
            "worldline.agents.tool_agent.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())
        ):
            self.assertEqual(agent._completion()["content"], "done")
        self.assertEqual(agent.usage["prompt_tokens"], 3)
        self.assertNotIn("completion_tokens", agent.usage)
        self.assertEqual(agent.usage["invalid_usage_fields"], 1)
        json.dumps(events, allow_nan=False)

    def test_malformed_provider_message_is_an_unscorable_provider_error(self):
        agent = self.agent(SimpleNamespace(server_port=0), [])
        payload = {"choices": [{"message": {"role": "assistant", "content": float("nan")}}]}
        with patch(
            "worldline.agents.tool_agent.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())
        ):
            with self.assertRaises(ModelAPIError) as caught:
                agent._completion()
        self.assertEqual(caught.exception.code, "model.http.422")
        self.assertFalse(caught.exception.retryable)

    def server(self, status):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if len(requests) == 1:
                    self.send_response(status)
                    self.send_header("Retry-After", "0")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"choices": [{"message": {"role": "assistant", "content": "done"}}]}).encode()
                    )

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return requests, server

    def agent(self, server, events):
        return OpenAICompatibleAgent(
            model="test",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            spec={"action_schema": {}},
            initial_observation={},
            emit=lambda *args: events.append(args),
            messages=[{"role": "user", "content": "same prompt"}],
            usage={},
            max_turns=5,
            deadline=time.monotonic() + 5,
            max_retries=2,
            retry_base_sec=0,
        )

    def test_transient_provider_error_retries_same_context(self):
        requests, server = self.server(503)
        events = []
        agent = self.agent(server, events)
        self.assertEqual(agent._completion()["content"], "done")
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(agent.usage["model_requests"], 2)
        self.assertTrue(any(event[0] == "model.retry" for event in events))

    def test_authentication_error_is_permanent(self):
        requests, server = self.server(401)
        agent = self.agent(server, [])
        with self.assertRaises(ModelAPIError) as caught:
            agent._completion()
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(len(requests), 1)
