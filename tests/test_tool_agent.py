from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from worldline.agents.tool_agent import OpenAICompatibleAgent, observation_content
from worldline.contracts import Task


class ToolAgentTest(unittest.TestCase):
    def test_model_http_tool_loop_keeps_feedback_and_blocks_verifier_tools(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                function = (
                    {"name": "evaluate", "arguments": "{}"}
                    if len(requests) == 1
                    else {"name": "act", "arguments": json.dumps({"action": {"joint_targets": [0, 0]}})}
                )
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": f"call-{len(requests)}", "type": "function", "function": function}],
                }
                if len(requests) == 3:
                    message = {"role": "assistant", "content": "done"}
                response = json.dumps({"choices": [{"message": message}], "usage": {"prompt_tokens": 10}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *_):
                pass

        class View:
            def act(self, action):
                self.action = action
                return {"joint_positions": [0.1, 0.2]}

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            messages, usage = [], {}
            spec = {"action_schema": {"type": "object"}}
            agent = OpenAICompatibleAgent(
                model="test",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                spec=spec,
                initial_observation={"joint_positions": [0, 0]},
                emit=lambda *_: None,
                messages=messages,
                usage=usage,
                max_turns=4,
                deadline=time.monotonic() + 10,
            )
            view = View()
            agent.run(Task("test", "reach", "test", "1"), view)
            self.assertIn("error", requests[1]["messages"][-1]["content"])
            self.assertEqual(json.loads(requests[2]["messages"][-1]["content"])["joint_positions"], [0.1, 0.2])
            self.assertEqual(view.action, {"joint_targets": [0, 0]})
            self.assertEqual(usage["prompt_tokens"], 30)
            self.assertNotIn("evaluate", {tool["function"]["name"] for tool in requests[0]["tools"]})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_observations_preserve_images_as_multimodal_content(self):
        value = observation_content({"state": [1], "images": [{"mime_type": "image/png", "data": "YWJj"}]})
        self.assertEqual(value[1], {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}})
