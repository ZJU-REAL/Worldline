"""Per-trial authenticated MCP/public-tools and Responses bridge for isolated Codex."""

import hmac
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from worldline.adapters.codex_responses import sse_events
from worldline.agents.tool_agent import observation_content, tools_for
from worldline.errors import AgentError, BudgetExceeded, InvalidAction, ModelAPIError
from worldline.runtime.tools import execute_tool


class CodexGateway:
    def __init__(self, adapter, sandbox, *, bind="0.0.0.0"):
        self.adapter, self.sandbox = adapter, sandbox
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        original_cancel = adapter.cancel_check
        adapter.cancel_check = lambda: self.stopping.is_set() or bool(original_cancel and original_cancel())
        self.failure, self.finished = None, False
        self.last_response = None
        self.last_input = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, code, value=None):
                data = json.dumps(value, ensure_ascii=False).encode() if value is not None else b""
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self):
                self.connection.settimeout(max(0.1, min(5, owner.adapter.deadline - time.monotonic())))
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + owner.token):
                    self.send(401, {"error": "unauthorized"})
                    return False
                return True

            def do_GET(self):
                if not self.authorized():
                    return
                if self.path.split("?")[0] == "/v1/models":
                    self.send(200, {"data": [{"id": owner.adapter.model, "object": "model"}]})
                else:
                    self.send(405, {"error": "POST required"})

            def do_POST(self):
                if not self.authorized():
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 16 * 1024 * 1024:
                        self.send(413, {"error": "request byte budget exceeded"})
                        return
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        raise ValueError("incomplete body")
                    body = json.loads(raw)
                    with owner.lock:
                        if owner.failure:
                            self.send(503, {"error": {"message": "trial stopped", "type": "server_error"}})
                        elif self.path == "/mcp":
                            result = owner.mcp(body)
                            self.send(204 if result is None else 200, result)
                        elif self.path == "/v1/responses":
                            if owner.finished:
                                response = owner.final_response()
                            else:
                                if body.get("previous_response_id"):
                                    if body["previous_response_id"] != owner.last_response:
                                        raise InvalidAction("only the latest response can be continued")
                                    body["input"] = owner.last_input + body.get("input", [])
                                owner.adapter.emit(
                                    "adapter.responses_request",
                                    {"input": body.get("input"), "tools": body.get("tools", [])},
                                )
                                response = owner.adapter.decide(body)
                                owner.last_response = response["id"]
                                owner.last_input = body.get("input", []) + response["output"]
                            if body.get("stream", False):
                                self.send_response(200)
                                self.send_header("Content-Type", "text/event-stream")
                                self.send_header("Cache-Control", "no-cache")
                                self.send_header("Connection", "close")
                                self.end_headers()
                                for event in sse_events(response):
                                    self.wfile.write(event)
                                self.wfile.flush()
                                self.close_connection = True
                            else:
                                self.send(200, response)
                        else:
                            self.send(404, {"error": "unknown public endpoint"})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except (BudgetExceeded, ModelAPIError, AgentError) as error:
                    owner.failure = error
                    self.send(503, {"error": {"message": "trial model request stopped", "type": "server_error"}})
                except (ValueError, KeyError, TypeError, InvalidAction):
                    self.send(400, {"error": {"message": "invalid public request", "type": "invalid_request_error"}})
                except Exception as error:
                    owner.failure = error
                    self.send(500, {"error": {"message": "public environment request failed", "type": "server_error"}})

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def get_request(self):
                connection, address = super().get_request()
                connection.settimeout(5)
                return connection, address

        self.server = Server((bind, 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self):
        return self.server.server_port

    def mcp(self, request):
        method, identity = request.get("method"), request.get("id")
        if not isinstance(method, str):
            raise InvalidAction("MCP method must be a string")
        if method.startswith("notifications/"):
            return None
        error = None
        if method == "initialize":
            result = {
                "protocolVersion": request.get("params", {}).get("protocolVersion", "2025-03-26"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "worldline", "version": "0.1"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "inputSchema": t["function"]["parameters"],
                    }
                    for t in tools_for(self.adapter.spec)
                ]
            }
        elif method == "tools/call":
            params = request["params"]
            try:
                if not isinstance(params, dict) or not isinstance(params.get("arguments", {}), dict):
                    raise InvalidAction("tool arguments must be an object")
                if self.finished:
                    raise InvalidAction("attempt already finished")
                self.adapter.emit(
                    "agent.codex_tool_request", {"tool": params["name"], "arguments": params.get("arguments", {})}
                )
                observation, self.finished = execute_tool(params["name"], params.get("arguments", {}), self.sandbox)
                content = observation_content(observation)
                if isinstance(content, str):
                    blocks = [{"type": "text", "text": content}]
                else:
                    blocks = [
                        {"type": "text", "text": p["text"]}
                        if p["type"] == "text"
                        else {
                            "type": "image",
                            "mimeType": p["image_url"]["url"].split(";", 1)[0][5:],
                            "data": p["image_url"]["url"].split(",", 1)[1],
                        }
                        for p in content
                    ]
                result = {"content": blocks, "isError": False}
                self.adapter.emit(
                    "agent.codex_tool_result",
                    {"tool": params["name"], "arguments": params.get("arguments", {}), "content": content},
                )
            except InvalidAction as failure:
                self.adapter.emit(
                    "agent.codex_tool_error",
                    {"tool": params.get("name") if isinstance(params, dict) else None, "error": str(failure)},
                )
                result = {"content": [{"type": "text", "text": json.dumps({"error": str(failure)})}], "isError": True}
            except BudgetExceeded as failure:
                self.adapter.emit(
                    "agent.codex_tool_error",
                    {"tool": params.get("name"), "error": str(failure), "budget_exhausted": True},
                )
                raise
        else:
            error, result = {"code": -32601, "message": "method not found"}, None
        return {"jsonrpc": "2.0", "id": identity, **({"error": error} if error else {"result": result})}

    def final_response(self):
        from worldline.adapters.codex_responses import response_for

        return response_for({"role": "assistant", "content": "Attempt submitted."}, {})

    def close(self):
        self.stopping.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        if not self.lock.acquire(timeout=3):
            raise ModelAPIError("Codex model request did not stop before teardown", status=422)
        self.lock.release()
