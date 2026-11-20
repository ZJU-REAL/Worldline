"""Real Codex CLI in an independent container, operating public MCP tools."""

import json
import os
import select
import socket
import subprocess
import sys
import time
import uuid

from worldline.adapters.codex_responses import ChatResponsesAdapter, ResponsesModelAdapter
from worldline.agents.tool_agent import observation_content
from worldline.errors import AgentError, BudgetExceeded, ModelAPIError
from worldline.harnesses.codex_gateway import CodexGateway
from worldline.runtime.containment import remove_container


def advertised_host():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("1.1.1.1", 80))
        return probe.getsockname()[0]


class CodexAgent:
    def __init__(self, *, model, options, **runtime):
        self.model, self.runtime = model, runtime
        options = dict(options)
        self.container_config = options.pop("codex")
        self.cancel_check = None
        mode = self.container_config.get("decision_mode", "json")
        self.adapter = (ResponsesModelAdapter if model.protocol == "responses" else ChatResponsesAdapter)(
            model=model.name,
            base_url=model.base_url,
            api_key_env=model.api_key_env,
            model_revision=model.revision,
            decision_mode=mode,
            structured=self.container_config.get("structured", model.settings["backend"] == "llama-cpp"),
            **options,
            **runtime,
        )

    @property
    def worldline_config(self):
        return {
            **self.adapter.worldline_config,
            "model_source": self.model.settings,
            "base_url": self.model.identity_url,
            "codex": self.container_config,
            "model_protocol": self.model.protocol,
            "decision_mode": self.adapter.decision_mode,
            "structured_decoding": self.adapter.structured,
            "trace_view": "actual adapted model messages and public MCP results",
        }

    def run(self, task, sandbox):
        initial = self.runtime["initial_observation"]
        if isinstance(initial, dict) and initial.get("images") and not self.model.vision:
            raise ValueError("task exposes images; select a vision model and pass --vision")
        self.adapter.cancel_check = self.cancel_check
        gateway = CodexGateway(self.adapter, sandbox)
        container = "worldline-codex-" + uuid.uuid4().hex
        process = guard = None
        try:
            host = self.container_config.get("bridge_host") or advertised_host()
            url = f"http://{host}:{gateway.port}"
            values = {
                "model_provider": "worldline",
                "approval_policy": "never",
                "web_search": "disabled",
                "model_context_window": self.container_config.get("context_window", 32768),
                "model_auto_compact_token_limit": 1000000,
                "model_providers.worldline": '{name="Worldline",base_url='
                + json.dumps(url + "/v1")
                + ',env_key="WORLDLINE_BRIDGE_KEY",wire_api="responses",requires_openai_auth=false}',
                "mcp_servers.worldline": "{url="
                + json.dumps(url + "/mcp")
                + ',bearer_token_env_var="WORLDLINE_BRIDGE_KEY",required=true,default_tools_approval_mode="approve",enabled_tools=["observe","act","reset","finish"],startup_timeout_sec=15,tool_timeout_sec=120}',
            }
            command = [
                "docker",
                "run",
                "--rm",
                "--interactive",
                "--name",
                container,
                "--network",
                "bridge",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "128",
                "--memory",
                str(self.container_config.get("memory_mb", 1024)) + "m",
                "--cpus",
                "1",
                "--user",
                "65532:65532",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
                "--tmpfs",
                "/work:rw,nosuid,nodev,size=64m,mode=1777",
                "--workdir",
                "/work",
                "--env",
                "CODEX_HOME=/tmp",
                "--env",
                "HOME=/tmp",
                "--env",
                "WORLDLINE_BRIDGE_KEY",
                self.container_config["image"],
                "/codex",
                "exec",
                "--json",
                "--ephemeral",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--sandbox",
                "read-only",
                "-m",
                self.model.name,
            ]
            for key, value in values.items():
                encoded = value if key.startswith(("model_providers.", "mcp_servers.")) else json.dumps(value)
                command += ["-c", key + "=" + encoded]
            for feature in (
                "shell_tool",
                "unified_exec",
                "shell_snapshot",
                "apps",
                "plugins",
                "hooks",
                "multi_agent",
                "memories",
                "browser_use",
                "computer_use",
                "image_generation",
                "goals",
                "sleep_tool",
            ):
                command += ["--disable", feature]
            command.append("-")
            prompt = (
                task.instruction
                + "\nOperate only the worldline MCP tools. Finish submits the attempt; an independent verifier scores it. Environment specification:\n"
                + json.dumps(self.runtime["spec"], ensure_ascii=False)
            )
            observation = observation_content(initial)
            if isinstance(observation, str):
                prompt += "\nInitial observation:\n" + observation
            else:
                # Images are available through observe and stay on the public tool channel.
                prompt += "\nCall worldline.observe to obtain the initial images and state."
            from worldline.runtime.container_guard import process_identity

            guard = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "worldline.runtime.container_guard",
                    str(os.getpid()),
                    process_identity(os.getpid()),
                    container,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, "WORLDLINE_BRIDGE_KEY": gateway.token},
                bufsize=0,
                start_new_session=True,
            )
            process.stdin.write(prompt.encode())
            process.stdin.close()
            pending = bytearray()
            while process.poll() is None:
                if gateway.failure:
                    raise gateway.failure
                if time.monotonic() >= self.runtime["deadline"] or (self.cancel_check and self.cancel_check()):
                    raise BudgetExceeded("Codex task deadline or lease exhausted")
                if select.select([process.stdout], [], [], 0.1)[0]:
                    chunk = os.read(process.stdout.fileno(), 65536)
                    pending.extend(chunk)
                    if len(pending) > 2 * 1024 * 1024:
                        raise AgentError("Codex event exceeds byte budget")
                    while b"\n" in pending:
                        line, _, remainder = pending.partition(b"\n")
                        pending = bytearray(remainder)
                        text = line.decode(errors="replace").replace(gateway.token, "<redacted>")
                        try:
                            event = json.loads(text)
                        except ValueError:
                            event = {"log": text[:4000]}
                        self.runtime["emit"]("codex.event", event)
            if gateway.failure:
                raise gateway.failure
            if process.returncode:
                raise ModelAPIError("Codex process failed; inspect codex.event entries in trace.jsonl", status=422)
        finally:
            remove_container(container)
            if process:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                process.stdout.close()
            if guard:
                guard.stdin.close()
                guard.wait(timeout=20)
            gateway.close()
