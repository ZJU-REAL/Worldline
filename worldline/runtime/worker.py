"""Trusted JSONL worker. Only the runner can request a verdict."""

from __future__ import annotations

import contextlib
import json
import sys
import os
import signal
from dataclasses import asdict

from worldline.contracts import Task
from worldline.errors import InvalidAction
from worldline.plugins import get_environment_plugin
from worldline.media import MediaStore
from pathlib import Path


def main() -> int:
    # A killed launcher must not leave a detached simulator/browser consuming resources.
    if sys.platform == "linux":
        import ctypes

        parent = os.getppid()

        def terminate(*_):
            # The launcher no longer owns a usable session. Kill the dedicated
            # group directly, including descendants that ignore SIGTERM.
            os.killpg(os.getpgrp(), signal.SIGKILL)

        signal.signal(signal.SIGTERM, terminate)
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != parent:
            terminate()
    environment = None
    media = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            method = request.get("method")
            try:
                # Simulator import/initialization chatter belongs in worker.log.
                with contextlib.redirect_stdout(sys.stderr):
                    if method == "start" and environment is None:
                        task = Task(**request["task"])
                        plugin = get_environment_plugin(task.environment)
                        if plugin.version != task.metadata["environment_version"]:
                            raise ValueError("environment version changed after task loading")
                        environment = plugin.factory(task, request["config"])
                        if request.get("artifact_root"):
                            media = MediaStore(Path(request["artifact_root"]))
                        info = getattr(environment, "execution_info", None)
                        result = {
                            "spec": environment.spec(),
                            "observation": environment.observe(),
                            "execution": info() if callable(info) else {},
                        }
                    elif environment is None:
                        raise RuntimeError("environment is not started")
                    elif method == "observe":
                        result = environment.observe()
                    elif method == "act":
                        result = environment.step(request["action"])
                    elif method == "reset":
                        result = environment.reset(request["seed"])
                    elif method == "verify":
                        verdict = environment.verify()
                        result = {**asdict(verdict), "evidence": [str(path) for path in verdict.evidence]}
                    elif method == "checkpoint" and environment.spec().get("capabilities", {}).get("snapshot"):
                        result = environment.checkpoint()
                    elif method == "restore" and environment.spec().get("capabilities", {}).get("snapshot"):
                        result = environment.restore(request["checkpoint"])
                    elif method == "close":
                        break
                    else:
                        raise InvalidAction(f"unsupported operation: {method}")
                response = {"result": result}
                if media is not None:
                    response = media.externalize(response)
            except Exception as error:
                response = {"error": {"type": type(error).__name__, "message": str(error)}}
            sys.stdout.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()
    finally:
        if environment is not None:
            with contextlib.redirect_stdout(sys.stderr):
                environment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
