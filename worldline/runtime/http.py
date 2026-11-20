"""Deadline-bound model HTTP transport, including DNS and slow response bodies."""

from __future__ import annotations

import base64
import json
import os
import select
import signal
import subprocess
import sys
import time

from worldline.errors import HTTPRequestError


def http_json(request, deadline, max_bytes, cancel_check=None):
    """Credentials travel only on the private stdin pipe, never in argv or logs."""
    process = subprocess.Popen(
        [sys.executable, "-m", "worldline.runtime.http_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        bufsize=0,
    )
    try:
        payload = (
            json.dumps(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "headers": dict(request.header_items()),
                    "body": request.data.decode() if request.data is not None else None,
                    "max_bytes": max_bytes,
                    "timeout": max(0.001, deadline - time.monotonic()),
                },
                ensure_ascii=False,
            ).encode()
            + b"\n"
        )
        os.set_blocking(process.stdin.fileno(), False)
        view = memoryview(payload)
        offset = 0
        while offset < len(payload):
            remaining = deadline - time.monotonic()
            if cancel_check and cancel_check():
                raise HTTPRequestError("HTTP request cancelled")
            if remaining <= 0:
                raise HTTPRequestError("HTTP request deadline exhausted")
            if not select.select([], [process.stdin], [], min(remaining, 0.2))[1]:
                continue
            try:
                offset += os.write(process.stdin.fileno(), view[offset:])
            except BlockingIOError:
                pass
        response = bytearray()
        while not response.endswith(b"\n"):
            remaining = deadline - time.monotonic()
            if cancel_check and cancel_check():
                raise HTTPRequestError("HTTP response cancelled")
            if remaining <= 0:
                raise HTTPRequestError("HTTP response deadline exhausted")
            if not select.select([process.stdout], [], [], min(remaining, 0.2))[0]:
                continue
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise HTTPRequestError("HTTP worker exited without a response")
            response.extend(chunk)
            if len(response) > max_bytes * 2 + 65536:
                raise HTTPRequestError("HTTP response exceeds transport byte budget", status=422)
        result = json.loads(response)
        if result.get("error"):
            raise HTTPRequestError(result["error"], status=result.get("status"), retry_after=result.get("retry_after"))
        raw = base64.b64decode(result["body"], validate=True)
        if len(raw) > max_bytes:
            raise HTTPRequestError("HTTP response exceeds byte budget", status=422)
        try:
            return json.loads(raw)
        except ValueError as error:
            raise HTTPRequestError("HTTP endpoint returned malformed JSON", status=422) from error
    finally:
        if process.poll() is None:
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        process.stdin.close()
        process.stdout.close()
