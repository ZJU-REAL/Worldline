"""One HTTP request in an isolated I/O process; no credentials in output."""

import base64
import json
import os
import signal
import sys
import urllib.error
import urllib.request


def main():
    if sys.platform == "linux":
        import ctypes

        parent = os.getppid()
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)
        if os.getppid() != parent:
            return 1
    payload = json.loads(sys.stdin.buffer.readline())
    request = urllib.request.Request(payload["url"], data=payload["body"].encode(), headers=payload["headers"])
    try:
        chunks, size = [], 0
        with urllib.request.urlopen(request, timeout=payload["timeout"]) as response:
            reader = getattr(response, "read1", response.read)
            while True:
                chunk = reader(min(65536, payload["max_bytes"] - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > payload["max_bytes"]:
                    result = {"error": "model response exceeds byte budget", "status": 422}
                    break
                chunks.append(chunk)
            if size <= payload["max_bytes"]:
                result = {"body": base64.b64encode(b"".join(chunks)).decode()}
    except urllib.error.HTTPError as error:
        try:
            retry_after = float(error.headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            retry_after = None
        result = {"error": f"model API returned HTTP {error.code}", "status": error.code, "retry_after": retry_after}
    except OSError as error:
        result = {"error": f"model API transport failed: {type(error).__name__}"}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
