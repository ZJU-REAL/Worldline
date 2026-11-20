"""Own a serving process group until the controller's private stdin closes."""

import json
import os
import select
import signal
import subprocess
import sys
import time


def main():
    command = json.loads(sys.stdin.buffer.readline())
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)

    def stopped(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        while process.poll() is None:
            if select.select([sys.stdin], [], [], 0.1)[0] and not os.read(sys.stdin.fileno(), 1):
                break
        return process.poll() or 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            time.sleep(0.2)
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
