"""Cross-process endpoint backpressure; credentials never enter shared state."""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
from contextlib import contextmanager

from worldline.errors import BudgetExceeded
from worldline.runtime.provisioning import state_root
from worldline.run import write_json


class EndpointLimiter:
    def __init__(self, endpoint, *, concurrency=4, requests_per_minute=0):
        if (
            type(concurrency) is not int
            or concurrency < 1
            or type(requests_per_minute) is not int
            or requests_per_minute < 0
        ):
            raise ValueError("invalid endpoint concurrency or request rate")
        self.concurrency, self.rpm = concurrency, requests_per_minute
        self.directory = state_root() / "endpoints" / hashlib.sha256(endpoint.encode()).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        policy = {"concurrency": concurrency, "requests_per_minute": requests_per_minute}
        with (self.directory / "policy.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            path = self.directory / "policy.json"
            raw = path.read_text() if path.exists() else ""
            if raw and json.loads(raw) != policy:
                raise ValueError(
                    "all workers for one endpoint must use the same concurrency/rate policy; reset policy only after stopping workers"
                )
            if not raw:
                write_json(path, policy)

    @contextmanager
    def request(self, deadline, cancel_check=None):
        slot = None
        try:
            while slot is None:
                if cancel_check and cancel_check():
                    raise BudgetExceeded("endpoint admission cancelled")
                if time.monotonic() >= deadline:
                    raise BudgetExceeded("endpoint admission deadline exhausted")
                for index in range(self.concurrency):
                    candidate = (self.directory / f"slot-{index}.lock").open("a")
                    try:
                        fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        slot = candidate
                        break
                    except BlockingIOError:
                        candidate.close()
                if slot is None:
                    time.sleep(0.05)
            if self.rpm:
                while True:
                    if cancel_check and cancel_check():
                        raise BudgetExceeded("endpoint rate wait cancelled")
                    now = time.monotonic()
                    with (self.directory / "rate.lock").open("a") as stream:
                        fcntl.flock(stream, fcntl.LOCK_EX)
                        path = self.directory / "rate.json"
                        raw = path.read_text() if path.exists() else ""
                        recent = [stamp for stamp in json.loads(raw or "[]") if now - 60 < stamp <= now]
                        if len(recent) < self.rpm:
                            recent.append(now)
                            write_json(path, recent)
                            break
                        delay = max(0.05, min(0.2, recent[0] + 60 - now))
                    if time.monotonic() + delay >= deadline:
                        raise BudgetExceeded("endpoint rate budget exhausted")
                    time.sleep(delay)
            yield
        finally:
            if slot:
                slot.close()
