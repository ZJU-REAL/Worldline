"""Host-wide leases with kernel crash cleanup and cancellable admission."""

from __future__ import annotations

import fcntl
import time
from pathlib import Path

from worldline.runtime.provisioning import state_root


class HostResourcePool:
    """CPU slots and physical GPU names are shared by all local runners.

    flock leases live as long as their descriptors; kernel release also handles
    SIGKILL. An explicit state directory identifies one resource namespace.
    """

    def __init__(self, cpus, gpu_devices, *, directory=None):
        if type(cpus) is not int or cpus < 1 or len(set(gpu_devices)) != len(gpu_devices):
            raise ValueError("invalid CPU capacity or duplicate GPU devices")
        self.capacity, self.gpu_capacity = cpus, len(gpu_devices)
        self.devices = tuple(str(x) for x in gpu_devices)
        if any(not x or "/" in x or "\\" in x or x in {".", ".."} for x in self.devices):
            raise ValueError("GPU names must be identifiers, not paths")
        self.directory = Path(directory or state_root() / "resources")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.leases = {}
        import threading

        self.lock = threading.Lock()

    def _claim(self, name):
        stream = (self.directory / name).open("a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return stream
        except BlockingIOError:
            stream.close()
            return None

    def acquire(self, cpus, gpu, *, cancel=None, timeout=None, owner=None):
        if type(cpus) is not int or cpus < 1 or cpus > self.capacity or gpu and not self.gpu_capacity:
            raise ValueError("task resources exceed host pool capacity")
        start = time.monotonic()
        while True:
            if cancel and cancel():
                raise InterruptedError("resource admission cancelled")
            if timeout is not None and time.monotonic() - start >= timeout:
                raise TimeoutError("resource admission timed out")
            # Serialize partial slot selection so two multi-CPU requests cannot
            # each hold a fraction of capacity and deadlock.
            with (self.directory / "admission.lock").open("a") as admission:
                fcntl.flock(admission, fcntl.LOCK_EX)
                held, device = [], None
                for index in range(self.capacity):
                    stream = self._claim(f"cpu-{index}.lock")
                    if stream:
                        held.append(stream)
                    if len(held) == cpus:
                        break
                if len(held) == cpus and gpu:
                    for candidate in self.devices:
                        stream = self._claim(f"gpu-{candidate}.lock")
                        if stream:
                            held.append(stream)
                            device = candidate
                            break
                if len(held) == cpus + int(bool(gpu)):
                    import threading

                    with self.lock:
                        self.leases.setdefault((owner or threading.get_ident(), cpus, device), []).append(held)
                    return device
                for stream in held:
                    stream.close()
            time.sleep(0.05)

    def release(self, cpus, device, *, owner=None):
        import threading

        key = (owner or threading.get_ident(), cpus, device)
        with self.lock:
            # A session's watchdog can release from another thread.
            if not self.leases.get(key):
                keys = [k for k, value in self.leases.items() if k[1:] == (cpus, device) and value]
                if len(keys) != 1:
                    raise ValueError("resource release does not identify one lease")
                key = keys[0]
            held = self.leases[key].pop()
        for stream in held:
            stream.close()

    def snapshot(self):
        return {
            "backend": "host-flock",
            "directory": str(self.directory),
            "cpu_capacity": self.capacity,
            "gpu_devices": self.devices,
        }
