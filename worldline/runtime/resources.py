"""Single-host CPU admission and exclusive GPU leases."""

import threading


class ResourcePool:
    """Admission limits, not OS resource enforcement. A GPU lease is exclusive."""

    def __init__(self, cpus: int, gpu_devices):
        if type(cpus) is not int or cpus < 1:
            raise ValueError("CPU capacity must be a positive integer")
        if len(set(gpu_devices)) != len(gpu_devices):
            raise ValueError("GPU devices must be unique to guarantee exclusive leases")
        self.capacity = self.available = cpus
        self.devices = list(gpu_devices)
        self.gpu_capacity = len(self.devices)
        self.condition = threading.Condition()

    def acquire(self, cpus, gpu, *, cancel=None, timeout=None):
        import time

        if type(cpus) is not int or cpus < 1 or cpus > self.capacity or gpu and not self.gpu_capacity:
            raise ValueError("task resources exceed configured CPU/GPU capacity")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while self.available < cpus or gpu and not self.devices:
                if cancel and cancel():
                    raise InterruptedError("resource admission cancelled")
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("resource admission timed out")
                self.condition.wait(timeout=0.05)
            self.available -= cpus
            return self.devices.pop(0) if gpu else None

    def release(self, cpus, device):
        with self.condition:
            if (
                type(cpus) is not int
                or cpus < 1
                or self.available + cpus > self.capacity
                or device is not None
                and device in self.devices
            ):
                raise ValueError("invalid or duplicate resource release")
            self.available += cpus
            if device is not None:
                self.devices.append(device)
            self.condition.notify_all()
