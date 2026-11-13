from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from worldline.runtime.shared_resources import HostResourcePool


class HostResourcesTest(unittest.TestCase):
    def test_independent_processes_cannot_share_gpu_and_crash_releases_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            script = "from worldline.runtime.shared_resources import HostResourcePool; import sys,time; p=HostResourcePool(2,('0',),directory=sys.argv[1]); p.acquire(1,True); print('ready',flush=True); time.sleep(30)"
            child = subprocess.Popen([sys.executable, "-c", script, temp], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "ready")
                pool = HostResourcePool(2, ("0",), directory=temp)
                with self.assertRaises(TimeoutError):
                    pool.acquire(1, True, timeout=0.1)
                child.kill()
                child.wait(timeout=5)
                device = pool.acquire(2, True, timeout=1)
                self.assertEqual(device, "0")
                pool.release(2, device)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                child.stdout.close()

    def test_wait_is_cancellable_and_watchdog_releases_by_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            pool = HostResourcePool(1, (), directory=Path(temp))
            pool.acquire(1, False, owner="episode")
            with self.assertRaises(InterruptedError):
                pool.acquire(1, False, cancel=lambda: True)
            thread = threading.Thread(target=lambda: pool.release(1, None, owner="episode"))
            thread.start()
            thread.join(timeout=2)
            pool.acquire(1, False, timeout=1)
            pool.release(1, None)
            with self.assertRaises(ValueError):
                pool.acquire(-1, False)
