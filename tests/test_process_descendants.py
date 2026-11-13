from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from worldline.runtime.process import ProcessEnvironment


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return False
    return path.read_text().split()[2] != "Z"


class ProcessDescendantTest(unittest.TestCase):
    def test_cleanup_kills_descendant_even_when_worker_has_already_exited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)']); print(p.pid,flush=True)"
            process = subprocess.Popen(
                [sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, start_new_session=True
            )
            pid = int(process.stdout.readline())
            process.wait(timeout=5)
            environment = object.__new__(ProcessEnvironment)
            environment.process, environment.log, environment._closed = process, (root / "log").open("wb"), False
            try:
                self.assertTrue(alive(pid))
                environment.close()
                limit = time.monotonic() + 2
                while alive(pid) and time.monotonic() < limit:
                    time.sleep(0.01)
                self.assertFalse(alive(pid))
                environment.close()
            finally:
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)

    def test_parent_death_reaps_a_child_that_ignores_term(self):
        with tempfile.TemporaryDirectory():
            child_code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)"
            worker_code = f"""import subprocess,sys,types
import worldline.runtime.worker as w
class Environment:
 def __init__(self,*_):
  self.child=subprocess.Popen([sys.executable,'-c',{child_code!r}])
 def spec(self):return {{}}
 def observe(self):return {{'child':self.child.pid}}
 def close(self):pass
w.get_environment_plugin=lambda _:types.SimpleNamespace(version='v1',factory=Environment)
w.main()
"""
            owner_code = f"""import subprocess,sys,json,time
p=subprocess.Popen([sys.executable,'-c',{worker_code!r}],stdin=subprocess.PIPE,stdout=subprocess.PIPE,start_new_session=True,text=True)
p.stdin.write(json.dumps({{'method':'start','task':{{'id':'task','instruction':'task','environment':'test','version':'v1','metadata':{{'environment_version':'v1'}}}},'config':{{}}}})+'\\n');p.stdin.flush()
response=json.loads(p.stdout.readline())
print(json.dumps({{'worker':p.pid,'child':response['result']['observation']['child']}}),flush=True)
time.sleep(120)
"""
            owner = subprocess.Popen(
                [sys.executable, "-c", owner_code], stdout=subprocess.PIPE, text=True, start_new_session=True
            )
            pids = json.loads(owner.stdout.readline())
            try:
                os.kill(owner.pid, signal.SIGKILL)
                owner.wait(timeout=5)
                limit = time.monotonic() + 4
                while (alive(pids["worker"]) or alive(pids["child"])) and time.monotonic() < limit:
                    time.sleep(0.02)
                self.assertFalse(alive(pids["worker"]))
                self.assertFalse(alive(pids["child"]))
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.wait()
                for pid in pids.values():
                    if alive(pid):
                        os.kill(pid, signal.SIGKILL)
                owner.stdout.close()
