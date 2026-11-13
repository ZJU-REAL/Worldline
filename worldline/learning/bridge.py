"""Episode lifecycle for external samplers, including per-tool release callbacks."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path
from collections import OrderedDict

from worldline.learning.rollout import NativeTrainingSession
from worldline.data_index import DiskIndex
from worldline.media import hydrate_media


class EpisodeRegistry:
    """Keep environment state across calls. Tool release never ends an episode.

    Trusted trainer code supplies task keys and generated assistant messages.
    Final reward always comes from the current environment's verifier.
    Repeated request IDs return the saved response instead of repeating actions.
    """

    def __init__(self, tasks, *, runs_dir=Path("runs/training"), resources=None, session_options=None, max_sessions=64):
        self.tasks = {key: Path(value) for key, value in tasks.items()}
        self.runs_dir, self.resources = runs_dir, resources
        self.options = session_options or {}
        self.sessions, self.responses = {}, {}
        self.lock = threading.RLock()
        self.completed = DiskIndex()
        if type(max_sessions) is not int or max_sessions < 1:
            self.completed.close()
            raise ValueError("max_sessions must be a positive integer")
        self.max_sessions = max_sessions
        self.admitting, self.closed = 0, False

    def begin(self, task_key):
        with self.lock:
            if self.closed:
                raise RuntimeError("training registry is closed")
            for ident, previous in list(self.sessions.items()):
                if previous.result is not None:
                    self.completed.set(ident, asdict(previous.training_outcome()))
                    self.sessions.pop(ident)
                    self.responses.pop(ident)
            if len(self.sessions) + self.admitting >= self.max_sessions:
                raise RuntimeError("training session capacity exhausted")
            self.admitting += 1
        try:
            session = NativeTrainingSession(
                self.tasks[task_key], runs_dir=self.runs_dir, resources=self.resources, **self.options
            )
            initial = session.start()
        except BaseException:
            with self.lock:
                self.admitting -= 1
            raise
        with self.lock:
            self.admitting -= 1
            if self.closed:
                session.abort(RuntimeError("registry closed during admission"))
                raise RuntimeError("training registry is closed")
            self.sessions[session.run_id] = session
            self.responses[session.run_id] = OrderedDict()
        return {"episode_id": session.run_id, **initial}

    def advance(self, episode_id, message, *, request_id):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id is required for mutating trainer calls")
        session = self.sessions[episode_id]
        fingerprint = hashlib.sha256(json.dumps(message, sort_keys=True, allow_nan=False).encode()).hexdigest()
        with session.lock:
            previous = self.responses[episode_id].get(request_id)
            if previous:
                if previous[0] != fingerprint:
                    raise ValueError("request_id was reused with different content")
                return hydrate_media(previous[1], session.run_dir)
            # Persist intent before any action. A crash after intent must be
            # discarded, never replayed into an unknown environment state.
            from worldline.run import write_json

            intent = session.run_dir / "trainer-requests" / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")
            if intent.exists():
                previous_intent = json.loads(intent.read_text())
                if previous_intent["fingerprint"] != fingerprint:
                    raise ValueError("request_id was reused with different content")
                if previous_intent["status"] == "completed":
                    return hydrate_media(previous_intent["result"], session.run_dir)
                session.abort(RuntimeError("previous trainer request has an ambiguous outcome"))
                raise RuntimeError("ambiguous trainer request must not be replayed")
            write_json(intent, {"fingerprint": fingerprint, "status": "started"})
            result = session.advance(message)
            saved = session.media.externalize(result)
            write_json(intent, {"fingerprint": fingerprint, "status": "completed", "result": saved})
            cache = self.responses[episode_id]
            cache[request_id] = (fingerprint, saved)
            if len(cache) > 128:
                cache.popitem(last=False)
            return result

    def context(self, episode_id, *, max_images=8):
        return self.sessions[episode_id].model_context(max_images)

    def release_tool(self, episode_id):
        return {"episode_id": episode_id, "retained": True}

    def finish(self, episode_id):
        with self.lock:
            if episode_id in self.completed:
                return self.completed.get(episode_id)
            result = asdict(self.sessions[episode_id].training_outcome())
            self.completed.set(episode_id, result)
            self.sessions.pop(episode_id)
            self.responses.pop(episode_id)
            return result

    def abort(self, episode_id, reason="trainer abandoned episode"):
        self.sessions[episode_id].abort(RuntimeError(reason))
        return self.finish(episode_id)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        for session in list(self.sessions.values()):
            if session.result is None:
                session.abort(RuntimeError("episode registry closed"))
        self.sessions.clear()
        self.responses.clear()
        self.completed.close()
