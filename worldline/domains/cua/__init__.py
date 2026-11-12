"""Computer use: BrowserGym environments share the native Worldline runtime."""

import hashlib
from pathlib import Path
from worldline.plugins import EnvironmentPlugin


def plugins():
    from worldline.adapters.browsergym import BrowserGymEnvironment

    version = hashlib.sha256(Path(BrowserGymEnvironment.source_path()).read_bytes()).hexdigest()
    return (
        EnvironmentPlugin(
            "cua.browsergym",
            "cua",
            f"browsergym-0.14.3/{version}",
            BrowserGymEnvironment,
            preflight=BrowserGymEnvironment.preflight,
        ),
    )
