"""Versioned model references, independent of tasks and Harness implementations."""

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from worldline.storage import file_sha256


def validate_url(url):
    p = urlparse(url)
    if p.scheme not in {"http", "https"} or not p.netloc or p.username or p.password or p.query or p.fragment:
        raise ValueError("model base URL must be HTTP(S), without credentials, query or fragment")
    return url.rstrip("/")


@dataclass(frozen=True)
class ModelConfig:
    reference: str
    backend: str = "auto"
    base_url: str | None = None
    name: str | None = None
    api_key_env: str = "WORLDLINE_API_KEY"
    revision: str | None = None
    protocol: str = "chat"
    vision: bool = False
    server_executable: str | None = None
    server_args: tuple[str, ...] = ()
    server_command: tuple[str, ...] = ()
    startup_timeout_sec: float = 600
    schema_version: str = "worldline.model/0.1"

    def __post_init__(self):
        if not isinstance(self.reference, str) or not self.reference.strip() or self.reference.startswith("-"):
            raise ValueError("model reference must be a nonempty identifier or weights path")
        if self.backend not in {"auto", "endpoint", "vllm", "llama-cpp", "command"}:
            raise ValueError("unknown model backend")
        if self.name is not None and (
            not isinstance(self.name, str) or not self.name.strip() or self.name.startswith("-")
        ):
            raise ValueError("served model name must be a nonempty identifier")
        if self.protocol not in {"chat", "responses"} or type(self.vision) is not bool:
            raise ValueError("invalid model protocol or vision capability")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
            raise ValueError("api_key_env must be an environment variable name")
        if (
            type(self.startup_timeout_sec) not in {int, float}
            or not math.isfinite(self.startup_timeout_sec)
            or not 0 < self.startup_timeout_sec <= 7200
        ):
            raise ValueError("startup timeout must be in (0,7200]")
        if self.schema_version != "worldline.model/0.1":
            raise ValueError("unsupported model configuration schema")
        if self.base_url:
            validate_url(self.base_url)
        if self.effective_backend == "endpoint" and not self.base_url:
            raise ValueError("endpoint backend requires --base-url")
        if self.effective_backend != "endpoint" and self.base_url:
            raise ValueError("owned serving chooses its own URL; omit --base-url")
        if self.effective_backend == "endpoint" and (self.server_args or self.server_command or self.server_executable):
            raise ValueError("server options do not apply to an existing endpoint")
        if self.effective_backend == "command" and not self.server_command:
            raise ValueError("command backend requires --server-command as a JSON argv array")
        if self.effective_backend != "command" and self.server_command:
            raise ValueError("--server-command requires --model-backend command")
        if self.effective_backend == "llama-cpp" and not Path(self.reference).expanduser().is_file():
            raise ValueError("llama-cpp requires a local GGUF file")
        for key in ("server_args", "server_command"):
            if not isinstance(getattr(self, key), (list, tuple)):
                raise ValueError(f"{key} must be an argv array")
            object.__setattr__(self, key, tuple(getattr(self, key)))
        for value in (*self.server_args, *self.server_command):
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("server argv must contain strings without NUL")
            if re.search(r"(?i)(api[-_]?key|password|secret|access[-_]?token|bearer)", value):
                raise ValueError("server argv must not contain credentials; use environment variables")
        reserved = {"--host", "--port", "--model", "--alias", "--served-model-name", "--revision", "-m"}
        if any(value.split("=")[0] in reserved for value in self.server_args):
            raise ValueError("server args cannot override the model, bind address, alias or revision")

    @property
    def effective_backend(self):
        if self.backend != "auto":
            return self.backend
        if self.base_url:
            return "endpoint"
        return "llama-cpp" if self.reference.lower().endswith(".gguf") else "vllm"

    def identity(self):
        value = asdict(self)
        value["backend"] = self.effective_backend
        path = Path(self.reference).expanduser()
        if path.exists():
            value["reference"] = str(path.resolve())
            if path.is_file():
                value["weights_sha256"] = file_sha256(path)
            else:
                # A local directory can contain many GB. Record its inventory;
                # an explicit checkpoint revision is needed for immutable provenance.
                value["local_inventory"] = {
                    str(p.relative_to(path)): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                    for p in sorted(path.rglob("*"))
                    if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model"}
                }
        return value


@dataclass(frozen=True)
class ResolvedModel:
    name: str
    base_url: str
    api_key_env: str
    revision: str | None
    protocol: str
    vision: bool
    settings: dict

    @property
    def identity_url(self):
        return self.base_url if self.settings["backend"] == "endpoint" else f"owned://{self.settings['backend']}"
