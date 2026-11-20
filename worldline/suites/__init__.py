"""Small packaged integration suites; external benchmarks remain external tasks."""

from pathlib import Path


def builtin_suites():
    return {
        "rua-smoke": {"task": "rua/reach-test", "vision": False, "purpose": "integration_fixture"},
        "cua-smoke": {"task": "cua/form-test", "vision": True, "purpose": "integration_fixture"},
    }


def suite_tasks(name):
    try:
        suite = builtin_suites()[name]
    except KeyError as error:
        raise ValueError(f"unknown suite: {name}; use worldline suites, or pass --tasks") from error
    bundled = Path(__file__).parent / "tasks" / suite["task"]
    source = Path(__file__).resolve().parents[2] / "examples" / suite["task"]
    path = bundled if bundled.is_dir() else source
    if not (path / "worldline.toml").is_file():
        raise ValueError(f"suite files missing: {path}")
    return [path.resolve()], suite
