"""Atomic publication and durability shared by records, media and data packages."""

from __future__ import annotations

import os
import tempfile
import shutil
import hashlib
from contextlib import contextmanager
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_package(directory: Path):
    import json

    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("files"):
        raise ValueError("package has no integrity manifest")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or file_sha256(path) != expected:
            raise ValueError(f"package file missing or changed: {name}")
    return manifest


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_bytes(path: Path, raw: bytes, *, exclusive: bool = False) -> bool:
    """Publish fully synced bytes. Exclusive publication never exposes a partial blob."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        else:
            os.replace(temporary, path)
        sync_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def publish_file(source: Path, path: Path, *, sha256: str) -> bool:
    """Copy and verify a blob in bounded chunks before exclusive publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as outgoing, source.open("rb") as incoming:
            for chunk in iter(lambda: incoming.read(256 * 1024), b""):
                digest.update(chunk)
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if digest.hexdigest() != sha256:
            raise ValueError("media artifact hash mismatch during copy")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if file_sha256(path) != sha256:
                raise ValueError("existing media artifact hash mismatch")
            return False
        sync_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def staged_directory(destination: Path):
    """Publish a complete data package, keeping failed builds apart for diagnosis."""
    destination = destination.resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("output directory must be empty or absent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        yield stage
        for path in stage.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        for path in sorted((p for p in stage.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            sync_directory(path)
        sync_directory(stage)
        os.rename(stage, destination)
        sync_directory(destination.parent)
    except BaseException as error:
        if stage.exists():
            failed = stage.with_name(stage.name.replace(".building-", ".failed-", 1))
            os.rename(stage, failed)
            error.add_note(f"Incomplete package retained for diagnosis: {failed}")
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
