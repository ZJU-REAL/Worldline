"""Small reproducible storage/trace benchmark, without any model API calls."""

from __future__ import annotations

import base64
import io
import json
import random
import time
from pathlib import Path

from worldline.media import MediaStore, hydrate_media
from worldline.run import write_json
from worldline.trace import TraceEvent, TraceWriter


def profile_trace(output_dir: Path, *, events=100, unique_images=1):
    from PIL import Image

    if events < 1 or not 1 <= unique_images <= events:
        raise ValueError("require events >= unique_images >= 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("profile output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    images = []
    rng = random.Random(0)
    for _ in range(unique_images):
        buffer = io.BytesIO()
        Image.frombytes("RGB", (128, 128), rng.randbytes(128 * 128 * 3)).save(buffer, format="PNG")
        images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    results = {}
    for name, sync_every in (("inline", 1), ("content-addressed", 32)):
        root = output_dir / name
        media = MediaStore(root) if name == "content-addressed" else None
        trace = TraceWriter(root / "trace.jsonl", sync_every=sync_every, media_store=media)
        started = time.perf_counter()
        for index in range(events):
            trace.append(
                TraceEvent(
                    "profile", "environment.observation", {"index": index, "image": images[index % unique_images]}
                )
            )
        trace.close()
        elapsed = time.perf_counter() - started
        results[name] = {
            "elapsed_sec": elapsed,
            "bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
            "media_files": media.files_written if media else 0,
            "fsync_calls": events + 1 if sync_every == 1 else events // 32 + 1,
        }
    first = json.loads((output_dir / "content-addressed/trace.jsonl").read_text().splitlines()[0])
    assert hydrate_media(first["payload"], output_dir / "content-addressed")["image"] == images[0]
    report = {
        "schema_version": "worldline.profile/0.1",
        "events": events,
        "unique_images": unique_images,
        "cases": results,
        "storage_reduction": 1 - results["content-addressed"]["bytes"] / results["inline"]["bytes"],
        "scope": "Synthetic valid PNG observations; storage and trace I/O only, no inference/simulator speed claim",
    }
    write_json(output_dir / "report.json", report)
    return report


def profile_dataset(output_dir: Path, *, frames=50, unique_images=10, image_size=256):
    """Measure the real export pipeline on temporary synthetic image trajectories."""
    import tempfile
    import tracemalloc
    from PIL import Image
    from worldline.datasets import export_dataset

    if not 1 <= unique_images <= frames <= 256 or not 64 <= image_size <= 1024:
        raise ValueError("require 1 <= unique_images <= frames <= 256 and image_size in [64, 1024]")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("profile output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix="worldline-data-profile-") as temp:
        root = Path(temp)
        origin = root / "run"
        store = MediaStore(origin)
        rng = random.Random(0)
        pictures, frame_sizes = [], []
        for _ in range(unique_images):
            buffer = io.BytesIO()
            Image.frombytes("RGB", (image_size, image_size), rng.randbytes(image_size * image_size * 3)).save(
                buffer, format="PNG"
            )
            frame_sizes.append(buffer.tell())
            pictures.append(store.put(base64.b64encode(buffer.getvalue()).decode(), "image/png", "data_url"))
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": pictures[index % unique_images]}}]}
            for index in range(frames)
        ]
        messages.append({"role": "assistant", "content": "Synthetic I/O profile only."})
        write_json(origin / "agent/trajectory.json", {"schema_version": "worldline.messages/0.2", "messages": messages})
        write_json(
            origin / "result.normalized.json",
            {
                "schema_version": "worldline.result/0.1",
                "backend": "profile",
                "trial_id": "profile",
                "task_id": "worldline/io-profile",
                "outcome": "success",
                "reward": 1,
                "source": str(origin),
                "task_metadata": {"split": "train", "synthetic_io_fixture": True},
                "artifacts": [{"kind": "agent_trace", "path": "agent/trajectory.json"}],
            },
        )
        for mode in ("inline", "reference"):
            output = root / mode / "sft.jsonl"
            tracemalloc.start()
            started = time.perf_counter()
            manifest = export_dataset([origin], output, format="sft", media_mode=mode)
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results[mode] = {
                "elapsed_sec": elapsed,
                "peak_python_bytes": peak,
                "jsonl_bytes": output.stat().st_size,
                "total_bytes": sum(path.stat().st_size for path in output.parent.rglob("*") if path.is_file()),
                "media_files": manifest["media"]["files"],
            }
    report = {
        "schema_version": "worldline.data_profile/0.1",
        "frames": frames,
        "unique_images": unique_images,
        "image_size": image_size,
        "logical_frame_bytes": sum(frame_sizes[index % unique_images] for index in range(frames)),
        "cases": results,
        "peak_memory_reduction": 1 - results["reference"]["peak_python_bytes"] / results["inline"]["peak_python_bytes"],
        "scope": "Synthetic valid PNGs and temporary fixture verdict; export I/O and Python allocation only. All synthetic data is removed; no model/simulator speed or task-score claim.",
    }
    write_json(output_dir / "report.json", report)
    return report
