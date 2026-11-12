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
