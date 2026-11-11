"""Preserve every wire event losslessly with source and byte-hash metadata."""

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def preserve_trace(trace: Path, output: Path, source_sha: str, *, server_log: Path | None = None):
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("A full source commit SHA is required")
    archive = output / "generation-wire.jsonl.gz"
    manifest_path = output / "generation-wire-manifest.json"
    server_archive = output / "model-server.log.gz"
    if archive.exists() or manifest_path.exists() or (server_log and server_archive.exists()):
        raise ValueError("Refusing to overwrite preserved generation evidence")
    raw = trace.read_bytes()
    events = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines()]
    stages = Counter(event["stage"] for event in events)
    compressed = gzip.compress(raw, mtime=0)
    manifest = dict(
        source_code_sha=source_sha,
        file=archive.name,
        raw_bytes=len(raw),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        compressed_bytes=len(compressed),
        compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        events=len(events),
        stages=dict(sorted(stages.items())),
        independent_human_reviews=0,
        review_status="Ungraded wire evidence, including rejected and failed attempts",
    )
    server_compressed = None
    if server_log is not None:
        server_raw = server_log.read_bytes()
        server_compressed = gzip.compress(server_raw, mtime=0)
        manifest["server_diagnostics"] = dict(
            file=server_archive.name,
            raw_bytes=len(server_raw),
            raw_sha256=hashlib.sha256(server_raw).hexdigest(),
            compressed_bytes=len(server_compressed),
            compressed_sha256=hashlib.sha256(server_compressed).hexdigest(),
        )
    output.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as file:
        file.write(compressed)
    if server_compressed is not None:
        with server_archive.open("xb") as file:
            file.write(server_compressed)
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--server-log", type=Path)
    args = parser.parse_args()
    manifest = preserve_trace(args.trace, args.output, args.source_sha, server_log=args.server_log)
    print(f"Preserved {manifest['events']} ungraded wire events")


if __name__ == "__main__":
    main()
