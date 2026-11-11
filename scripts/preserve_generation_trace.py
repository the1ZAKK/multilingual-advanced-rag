"""Preserve every wire event losslessly with source and byte-hash metadata."""

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def preserve_trace(trace: Path, output: Path, source_sha: str):
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("A full source commit SHA is required")
    archive = output / "generation-wire.jsonl.gz"
    manifest_path = output / "generation-wire-manifest.json"
    if archive.exists() or manifest_path.exists():
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
    output.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as file:
        file.write(compressed)
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    manifest = preserve_trace(args.trace, args.output, args.source_sha)
    print(f"Preserved {manifest['events']} ungraded wire events")


if __name__ == "__main__":
    main()
