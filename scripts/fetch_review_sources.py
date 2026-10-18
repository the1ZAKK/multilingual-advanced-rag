"""Fetch only hash-pinned public evaluation PDFs, then verify native-page gold quotes."""

import argparse
import hashlib
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from rag.evaluation.review import ReviewSuite, verify_sources

REVISION = "18bc2fb43bd202cd68acda3ce231a935746c0431"


def fetch(dataset: Path, output: Path):
    suite = ReviewSuite.model_validate_json(dataset.read_text(encoding="utf-8"))
    if output.exists():
        raise ValueError("Choose a new source directory; existing files are preserved")
    output.mkdir(parents=True)
    for source in suite.sources:
        url = f"https://raw.githubusercontent.com/the1ZAKK/project/{REVISION}/source%20papers/{quote(source.filename, safe='')}"
        with urlopen(url, timeout=120) as response:
            data = response.read(26 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != source.sha256:
            raise ValueError(f"Pinned public source hash mismatch: {source.id}")
        with (output / source.filename).open("xb") as file:
            file.write(data)
    verify_sources(suite, output)
    print(f"Verified {len(suite.sources)} public PDF hashes and all frozen reference quotes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.dataset, args.output)


if __name__ == "__main__":
    main()
