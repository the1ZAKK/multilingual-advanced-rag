"""Fetch only hash-pinned public evaluation PDFs, then verify native-page gold quotes."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pymupdf

from rag.evaluation.review import ReviewSuite, verify_sources

ORIGINS = Path(__file__).resolve().parents[1] / "data/evaluation/source-origins.json"


def fetch(dataset: Path, output: Path):
    suite = ReviewSuite.model_validate_json(dataset.read_text(encoding="utf-8"))
    if output.exists():
        raise ValueError("Choose a new source directory; existing files are preserved")
    origins = json.loads(ORIGINS.read_text())
    output.mkdir(parents=True)
    for source in suite.sources:
        url = origins[source.sha256]
        if urlparse(url).scheme != "https" or urlparse(url).hostname != "arxiv.org":
            raise ValueError("Evaluation source must be a pinned official HTTPS arXiv URL")
        with urlopen(url, timeout=120) as response:
            data = response.read(26 * 1024 * 1024)
        actual = hashlib.sha256(data).hexdigest()
        if actual != source.sha256:
            references = [
                q for case in suite.cases if case.source_id == source.id for q in case.references
            ]
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                matches = [
                    1 <= q.page <= len(pdf) and q.text in pdf[q.page - 1].get_text()
                    for q in references
                ]
                pages = len(pdf)
            diagnostic = dict(
                source_id=source.id,
                url=url,
                expected_sha256=source.sha256,
                actual_sha256=actual,
                pages=pages,
                native_reference_matches=matches,
            )
            (output / "fetch-error.json").write_text(json.dumps(diagnostic, indent=2))
            raise ValueError(
                f"Pinned official source hash mismatch: {source.id}; model evaluation not started"
            )
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
