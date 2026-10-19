import hashlib
import json
from io import BytesIO

import pytest

from scripts import fetch_review_sources


def source_fixture(tmp_path, monkeypatch, pdf_bytes, expected=None):
    raw = pdf_bytes("A fixed benchmark contains thirteen cases.")
    actual = hashlib.sha256(raw).hexdigest()
    expected = expected or actual
    data = {
        "description": "Synthetic download contract, not research evaluation.",
        "sources": [{"id": "one", "filename": "one.pdf", "sha256": expected}],
        "cases": [
            {
                "id": "one-case",
                "source_id": "one",
                "question": "How many cases?",
                "language": "en",
                "expected_points": ["Thirteen."],
                "references": [{"page": 1, "text": "A fixed benchmark contains thirteen cases."}],
            }
        ],
    }
    dataset = tmp_path / "suite.json"
    dataset.write_text(json.dumps(data))
    origins = tmp_path / "origins.json"
    origins.write_text(json.dumps({expected: "https://arxiv.org/pdf/0000.00000v1"}))
    monkeypatch.setattr(fetch_review_sources, "ORIGINS", origins)
    monkeypatch.setattr(fetch_review_sources, "urlopen", lambda *args, **kwargs: BytesIO(raw))
    return dataset, raw, actual


def test_pinned_source_download_preserves_exact_bytes(tmp_path, monkeypatch, pdf_bytes):
    dataset, raw, _ = source_fixture(tmp_path, monkeypatch, pdf_bytes)
    output = tmp_path / "papers"
    fetch_review_sources.fetch(dataset, output)
    assert (output / "one.pdf").read_bytes() == raw
    with pytest.raises(ValueError, match="existing files are preserved"):
        fetch_review_sources.fetch(dataset, output)


def test_different_pdf_identity_reports_native_match_but_refuses_use(
    tmp_path, monkeypatch, pdf_bytes
):
    dataset, _, actual = source_fixture(tmp_path, monkeypatch, pdf_bytes, "0" * 64)
    output = tmp_path / "papers"
    with pytest.raises(ValueError, match="model evaluation not started"):
        fetch_review_sources.fetch(dataset, output)
    report = json.loads((output / "fetch-error.json").read_text())
    assert report["actual_sha256"] == actual
    assert report["native_reference_matches"] == [True]
    assert not (output / "one.pdf").exists()
