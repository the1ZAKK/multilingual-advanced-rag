import gzip
import hashlib
import json

import pytest

from scripts.preserve_generation_trace import preserve_trace


def test_trace_preserves_rejected_events_and_exact_multilingual_bytes(tmp_path):
    raw = (
        '{"stage":"generation","result":{"text":"سؤال"}}\n'
        '{"stage":"validation-rejection","result":{"error":"wrong role"}}\n'
    ).encode()
    trace = tmp_path / "wire.jsonl"
    trace.write_bytes(raw)
    output = tmp_path / "preserved"
    result = preserve_trace(trace, output, "a" * 40)
    compressed = (output / "generation-wire.jsonl.gz").read_bytes()
    assert gzip.decompress(compressed) == raw
    assert result["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["compressed_sha256"] == hashlib.sha256(compressed).hexdigest()
    assert result["stages"] == {"generation": 1, "validation-rejection": 1}
    assert result["events"] == 2 and result["independent_human_reviews"] == 0
    assert json.loads((output / "generation-wire-manifest.json").read_text()) == result
    with pytest.raises(ValueError, match="overwrite"):
        preserve_trace(trace, output, "b" * 40)
    assert (output / "generation-wire.jsonl.gz").read_bytes() == compressed


def test_trace_rejects_malformed_evidence_before_writing(tmp_path):
    trace = tmp_path / "wire.jsonl"
    trace.write_text('{"stage":"generation"}\nBROKEN\n')
    output = tmp_path / "preserved"
    with pytest.raises(json.JSONDecodeError):
        preserve_trace(trace, output, "a" * 40)
    assert not output.exists()
    with pytest.raises(ValueError, match="full source commit"):
        preserve_trace(trace, output, "candidate")
