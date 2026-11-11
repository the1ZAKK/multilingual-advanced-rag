import gzip
import hashlib
import json

import pytest

from scripts.preserve_generation_trace import preserve_trace


@pytest.mark.parametrize("include_server_log", [False, True])
def test_trace_preserves_rejected_events_and_exact_multilingual_bytes(tmp_path, include_server_log):
    raw = (
        '{"stage":"generation","result":{"text":"سؤال"}}\n'
        '{"stage":"validation-rejection","result":{"error":"wrong role"}}\n'
    ).encode()
    trace = tmp_path / "wire.jsonl"
    trace.write_bytes(raw)
    output = tmp_path / "preserved"
    server = tmp_path / "server.log"
    server_raw = "prompt cache is disabled\nworker failed: خطأ\n".encode()
    server.write_bytes(server_raw)
    result = preserve_trace(
        trace, output, "a" * 40, server_log=server if include_server_log else None
    )
    compressed = (output / "generation-wire.jsonl.gz").read_bytes()
    assert gzip.decompress(compressed) == raw
    assert result["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["compressed_sha256"] == hashlib.sha256(compressed).hexdigest()
    assert result["stages"] == {"generation": 1, "validation-rejection": 1}
    assert result["events"] == 2 and result["independent_human_reviews"] == 0
    if include_server_log:
        server_compressed = (output / "model-server.log.gz").read_bytes()
        assert gzip.decompress(server_compressed) == server_raw
        assert result["server_diagnostics"]["raw_sha256"] == hashlib.sha256(server_raw).hexdigest()
        assert (
            result["server_diagnostics"]["compressed_sha256"]
            == hashlib.sha256(server_compressed).hexdigest()
        )
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
