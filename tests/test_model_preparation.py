import hashlib
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from scripts import prepare_model


def fake_download(monkeypatch, path):
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: str(path)),
    )


def test_checksum_failure_prevents_any_model_import(tmp_path, monkeypatch):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"not official model bytes")
    fake_download(monkeypatch, path)
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: pytest.fail("Unverified model reached HTTP import")
    )
    with pytest.raises(ValueError, match="checksum verification failed"):
        prepare_model.prepare(tmp_path / "cache", tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def test_verified_blob_upload_and_report(tmp_path, monkeypatch):
    path = tmp_path / "tiny.gguf"
    data = b"contract fixture only"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    fake_download(monkeypatch, path)
    monkeypatch.setattr(prepare_model, "SHA256", digest)
    monkeypatch.setattr(prepare_model, "SIZE", len(data))
    imported = []
    real_client = httpx.Client

    def handle(request):
        if request.method == "HEAD":
            return httpx.Response(404)
        if "/api/blobs/" in request.url.path:
            assert request.read() == data
            return httpx.Response(201)
        if request.url.path == "/api/create":
            body = json.loads(request.content)
            assert body["files"] == {prepare_model.FILENAME: "sha256:" + digest}
            imported.append(body["model"])
            return httpx.Response(200, json={"status": "success"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "qwen3-hf:4b", "digest": "fixture"}]}
            )
        return httpx.Response(200, json={"version": "fixture"})

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle))
    )
    result = prepare_model.prepare(tmp_path / "cache", tmp_path / "report.json")
    assert imported == ["qwen3-hf:4b"]
    assert result["sha256"] == digest
    assert json.loads((tmp_path / "report.json").read_text())["bytes"] == len(data)
