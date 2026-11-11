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


@pytest.mark.parametrize("profile", ["base", "instruct", "qwen25"])
def test_verified_blob_upload_and_report(tmp_path, monkeypatch, profile):
    path = tmp_path / "tiny.gguf"
    data = b"contract fixture only"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    fake_download(monkeypatch, path)
    monkeypatch.setattr(prepare_model, "SHA256", digest)
    monkeypatch.setattr(prepare_model, "SIZE", len(data))
    profile_name = {"instruct": "INSTRUCT_PROFILE", "qwen25": "QWEN25_PROFILE"}.get(profile)
    if profile_name:
        monkeypatch.setattr(
            prepare_model,
            profile_name,
            {**getattr(prepare_model, profile_name), "sha256": digest, "bytes": len(data)},
        )
    expected_name = {
        "base": "qwen3-hf:4b",
        "instruct": "qwen3-instruct-hf:4b",
        "qwen25": "qwen25-instruct-hf:7b",
    }[profile]
    filename = (
        prepare_model.FILENAME
        if not profile_name
        else getattr(prepare_model, profile_name)["filename"]
    )
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
            assert body["files"] == {filename: "sha256:" + digest}
            imported.append(body["model"])
            return httpx.Response(200, json={"status": "success"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": expected_name, "digest": "fixture"}]}
            )
        return httpx.Response(200, json={"version": "fixture"})

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle))
    )
    result = prepare_model.prepare(tmp_path / "cache", tmp_path / "report.json", profile=profile)
    assert imported == [expected_name]
    assert result["profile"] == profile
    assert result["sha256"] == digest
    assert json.loads((tmp_path / "report.json").read_text())["bytes"] == len(data)


def test_instruct_profile_pins_download_and_rejects_other_model_bytes(tmp_path, monkeypatch):
    path = tmp_path / "wrong-model.gguf"
    path.write_bytes(b"different model")
    calls = []

    def download(repository, filename, **kwargs):
        calls.append((repository, filename, kwargs["revision"]))
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: pytest.fail("Wrong model reached import"))
    with pytest.raises(ValueError, match="checksum verification failed"):
        prepare_model.prepare(tmp_path / "cache", tmp_path / "report.json", profile="instruct")
    assert calls == [
        (
            "unsloth/Qwen3-4B-Instruct-2507-GGUF",
            "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            "a06e946bb6b655725eafa393f4a9745d460374c9",
        )
    ]
    assert not (tmp_path / "report.json").exists()


def test_qwen25_comparison_profile_verifies_revision_and_bytes_before_import(tmp_path, monkeypatch):
    path = tmp_path / "wrong-model.gguf"
    path.write_bytes(b"wrong comparison weights")
    calls = []

    def download(repository, filename, **kwargs):
        calls.append((repository, filename, kwargs["revision"]))
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: pytest.fail("Unverified model reached import")
    )
    with pytest.raises(ValueError, match="checksum verification failed"):
        prepare_model.prepare(tmp_path / "cache", tmp_path / "report.json", profile="qwen25")
    assert calls == [
        (
            "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031",
        )
    ]
    assert not (tmp_path / "report.json").exists()
