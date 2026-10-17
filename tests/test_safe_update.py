from zipfile import ZipFile

import pytest

from scripts.apply_update import apply_update, digest


def fixture(tmp_path):
    source, target = tmp_path / "update", tmp_path / "installed"
    source.mkdir()
    target.mkdir()
    (source / "module.py").write_bytes(b"new code")
    (target / "module.py").write_bytes(b"old code")
    (target / ".env").write_bytes(b"private settings")
    (target / "data").mkdir()
    (target / "data/corpus.sqlite3").write_bytes(b"private data")
    manifest = {
        "files": [
            {
                "path": "module.py",
                "baseline_sha256": digest(b"old code"),
                "sha256": digest(b"new code"),
            }
        ]
    }
    return source, target, manifest


def test_upgrade_preflights_backs_up_and_preserves_data(tmp_path):
    source, target, manifest = fixture(tmp_path)
    assert apply_update(source, target, manifest, check_only=True)["pending_files"] == 1
    assert (target / "module.py").read_bytes() == b"old code"
    result = apply_update(source, target, manifest)
    assert (target / "module.py").read_bytes() == b"new code"
    assert (target / ".env").read_bytes() == b"private settings"
    assert (target / "data/corpus.sqlite3").read_bytes() == b"private data"
    with ZipFile(result["backup"]) as backup:
        assert backup.read("module.py") == b"old code"
    assert apply_update(source, target, manifest)["changed_files"] == 0


def test_upgrade_refuses_unshared_edits_before_any_writes(tmp_path):
    source, target, manifest = fixture(tmp_path)
    (target / "module.py").write_bytes(b"my later work")
    with pytest.raises(ValueError, match="Unshared local edits"):
        apply_update(source, target, manifest)
    assert (target / "module.py").read_bytes() == b"my later work"
    assert not (target / "reports").exists()


@pytest.mark.parametrize("path", [".env", "data/index/corpus.sqlite3", "../escape.py"])
def test_upgrade_manifest_cannot_target_runtime_or_parent_paths(tmp_path, path):
    source, target, manifest = fixture(tmp_path)
    manifest["files"][0]["path"] = path
    with pytest.raises(ValueError, match="Protected or invalid"):
        apply_update(source, target, manifest)


def test_partial_write_failure_restores_original_source(tmp_path, monkeypatch):
    from pathlib import Path

    source, target, manifest = fixture(tmp_path)
    original = Path.write_bytes
    fail = [True]

    def write(path, data):
        if path == target / "module.py" and data == b"new code" and fail[0]:
            fail[0] = False
            original(path, b"partial")
            raise OSError("simulated disk write failure")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", write)
    with pytest.raises(OSError, match="simulated"):
        apply_update(source, target, manifest)
    assert (target / "module.py").read_bytes() == b"old code"
