import subprocess
from pathlib import Path

import pytest

from scripts.apply_update import apply_update
from scripts.prepare_update import prepare


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def test_release_chain_updates_original_and_previous_without_accepting_private_edits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q")
    (source / "README.md").write_text("original")
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "baseline",
    )
    baseline = git(source, "rev-parse", "HEAD")
    (source / "README.md").write_text("release one")
    first = prepare(source, baseline)
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "README.md").write_text("original")
    apply_update(source, prior, first)
    (source / "README.md").write_text("release two")
    (source / "scripts").mkdir()
    (source / "scripts/new.py").write_text("print('new source')")
    (source / ".env").write_text("PRIVATE=sentinel")
    (source / "scripts/private.db").write_bytes(b"private database")
    second = prepare(source, baseline, first)
    assert {item["path"] for item in second["files"]} == {"README.md", "scripts/new.py"}
    assert apply_update(source, prior, second)["changed_files"] == 2
    original = tmp_path / "original"
    original.mkdir()
    (original / "README.md").write_text("original")
    assert apply_update(source, original, second)["changed_files"] == 2
    assert apply_update(source, original, second)["changed_files"] == 0
    (prior / "README.md").write_text("my private edits")
    with pytest.raises(ValueError, match="Unshared local edits"):
        apply_update(source, prior, second)


def test_manifest_rejects_deleted_tracked_sources(tmp_path):
    git(tmp_path, "init", "-q")
    readme = Path(tmp_path) / "README.md"
    readme.write_text("original")
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "baseline",
    )
    readme.unlink()
    with pytest.raises(ValueError, match="manual review"):
        prepare(tmp_path, "HEAD")


def test_manifest_rejects_removed_previous_release_file_even_if_never_tracked(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("original")
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "baseline",
    )
    (tmp_path / "scripts").mkdir()
    new = tmp_path / "scripts/new.py"
    new.write_text("pass")
    previous = prepare(tmp_path, "HEAD")
    new.unlink()
    with pytest.raises(ValueError, match="manual review"):
        prepare(tmp_path, "HEAD", previous)
