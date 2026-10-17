import json
from zipfile import ZipFile

from scripts.package_source import package


def test_source_package_omits_private_runtime_and_generated_files(tmp_path):
    root = tmp_path / "project"
    (root / "src/rag/__pycache__").mkdir(parents=True)
    (root / "src/rag/main.py").write_text("pass")
    (root / "src/rag/__pycache__/main.pyc").write_bytes(b"bytecode")
    (root / ".env").write_text("PRIVATE=not-for-archive")
    (root / ".env.example").write_text("RAG_DEVICE=cpu")
    (root / "data/index").mkdir(parents=True)
    (root / "data/index/corpus.sqlite3").write_bytes(b"private corpus")
    (root / "README.md").write_text("Setup instructions")
    output = tmp_path / "source.zip"
    package(root, output)
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "src/rag/main.py",
            ".env.example",
            "README.md",
            "SOURCE-MANIFEST.json",
        }
        manifest = json.loads(archive.read("SOURCE-MANIFEST.json"))
        assert len(manifest) == 3 and all(len(row["sha256"]) == 64 for row in manifest)
