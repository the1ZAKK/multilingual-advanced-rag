"""Create a portable source archive without environments, models, corpora or secrets."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

DIRECTORIES = {"src", "api", "app", "tests", "scripts", "docs", ".github", "data/evaluation"}
FILES = {
    ".gitattributes",
    "UPDATE-MANIFEST.json",
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "TASKS.md",
    "Dockerfile",
    "docker-compose.yml",
}
OMIT = {"__pycache__", ".venv", ".cache", ".git", ".pytest_cache", ".ruff_cache"}


def package(root: Path, output: Path) -> dict:
    root = root.resolve()
    selected = {root / name for name in FILES if (root / name).is_file()}
    for name in DIRECTORIES:
        if (root / name).is_dir():
            selected.update(p for p in (root / name).rglob("*") if p.is_file())
    manifest = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            relative = path.relative_to(root)
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root)
                or set(relative.parts) & OMIT
                or path.suffix.lower()
                in {".pyc", ".pyo", ".zip", ".sqlite3", ".db", ".bin", ".gguf", ".safetensors"}
                or (path.name.startswith(".env") and path.name != ".env.example")
            ):
                continue
            data = path.read_bytes()
            archive.writestr(relative.as_posix(), data)
            manifest.append(
                {
                    "path": relative.as_posix(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        archive.writestr("SOURCE-MANIFEST.json", json.dumps(manifest, indent=2))
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity verification failed")
    if output.stat().st_size >= 32 * 1024 * 1024:
        raise ValueError(
            "Source archive exceeds 32 MiB; review large docs/test assets before sharing"
        )
    return {"files": len(manifest), "bytes": output.stat().st_size, "archive": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("project-source.zip"))
    args = parser.parse_args()
    print(json.dumps(package(Path(__file__).resolve().parents[1], args.output), indent=2))


if __name__ == "__main__":
    main()
