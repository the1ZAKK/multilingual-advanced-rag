"""Prepare a source update manifest from a Git baseline and optional previous release.

Developer release tool: does not modify the baseline, target installations or Git.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.package_source import DIRECTORIES, FILES, OMIT


def prepare(root: Path, baseline: str, previous: dict | None = None) -> dict:
    root = root.resolve()
    baseline = subprocess.check_output(
        ["git", "rev-parse", "--verify", baseline + "^{commit}"], cwd=root, text=True
    ).strip()
    previous_files = {item["path"]: item for item in (previous or {}).get("files", [])}
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
        )
        .decode()
        .split("\0")
    )
    files = []
    for name in sorted(set(names) | set(previous_files)):
        if not name or name in {"UPDATE-MANIFEST.json", "SOURCE-MANIFEST.json"}:
            continue
        relative = Path(name)
        if name not in FILES and not any(
            relative.is_relative_to(directory) for directory in DIRECTORIES
        ):
            continue
        path = root / relative
        if (
            set(relative.parts) & OMIT
            or path.suffix.lower()
            in {".pyc", ".pyo", ".zip", ".sqlite3", ".db", ".bin", ".gguf", ".safetensors"}
            or (path.name.startswith(".env") and path.name != ".env.example")
        ):
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing/deleted/symlinked source requires manual review: {name}")
        old = subprocess.run(["git", "show", baseline + ":" + name], cwd=root, capture_output=True)
        old_sha = hashlib.sha256(old.stdout).hexdigest() if old.returncode == 0 else None
        new_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if old_sha == new_sha:
            if name in previous_files and previous_files[name]["sha256"] != new_sha:
                raise ValueError(
                    f"Reverting a shipped file requires manual migration review: {name}"
                )
            continue
        prior = previous_files.get(name, {})
        accepted = set(prior.get("accepted_sha256", []))
        if prior.get("sha256"):
            accepted.add(prior["sha256"])
        accepted.discard(new_sha)
        accepted.discard(old_sha)
        files.append(
            {
                "path": name,
                "sha256": new_sha,
                "baseline_sha256": old_sha,
                "accepted_sha256": sorted(accepted),
            }
        )
    return {"baseline": baseline, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--previous-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("UPDATE-MANIFEST.json"))
    args = parser.parse_args()
    previous = (
        json.loads(args.previous_manifest.read_text(encoding="utf-8"))
        if args.previous_manifest
        else None
    )
    result = prepare(Path(__file__).resolve().parents[1], args.baseline, previous)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Prepared {len(result['files'])} source updates")


if __name__ == "__main__":
    main()
