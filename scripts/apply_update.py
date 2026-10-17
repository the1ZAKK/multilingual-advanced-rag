"""Apply the reviewed source update while preserving runtime data and later edits.

Run this from a separately extracted update folder, with --target pointing at
an existing installation. No packages, models, services or database are changed.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

PROTECTED = {".env", ".venv", ".cache", ".git", "reports"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def apply_update(source: Path, target: Path, manifest: dict, *, check_only=False):
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError(
            "Extract the update into a separate folder, not over the existing installation"
        )
    if not target.is_dir():
        raise ValueError("Target project directory does not exist")
    pending = []
    for item in manifest["files"]:
        relative = Path(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] in PROTECTED
            or (relative.parts[0] == "data" and relative.parts[:2] != ("data", "evaluation"))
        ):
            raise ValueError(f"Protected or invalid update path: {relative}")
        incoming = (source / relative).resolve()
        destination = (target / relative).resolve()
        if not incoming.is_relative_to(source) or not destination.is_relative_to(target):
            raise ValueError("Update path escapes the project directory")
        data = incoming.read_bytes()
        if digest(data) != item["sha256"]:
            raise ValueError(f"Incoming source integrity mismatch: {relative}")
        previous = destination.read_bytes() if destination.exists() else None
        if previous == data:
            continue
        if previous is not None and digest(previous) != item.get("baseline_sha256"):
            raise ValueError(
                f"Unshared local edits detected in {relative}; nothing was changed. Review/merge this file first."
            )
        if previous is None and item.get("baseline_sha256") is not None:
            raise ValueError(f"Expected baseline file is missing: {relative}; nothing was changed")
        pending.append((relative, destination, data, previous))
    if check_only or not pending:
        return {"pending_files": len(pending), "changed_files": 0, "backup": None}
    backup_dir = target / "reports" / "source-updates"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    backup = backup_dir / (stamp + ".zip")
    with ZipFile(backup, "w", compression=ZIP_DEFLATED) as archive:
        for relative, _, _, previous in pending:
            if previous is not None:
                archive.writestr(relative.as_posix(), previous)
        archive.writestr(
            "RESTORE-MANIFEST.json",
            json.dumps(
                [{"path": p.as_posix(), "existed": old is not None} for p, _, _, old in pending],
                indent=2,
            ),
        )
    changed = []
    try:
        for relative, destination, data, previous in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            changed.append((destination, previous))
            destination.write_bytes(data)
    except OSError:
        for destination, previous in reversed(changed):
            if previous is None:
                destination.unlink(missing_ok=True)
            else:
                destination.write_bytes(previous)
        raise
    return {"pending_files": len(pending), "changed_files": len(changed), "backup": str(backup)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "UPDATE-MANIFEST.json").read_text(encoding="utf-8"))
    try:
        result = apply_update(root, args.target, manifest, check_only=args.check)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Update refused: {exc}\n")
    print(json.dumps(result, indent=2))
    print("Existing .env, databases, model caches and virtual environment were not modified.")


if __name__ == "__main__":
    main()
