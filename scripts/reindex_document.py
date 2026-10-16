"""Explicit local migration, with a SQLite backup and hash-verified original PDF."""

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rag.config import Settings
from rag.service import RAGService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document_id")
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    settings = Settings()
    backup_dir = settings.data_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".sqlite3")
    with sqlite3.connect(settings.data_dir / "corpus.sqlite3") as original:
        with sqlite3.connect(backup) as target:
            original.backup(target)
    service = RAGService(settings)
    info = service.reindex_document(args.document_id, args.pdf.read_bytes() if args.pdf else None)
    print(info.model_dump_json())
    print(f"Backup: {backup.resolve()}")


if __name__ == "__main__":
    main()
