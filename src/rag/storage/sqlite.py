import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from rag.models import Chunk, DocumentInfo


class Store:
    """SQLite is authoritative; FAISS/BM25 are rebuildable derived indexes."""

    def __init__(self, directory: Path, embedding_key: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "corpus.sqlite3"
        self.embedding_key = embedding_key
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, info TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, document_id TEXT NOT NULL
                    REFERENCES documents(id) ON DELETE CASCADE, data TEXT NOT NULL, vector BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS source_pdfs (document_id TEXT PRIMARY KEY
                    REFERENCES documents(id) ON DELETE CASCADE, pdf BLOB NOT NULL);
            """)
            row = db.execute("SELECT value FROM meta WHERE key='embedding'").fetchone()
            if row and row[0] != embedding_key:
                raise RuntimeError(
                    "Embedding model changed: use a new RAG_DATA_DIR or explicitly reingest the corpus"
                )
            db.execute("INSERT OR IGNORE INTO meta VALUES ('embedding', ?)", (embedding_key,))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def documents(self) -> list[DocumentInfo]:
        with self.connect() as db:
            return [
                DocumentInfo.model_validate_json(r[0])
                for r in db.execute("SELECT info FROM documents ORDER BY id")
            ]

    def corpus(self) -> tuple[list[Chunk], np.ndarray]:
        with self.connect() as db:
            rows = db.execute("SELECT data, vector FROM chunks ORDER BY id").fetchall()
        chunks = [Chunk.model_validate_json(r[0]) for r in rows]
        vectors = (
            np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 384), dtype=np.float32)
        )
        return chunks, vectors

    def add(
        self,
        info: DocumentInfo,
        chunks: list[Chunk],
        vectors: np.ndarray,
        source: bytes | None = None,
        *,
        replace: bool = False,
    ):
        with self.connect() as db:
            if replace:
                if not db.execute(
                    "SELECT 1 FROM documents WHERE id=?", (info.document_id,)
                ).fetchone():
                    raise ValueError("Cannot reindex an unknown document")
                db.execute(
                    "UPDATE documents SET info=? WHERE id=?",
                    (info.model_dump_json(), info.document_id),
                )
                db.execute("DELETE FROM chunks WHERE document_id=?", (info.document_id,))
            else:
                db.execute(
                    "INSERT INTO documents VALUES (?, ?)",
                    (info.document_id, info.model_dump_json()),
                )
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                [
                    (c.chunk_id, c.document_id, c.model_dump_json(), v.astype(np.float32).tobytes())
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
            if source is not None:
                db.execute(
                    "INSERT OR IGNORE INTO source_pdfs VALUES (?, ?)", (info.document_id, source)
                )

    def source(self, document_id: str) -> bytes | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT pdf FROM source_pdfs WHERE document_id=?", (document_id,)
            ).fetchone()
        return row[0] if row else None

    def delete(self, document_id: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount > 0
