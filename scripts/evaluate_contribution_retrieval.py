"""Opt-in, PDF-grounded two-question retrieval diagnosis, not a general benchmark."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import httpx
import pymupdf
from diagnose_3lm import QUESTIONS

from rag.config import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    document_id = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    with sqlite3.connect(Settings().data_dir / "corpus.sqlite3") as db:
        chunks = [
            json.loads(row[0])
            for row in db.execute("SELECT data FROM chunks WHERE document_id=?", (document_id,))
        ]
    gold = [c for c in chunks if c.get("section") == "Contributions"]
    with pymupdf.open(args.pdf) as pdf:
        native = [page.get_text("text", sort=False) for page in pdf]
    # Check the distinctive authored statements against the actual PDF, independently
    # of the retrieval/generation pipeline and its mode flag.
    for ordinal in ("First", "Second", "Third"):
        assert any(ordinal + ", we" in c["text"] for c in gold)
        assert any(ordinal + ", we" in page for page in native)
    gold_ids = {c["chunk_id"] for c in gold}
    results = []
    with httpx.Client(timeout=180, trust_env=False) as client:
        for query in QUESTIONS:
            for mode in ("dense", "bm25", "hybrid", "hybrid-rerank"):
                request = {"query": query, "mode": mode, "top_k": 30}
                search = client.post("http://localhost:8000/search", json=request)
                search.raise_for_status()
                ranked = search.json()
                chat = client.post(
                    "http://localhost:8000/chat",
                    json={"query": query, "mode": mode, "use_llm": False},
                )
                chat.raise_for_status()
                evidence = chat.json()["evidence"]
                row = {
                    "question": query,
                    "mode": mode,
                    "gold_ranks": {
                        cid: next(
                            (
                                i
                                for i, h in enumerate(ranked["hits"], 1)
                                if h["chunk"]["chunk_id"] == cid
                            ),
                            None,
                        )
                        for cid in sorted(gold_ids)
                    },
                    "recall_at_5": len(
                        gold_ids & {h["chunk"]["chunk_id"] for h in ranked["hits"][:5]}
                    )
                    / len(gold_ids),
                    "selected_evidence_recall": len(
                        gold_ids & {h["chunk"]["chunk_id"] for h in evidence}
                    )
                    / len(gold_ids),
                    "query_latency_ms": ranked["latency_ms"],
                    "rerank_latency_ms": ranked["rerank_latency_ms"],
                    "evidence_ids": [h["chunk"]["chunk_id"] for h in evidence],
                    "ranking": ranked,
                }
                results.append(row)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "mode",
                                "gold_ranks",
                                "recall_at_5",
                                "selected_evidence_recall",
                                "rerank_latency_ms",
                            )
                        }
                    ),
                    flush=True,
                )
    Path("reports/3lm-diagnosis/retrieval-evaluation.json").write_text(
        json.dumps(
            {
                "pdf_sha256": document_id,
                "gold_chunks": gold,
                "results": results,
                "note": "Two supplied questions only; warmed local measurements, not broad multilingual quality claims.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
