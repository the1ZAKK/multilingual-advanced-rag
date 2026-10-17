import csv
import json
from types import SimpleNamespace

import pytest

from rag.evaluation.benchmark import (
    EvaluationCase,
    EvaluationRequest,
    citation_provenance,
    evaluate,
    export_results,
)
from rag.models import ChatRequest


def test_evaluation_exports_reviewable_answers_and_explicit_scored_denominators(
    service, pdf_bytes, tmp_path
):
    doc = service.ingest(pdf_bytes(), "Research.pdf")
    cid = service.index.chunks[0].chunk_id
    results = evaluate(
        service,
        EvaluationRequest(
            cases=[
                EvaluationCase(
                    id="review",
                    question="BM25 retrieval",
                    relevant={cid: 1},
                    document_ids=[doc.document_id],
                )
            ],
            modes=["hybrid"],
            generate_answers=True,
            include_answers=True,
        ),
    )
    row = results["rows"][0]
    assert row["answer"] and row["citations"] and row["citation_provenance_valid"]
    assert results["summary"][0]["citation_provenance_valid_count"] == 1
    assert results["summary"][0]["query_latency_p95_ms"] >= 0
    assert results["configuration"]["rrf_k"] == service.settings.rrf_k
    export_results(results, tmp_path)
    with (tmp_path / "benchmark.csv").open(encoding="utf-8", newline="") as f:
        record = next(csv.DictReader(f))
    assert json.loads(record["citations"])[0]["document_id"] == doc.document_id


def test_refusal_has_no_vacuous_citation_perfect_score():
    assert citation_provenance(SimpleNamespace(citations=[])) is None


def test_quote_presence_is_not_enough_when_document_identity_is_wrong(service, pdf_bytes):
    service.ingest(pdf_bytes(), "Research.pdf")
    result = service.chat(ChatRequest(query="BM25", use_llm=False))
    assert citation_provenance(result)
    corrupted = result.model_copy(deep=True)
    corrupted.citations[0].document_id = "different-document"
    assert citation_provenance(corrupted) is False


def test_gold_passages_outside_case_scope_are_rejected(service, pdf_bytes):
    first = service.ingest(pdf_bytes(), "First.pdf")
    second = service.ingest(pdf_bytes("Other document about arithmetic."), "Second.pdf")
    cid = next(c.chunk_id for c in service.index.chunks if c.document_id == first.document_id)
    request = EvaluationRequest(
        cases=[
            EvaluationCase(
                id="bad-scope",
                question="BM25",
                relevant={cid: 1},
                document_ids=[second.document_id],
            )
        ],
        modes=["bm25"],
    )
    with pytest.raises(ValueError, match="outside"):
        evaluate(service, request)
