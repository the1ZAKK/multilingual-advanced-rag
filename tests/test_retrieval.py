import numpy as np
import pytest

from rag.models import SearchRequest
from rag.retrieval.index import rrf
from rag.service import RAGService


def test_faiss_bm25_hybrid(service, pdf_bytes):
    doc = service.ingest(
        pdf_bytes("Hybrid retrieval combines semantic vectors and BM25 keywords."), "a.pdf"
    )
    service.ingest(pdf_bytes("Neutrinos are particles in a physics experiment."), "b.pdf")
    for mode in ["dense", "bm25", "hybrid", "hybrid-rerank"]:
        result = service.search(SearchRequest(query="BM25 retrieval", mode=mode))
        assert result.hits[0].chunk.document_id == doc.document_id
        assert result.latency_ms >= 0
        if mode == "hybrid-rerank":
            assert result.hits[0].rerank_score is not None


def test_rrf_scores_and_dedup(service, pdf_bytes):
    service.ingest(pdf_bytes(), "a.pdf")
    a = service.search(SearchRequest(query="retrieval", mode="dense")).hits[0]
    b = service.search(SearchRequest(query="retrieval", mode="bm25")).hits[0]
    fused = rrf([[a, a], [b]], 60)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(2 / 61)
    assert fused[0].dense_score is not None and fused[0].bm25_score is not None


def test_arabic_lexical(service, pdf_bytes):
    service.ingest(pdf_bytes("الاسترجاع الهجين يجمع البحث الدلالي والكلمات", arabic=True), "ar.pdf")
    hits = service.search(SearchRequest(query="الهَجِين", mode="bm25")).hits
    assert hits and hits[0].chunk.document_name == "ar.pdf"


def test_duplicate_restart_delete(service, settings, pdf_bytes):
    data = pdf_bytes()
    info = service.ingest(data, "a.pdf")
    assert service.ingest(data, "other.pdf").duplicate
    restored = RAGService(settings)
    assert len(restored.documents()) == 1
    assert (
        restored.search(SearchRequest(query="BM25")).hits[0].chunk.document_id == info.document_id
    )
    assert restored.delete(info.document_id)
    assert not restored.delete(info.document_id)
    empty = RAGService(settings)
    assert not empty.documents() and not empty.search(SearchRequest(query="BM25")).hits


def test_model_mismatch(service, settings, pdf_bytes):
    service.ingest(pdf_bytes(), "a.pdf")

    class Other:
        key = "other"

    with pytest.raises(RuntimeError, match="changed"):
        RAGService(settings, Other())


def test_cache_corruption_rebuild(service, settings, pdf_bytes):
    service.ingest(pdf_bytes(), "a.pdf")
    (settings.data_dir / "vectors.faiss").write_bytes(b"broken")
    rebuilt = RAGService(settings)
    assert rebuilt.search(SearchRequest(query="BM25")).hits


def test_failed_embedding_preserves_corpus(service, pdf_bytes, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr(service.embedder, "encode", fail)
    with pytest.raises(RuntimeError):
        service.ingest(pdf_bytes(), "a.pdf")
    assert service.documents() == [] and service.index.chunks == []


def test_whitespace_query(service):
    with pytest.raises(ValueError):
        service.search(SearchRequest(query="   "))


def test_cross_encoder_adapter(service, pdf_bytes, monkeypatch):
    service.ingest(pdf_bytes(), "a.pdf")

    class MockModel:
        def predict(self, pairs, batch_size):
            assert pairs[0][0] == "retrieval"
            return np.arange(len(pairs), dtype=float)

    service.reranker.settings = service.settings.model_copy(
        update={"reranker_backend": "cross-encoder"}
    )
    service.reranker.model = MockModel()
    result = service.search(SearchRequest(query="retrieval", mode="hybrid-rerank"))
    assert all(h.rerank_score is not None for h in result.hits)


def test_bm25_casefold_config(settings, pdf_bytes):
    svc = RAGService(settings.model_copy(update={"bm25_casefold": False}))
    svc.ingest(pdf_bytes("FAISS retrieves VECTORS."), "case.pdf")
    assert svc.search(SearchRequest(query="FAISS", mode="bm25")).hits
    assert not svc.search(SearchRequest(query="faiss", mode="bm25")).hits
