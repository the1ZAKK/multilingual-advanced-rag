import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from rag.models import ChatRequest, SearchRequest


def test_document_filter_is_applied_before_top_k(service, pdf_bytes):
    target = service.ingest(pdf_bytes("Retrieval is relevant to research."), "Target.pdf")
    for i in range(8):
        service.ingest(
            pdf_bytes(f"Retrieval retrieval retrieval research research {i}."), f"Other{i}.pdf"
        )
    for mode in ["dense", "bm25", "hybrid", "hybrid-rerank"]:
        hits = service.search(
            SearchRequest(
                query="retrieval research", top_k=1, mode=mode, document_ids=[target.document_id]
            )
        ).hits
        assert len(hits) == 1
        assert hits[0].chunk.document_id == target.document_id
    answer = service.chat(
        ChatRequest(query="retrieval", document_ids=[target.document_id], use_llm=False)
    )
    assert {h.chunk.document_id for h in answer.evidence} == {target.document_id}


def test_invalid_scope_is_not_treated_as_all_documents(service, settings):
    with pytest.raises(ValueError, match="Unknown document"):
        service.search(SearchRequest(query="research", document_ids=["missing"]))
    with TestClient(create_app(settings, service)) as client:
        assert (
            client.post("/search", json={"query": "research", "document_ids": []}).status_code
            == 422
        )
        assert (
            client.post(
                "/chat", json={"query": "research", "document_ids": ["missing"]}
            ).status_code
            == 422
        )


def test_overview_selects_scoped_abstract_before_appendix_question_examples(service, pdf_bytes):
    from rag.models import Chunk, Hit

    abstract = Chunk(
        chunk_id="d:p1:0-80",
        document_id="d",
        document_name="Novel.pdf",
        page=1,
        text="This benchmark contains school exam questions.",
        normalized="this benchmark contains school exam questions",
        start=0,
        end=80,
        section="Abstract",
    )
    appendix = abstract.model_copy(
        update={
            "chunk_id": "d:p9:0-80",
            "page": 9,
            "section": "Appendix",
            "text": "Examples of questions: what is the correct answer?",
        }
    )
    service.index.chunks = [abstract, appendix]
    evidence = service.select_evidence(
        [Hit(chunk=appendix, score=1, bm25_score=1)],
        "What is the benchmark and where are its questions sourced?",
        {"d"},
    )
    assert evidence[0].chunk.chunk_id == abstract.chunk_id
    assert evidence[0].score == 0  # Source context is not misreported as a retrieval score.


def test_explicit_abstract_request_anchors_unheaded_first_page_only_in_scope(service):
    from rag.models import Chunk, Hit

    front = Chunk(
        chunk_id="d:p1:0-70",
        document_id="d",
        document_name="Novel.pdf",
        page=1,
        text="The result applies to all NPT states.",
        normalized="result all npt",
        start=0,
        end=70,
        section=None,
    )
    wrong = front.model_copy(update={"chunk_id": "other:p1:0-70", "document_id": "other"})
    body = front.model_copy(update={"chunk_id": "d:p6:0-70", "page": 6, "section": "Proof"})
    service.index.chunks = [wrong, front, body]
    hits = [Hit(chunk=body, score=1)]
    for question in ("According to the abstract, which states?", "وفقًا لملخص الورقة، أي حالات؟"):
        result = service.select_evidence(hits, question, {"d"})
        assert result[0].chunk == front
        assert result[0].chunk.section is None  # Never fabricate a structural label.
        assert all(h.chunk.document_id == "d" for h in result)
    assert service.select_evidence(hits, "Explain this proof", {"d"})[0].chunk == body
