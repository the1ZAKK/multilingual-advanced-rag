import httpx
import pytest

from rag.generation.ollama import GenerationUnavailable, validate_answer
from rag.models import ChatRequest, Chunk, Hit, SearchRequest


def semantic_response(data):
    return {
        "verdicts": [
            {
                "item_id": item["item_id"],
                "supported": True,
                "complete": True,
                "direct": True,
                "reason": "The mock premise supports this contract-test claim.",
            }
            for item in data["items"]
        ]
    }


def test_citations_and_fallback(service, pdf_bytes, monkeypatch):
    info = service.ingest(pdf_bytes(), "source.pdf")
    monkeypatch.setattr(
        service.generator,
        "generate",
        lambda *args: (_ for _ in ()).throw(GenerationUnavailable("offline")),
    )
    result = service.chat(ChatRequest(query="BM25 retrieval"))
    assert result.generation_mode == "extractive" and result.warning == "offline"
    for citation in result.citations:
        assert citation.document_id == info.document_id and citation.page == 1
        assert citation.quote in next(
            h.chunk.text for h in result.evidence if h.chunk.chunk_id == citation.chunk_id
        )
        assert f"[{citation.label}]" in result.answer


def test_empty_evidence_and_arabic(service):
    result = service.chat(ChatRequest(query="ماذا؟", language="ar"))
    assert result.generation_mode == "insufficient-evidence" and not result.citations
    assert "أدلة" in result.answer


def test_reject_fabricated_citation(service, pdf_bytes):
    service.ingest(pdf_bytes(), "source.pdf")
    evidence = service.search(SearchRequest(query="retrieval")).hits
    for payload in [
        {"answer": "Claim [S9]", "citations": [{"label": "S9", "quote": "invented"}]},
        {"answer": "Claim [S1]", "citations": [{"label": "S1", "quote": "invented"}]},
        {
            "answer": "Uncited claim",
            "citations": [{"label": "S1", "quote": evidence[0].chunk.text}],
        },
        {
            "answer": "Claim [S1] [made-up]",
            "citations": [{"label": "S1", "quote": evidence[0].chunk.text}],
        },
    ]:
        with pytest.raises(ValueError):
            validate_answer(payload, evidence)


def test_valid_grounded_response(service, pdf_bytes):
    service.ingest(pdf_bytes(), "source.pdf")
    evidence = service.search(SearchRequest(query="retrieval")).hits
    answer, citations = validate_answer(
        {
            "answer": "Hybrid retrieval [S1]",
            "citations": [{"label": "S1", "quote": evidence[0].chunk.text}],
        },
        evidence,
    )
    assert answer and citations[0].page == 1


def test_valid_grounded_response_with_text_citation_field(service, pdf_bytes):
    service.ingest(pdf_bytes(), "source.pdf")
    evidence = service.search(SearchRequest(query="retrieval")).hits
    quote = evidence[0].chunk.text
    answer, citations = validate_answer(
        {
            "answer": "Hybrid retrieval combines semantic and keyword search.",
            "citations": [{"label": "S1", "text": quote}],
        },
        evidence,
    )
    assert answer.endswith("[S1]") and citations[0].quote == quote


def test_three_lm_retries_invalid_quotes_without_losing_sources(service, monkeypatch, caplog):
    caplog.set_level("INFO", logger="rag.generation.ollama")
    texts = [
        "The first benchmark uses text-\nbooks and worksheets.",
        "An unrelated passage about evaluation metrics.",
    ]
    evidence = [
        Hit(
            chunk=Chunk(
                chunk_id=str(index),
                document_id="a" * 64,
                document_name="3LM.pdf",
                page=1,
                text=text,
                normalized=text,
                start=0,
                end=len(text),
            ),
            score=1.0,
        )
        for index, text in enumerate(texts)
    ]
    responses = [
        {
            "answer": "The benchmark uses textbooks [S1]",
            "citations": [
                {"label": "S1", "quote": "The first benchmark uses text-books and worksheets."}
            ],
        },
        {
            "answer": "The benchmark uses the source wording [S1]",
            "citations": [{"label": "S1", "quote": texts[0]}],
        },
    ]
    requests = []

    def handle(request):
        import json

        body = json.loads(request.content)
        data = json.loads(body["messages"][1]["content"])
        if data.get("task") == "semantic-grounding-audit":
            return httpx.Response(
                200, json={"message": {"content": json.dumps(semantic_response(data))}}
            )
        requests.append(data)
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps(responses[len(requests) - 1])}},
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        "rag.generation.ollama.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle)),
    )
    answer, citations = service.generator.generate(
        "What does the benchmark contain?", "en", evidence
    )

    assert len(requests) == 2
    assert requests[0]["evidence"] == requests[1]["evidence"]
    assert "validation_error" in requests[1]
    assert answer == "The benchmark uses the source wording [S1]"
    assert citations[0].label == "S1" and citations[0].quote == texts[0]
    assert "ollama_initial_validation_failed: ValueError: Quote is not present" in caplog.text
    assert "ollama_retry_validation_succeeded" in caplog.text


def test_three_lm_logs_initial_and_retry_validation_failures(service, monkeypatch, caplog):
    caplog.set_level("INFO", logger="rag.generation.ollama")
    evidence = [
        Hit(
            chunk=Chunk(
                chunk_id=str(index),
                document_id="a" * 64,
                document_name="3LM.pdf",
                page=1,
                text=text,
                normalized=text,
                start=0,
                end=len(text),
            ),
            score=1.0,
        )
        for index, text in enumerate(["first source text", "second source text"])
    ]
    responses = [
        {"answer": "Unsupported [S1]", "citations": [{"label": "S1", "quote": "wrong"}]},
        {"answer": "Unsupported [S1]", "citations": [{"label": "S1", "quote": "wrong"}]},
    ]
    call_count = 0

    def handle(request):
        nonlocal call_count
        response = responses[call_count]
        call_count += 1
        return httpx.Response(
            200, json={"message": {"content": __import__("json").dumps(response)}}
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        "rag.generation.ollama.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle)),
    )
    with pytest.raises(GenerationUnavailable):
        service.generator.generate("question", "en", evidence)

    assert "ollama_initial_validation_failed: ValueError: Quote is not present" in caplog.text
    assert "ollama_retry_validation_failed: ValueError: Quote is not present" in caplog.text


def test_ollama_http_contract(service, pdf_bytes, monkeypatch):
    service.ingest(pdf_bytes(), "source.pdf")
    evidence = service.search(SearchRequest(query="retrieval")).hits
    seen = []

    def handle(request):
        import json

        body = json.loads(request.content)
        seen.append(body)
        data = json.loads(body["messages"][1]["content"])
        if data.get("task") == "semantic-grounding-audit":
            return httpx.Response(
                200, json={"message": {"content": json.dumps(semantic_response(data))}}
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "answer": "Supported [S1]",
                            "citations": [{"label": "S1", "quote": evidence[0].chunk.text}],
                        }
                    )
                }
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        "rag.generation.ollama.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle)),
    )
    answer, citations = service.generator.generate("retrieval", "en", evidence)
    assert answer == "Supported [S1]" and citations
    assert seen[0]["format"]["required"] == ["statements"]
    assert seen[0]["stream"] is False
    assert "untrusted" in seen[0]["messages"][0]["content"]
    assert "character-for-character" in seen[0]["messages"][0]["content"]
    assert seen[1]["format"]["required"] == ["verdicts"]
