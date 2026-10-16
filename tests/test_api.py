from fastapi.testclient import TestClient

from api.main import create_app


def test_end_to_end(service, settings, pdf_bytes):
    with TestClient(create_app(settings, service)) as client:
        assert client.get("/health").json()["status"] == "ok"
        uploaded = client.post(
            "/documents/upload", files=[("files", ("paper.pdf", pdf_bytes(), "application/pdf"))]
        )
        assert uploaded.status_code == 200
        doc = uploaded.json()["documents"][0]
        assert len(client.get("/documents").json()) == 1
        found = client.post("/search", json={"query": "BM25", "mode": "hybrid"}).json()
        assert found["hits"][0]["chunk"]["document_id"] == doc["document_id"]
        chat = client.post("/chat", json={"query": "BM25", "use_llm": False}).json()
        assert chat["citations"][0]["page"] == 1
        evaluated = client.post(
            "/evaluate",
            json={
                "cases": [
                    {
                        "id": "q1",
                        "question": "BM25",
                        "relevant": {found["hits"][0]["chunk"]["chunk_id"]: 1},
                    }
                ],
                "modes": ["dense", "bm25", "hybrid", "hybrid-rerank"],
            },
        )
        assert evaluated.status_code == 200 and len(evaluated.json()["summary"]) == 4
        assert client.delete("/documents/" + doc["document_id"]).status_code == 200
        assert client.delete("/documents/" + doc["document_id"]).status_code == 404
        assert client.get("/documents").json() == []
        assert (
            client.post("/chat", json={"query": "BM25"}).json()["generation_mode"]
            == "insufficient-evidence"
        )


def test_validation_and_partial_upload(service, settings, pdf_bytes):
    with TestClient(create_app(settings, service)) as client:
        assert client.post("/search", json={"query": "", "top_k": 0}).status_code == 422
        assert client.post("/search", json={"query": "   "}).status_code == 422
        bad = client.post(
            "/documents/upload", files={"files": ("bad.pdf", b"not pdf", "application/pdf")}
        )
        assert bad.status_code == 422 and bad.json()["errors"]
        mixed = client.post(
            "/documents/upload",
            files=[
                ("files", ("good.pdf", pdf_bytes(), "application/pdf")),
                ("files", ("bad.pdf", b"invalid", "application/pdf")),
            ],
        )
        assert (
            mixed.status_code == 200
            and len(mixed.json()["documents"]) == 1
            and len(mixed.json()["errors"]) == 1
        )
        assert client.delete("/documents/../../etc").status_code == 404
        assert client.post("/evaluate", json={"cases": []}).status_code == 422
