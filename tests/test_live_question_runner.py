import json

from scripts.live_questions import run_questions


def test_same_name_documents_require_explicit_identity(service, pdf_bytes, tmp_path):
    first = service.ingest(pdf_bytes("Hybrid retrieval combines vectors and BM25."), "same.pdf")
    second = service.ingest(pdf_bytes("Neutron measurements use a detector."), "same.pdf")
    cases = [
        {
            "id": "ambiguous",
            "question": "Hybrid retrieval BM25",
            "language": "en",
            "document_name": "same.pdf",
        }
    ]
    output = tmp_path / "ambiguous.json"
    records = run_questions(service, cases, output, use_llm=False)
    assert "ambiguous" in records[0]["error"]
    assert "response" not in records[0]
    pinned = [{**cases[0], "id": "pinned", "document_id": first.document_id}]
    path = tmp_path / "pinned.json"
    records = run_questions(service, pinned, path, use_llm=False)
    assert records[0]["document_id"] == first.document_id
    assert records[0]["response"]["evidence"]
    assert all(
        h["chunk"]["document_id"] == first.document_id for h in records[0]["response"]["evidence"]
    )
    assert second.document_id != first.document_id
    one, two = [json.loads(p.read_text()) for p in (output, path)]
    assert one["run_id"] != two["run_id"]
    assert one["use_llm"] is two["use_llm"] is False
