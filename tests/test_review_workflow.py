import csv
import hashlib

import pymupdf
import pytest

from rag.evaluation.review import ReviewSuite, export_review, summarize_reviews, verify_sources
from scripts.validate_workflow import run


def fixture(tmp_path, pdf_bytes):
    raw = pdf_bytes()
    source = tmp_path / "paper.pdf"
    source.write_bytes(raw)
    with pymupdf.open(stream=raw, filetype="pdf") as pdf:
        quote = pdf[0].get_text()
    suite = ReviewSuite.model_validate(
        {
            "description": "Synthetic test only",
            "sources": [
                {"id": "paper", "filename": source.name, "sha256": hashlib.sha256(raw).hexdigest()}
            ],
            "cases": [
                {
                    "id": "c1",
                    "source_id": "paper",
                    "question": "Hybrid retrieval BM25",
                    "language": "en",
                    "expected_points": ["Source states hybrid retrieval."],
                    "references": [{"page": 1, "text": quote}],
                }
            ],
        }
    )
    result = {
        "run_id": "run-1",
        "records": [
            {
                "id": "c1",
                "question": suite.cases[0].question,
                "language": "en",
                "document_id": suite.sources[0].sha256,
                "response": {
                    "answer": "=UNTRUSTED()",
                    "generation_mode": "extractive",
                    "warning": None,
                },
                "citation_provenance_valid": True,
                "language_matches": True,
            }
        ],
    }
    return suite, result


def rewrite(path, change):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    change(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_review_does_not_turn_citation_checks_into_correctness_grades(tmp_path, pdf_bytes):
    suite, result = fixture(tmp_path, pdf_bytes)
    path = tmp_path / "review.csv"
    export_review(suite, result, path)
    summary = summarize_reviews(suite, result, path)
    assert summary["reviewed"] == 0
    assert summary["by_reviewer_type"]["human"]["pass_rate"] is None
    assert summary["by_reviewer_type"]["ai_assisted"]["pass_rate"] is None
    assert "'=UNTRUSTED()" in path.read_text(encoding="utf-8-sig")
    with pytest.raises(FileExistsError):
        export_review(suite, result, path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "stale-run"),
        ("answer", "changed answer"),
        ("expected_points", "fabricated gold"),
        ("response_sha256", "wrong"),
    ],
)
def test_review_cannot_be_reused_for_another_run_or_changed_answer(
    tmp_path, pdf_bytes, field, value
):
    suite, result = fixture(tmp_path, pdf_bytes)
    path = tmp_path / "review.csv"
    export_review(suite, result, path)
    rewrite(path, lambda row: row.update({field: value}))
    with pytest.raises(ValueError, match="Immutable"):
        summarize_reviews(suite, result, path)


def test_reviews_distinguish_ai_assistance_and_require_failure_notes(tmp_path, pdf_bytes):
    suite, result = fixture(tmp_path, pdf_bytes)
    path = tmp_path / "review.csv"
    export_review(suite, result, path)
    rewrite(
        path,
        lambda row: row.update(reviewer="Assistant", reviewer_type="ai_assisted", verdict="fail"),
    )
    with pytest.raises(ValueError, match="explanation"):
        summarize_reviews(suite, result, path)
    rewrite(path, lambda row: row.update(notes="Missing requested source facts."))
    summary = summarize_reviews(suite, result, path)
    assert summary["by_reviewer_type"]["human"]["reviewed"] == 0
    assert summary["by_reviewer_type"]["ai_assisted"]["counts"]["fail"] == 1


def test_source_reference_checks_hash_and_native_pdf_page(tmp_path, pdf_bytes):
    suite, _ = fixture(tmp_path, pdf_bytes)
    assert len(verify_sources(suite, tmp_path)) == 1
    suite.cases[0].references[0].page = 2
    with pytest.raises(ValueError, match="quote/page"):
        verify_sources(suite, tmp_path)
    (tmp_path / "paper.pdf").write_bytes(pdf_bytes("Different document"))
    with pytest.raises(ValueError, match="Wrong PDF"):
        verify_sources(suite, tmp_path)


def test_isolated_workflow_never_overwrites_existing_output_or_private_settings(
    tmp_path, pdf_bytes, monkeypatch
):
    suite, _ = fixture(tmp_path, pdf_bytes)
    dataset = tmp_path / "cases.json"
    dataset.write_text(suite.model_dump_json())
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "private-index"))
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "hash")
    output = tmp_path / "run"
    result = run(dataset, tmp_path, output, learned=False)
    assert (
        result["use_llm"] is False
        and result["records"][0]["document_id"] == suite.sources[0].sha256
    )
    assert not (tmp_path / "private-index").exists()
    assert (output / "review.csv").exists()
    with pytest.raises(ValueError, match="already exists"):
        run(dataset, tmp_path, output)


def test_preflight_bad_reference_creates_no_run_directory(tmp_path, pdf_bytes):
    suite, _ = fixture(tmp_path, pdf_bytes)
    suite.cases[0].references[0].text = "invented reference"
    dataset = tmp_path / "cases.json"
    dataset.write_text(suite.model_dump_json())
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="quote/page"):
        run(dataset, tmp_path, output)
    assert not output.exists()


def test_chat_export_keeps_failure_turns_without_inventing_answers_or_grades():
    from rag.evaluation.review import chat_review_export

    messages = [
        {"role": "user", "text": "First question"},
        {
            "role": "assistant",
            "text": "answer",
            "response": {"answer": "answer", "citations": []},
            "request": {"language": "en"},
        },
        {"role": "user", "text": "Failed question"},
        {"role": "assistant", "text": "API 500", "error": "API 500", "request": {"language": "ar"}},
    ]
    exported = chat_review_export(messages)
    assert [t["question"] for t in exported["turns"]] == ["First question", "Failed question"]
    assert exported["turns"][1]["response"] is None
    assert exported["turns"][1]["error"] == "API 500"
    assert all(t["review"]["verdict"] == "not_reviewed" for t in exported["turns"])
    exported["turns"][0]["response"]["answer"] = "later review edit"
    assert messages[1]["response"]["answer"] == "answer"
