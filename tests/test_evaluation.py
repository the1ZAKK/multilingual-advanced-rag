import pytest

from rag.evaluation.benchmark import EvaluationCase, EvaluationRequest, evaluate, export_results
from rag.evaluation.metrics import evidence_token_support, retrieval_metrics, token_f1


def test_metrics():
    scores = retrieval_metrics(["x", "a", "b"], {"a": 2, "b": 1})
    assert scores["recall@1"] == 0 and scores["recall@5"] == 1
    assert scores["precision@5"] == 0.4 and scores["mrr"] == 0.5
    assert 0 < scores["ndcg@5"] < 1
    assert retrieval_metrics(["a", "a"], {"a": 1})["recall@5"] == 1
    assert token_f1("retrieval search", "retrieval") == pytest.approx(2 / 3)
    assert evidence_token_support("search retrieval", ["retrieval"]) == 0.5


def test_perfect_ndcg():
    assert retrieval_metrics(["a", "b"], {"a": 2, "b": 1})["ndcg@5"] == 1


def test_benchmark_exports(service, pdf_bytes, tmp_path):
    doc = service.ingest(pdf_bytes(), "a.pdf")
    cid = next(c.chunk_id for c in service.index.chunks if c.document_id == doc.document_id)
    req = EvaluationRequest(
        cases=[
            EvaluationCase(
                id="q1", question="BM25 retrieval", relevant={cid: 1}, reference_answer="BM25"
            )
        ],
        generate_answers=True,
    )
    results = evaluate(service, req)
    assert len(results["rows"]) == 4 and results["rows"][0]["recall@1"] == 1
    export_results(results, tmp_path / "results")
    assert (tmp_path / "results/benchmark.csv").read_text().count("q1") == 4


def test_unknown_annotation(service):
    req = EvaluationRequest(cases=[EvaluationCase(id="q1", question="q", relevant={"unknown": 1})])
    with pytest.raises(ValueError, match="unknown"):
        evaluate(service, req)


@pytest.mark.parametrize("grade", [0, -1, float("nan"), float("inf")])
def test_invalid_grades(grade):
    with pytest.raises(ValueError):
        EvaluationCase(id="a", question="q", relevant={"c": grade})
