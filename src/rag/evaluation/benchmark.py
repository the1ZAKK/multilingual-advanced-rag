import csv
import json
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field, model_validator

from rag.evaluation.metrics import evidence_token_support, retrieval_metrics, token_f1
from rag.models import ChatRequest, SearchRequest


class EvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)
    relevant: dict[str, float] = Field(min_length=1)
    reference_answer: str | None = None
    language: str = "en"

    @model_validator(mode="after")
    def validate_grades(self):
        retrieval_metrics([], self.relevant)
        if self.language not in {"ar", "en", "auto"}:
            raise ValueError("language must be ar, en, or auto")
        return self


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1, max_length=100)
    modes: list[str] = Field(
        default_factory=lambda: ["dense", "bm25", "hybrid", "hybrid-rerank"],
        min_length=1,
        max_length=4,
    )
    generate_answers: bool = False
    use_llm: bool = False

    @model_validator(mode="after")
    def validate_modes(self):
        if len(set(self.modes)) != len(self.modes) or not set(self.modes) <= {
            "dense",
            "bm25",
            "hybrid",
            "hybrid-rerank",
        }:
            raise ValueError("Provide distinct supported retrieval modes")
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Case IDs must be unique")
        return self


def evaluate(service, request: EvaluationRequest) -> dict:
    # Freeze corpus membership across the benchmark. No missing IDs masquerade as poor retrieval.
    with service.lock:
        known = {c.chunk_id for c in service.index.chunks}
        missing = {cid for c in request.cases for cid in c.relevant if cid not in known}
        if missing:
            raise ValueError(
                f"Evaluation contains {len(missing)} unknown passage identifiers; reannotate for this corpus"
            )
        rows = []
        for mode in request.modes:
            for case in request.cases:
                search = service.search(SearchRequest(query=case.question, mode=mode, top_k=10))
                row = {
                    "case_id": case.id,
                    "language": case.language,
                    "mode": mode,
                    **retrieval_metrics([h.chunk.chunk_id for h in search.hits], case.relevant),
                    "query_latency_ms": search.latency_ms,
                    "rerank_latency_ms": search.rerank_latency_ms,
                }
                if request.generate_answers:
                    answer = service.chat(
                        ChatRequest(
                            query=case.question,
                            mode=mode,
                            top_k=5,
                            language=case.language,
                            use_llm=request.use_llm,
                        )
                    )
                    row.update(
                        response_latency_ms=answer.latency_ms,
                        generation_mode=answer.generation_mode,
                        evidence_token_support=evidence_token_support(
                            answer.answer, [h.chunk.text for h in answer.evidence]
                        ),
                    )
                    if case.reference_answer:
                        row["answer_token_f1"] = token_f1(answer.answer, case.reference_answer)
                rows.append(row)
        summaries = []
        for mode in request.modes:
            subset = [r for r in rows if r["mode"] == mode]
            numeric = [k for k, v in subset[0].items() if isinstance(v, (int, float))]
            summaries.append(
                {"mode": mode, **{k: mean(r[k] for r in subset if k in r) for k in numeric}}
            )
        return {
            "rows": rows,
            "summary": summaries,
            "embedding_backend": service.settings.embedding_backend,
            "embedding_model": service.embedder.key,
            "reranker_backend": service.settings.reranker_backend,
            "reranker_model": service.settings.reranker_model,
            "corpus_chunks": len(known),
            "note": "Scores describe only the supplied annotations. Lexical answer proxies are not semantic faithfulness or correctness.",
        }


def export_results(results: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = list(dict.fromkeys(k for row in results["rows"] for k in row))
    with (directory / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results["rows"])
