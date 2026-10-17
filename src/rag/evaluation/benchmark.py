import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, model_validator

from rag.evaluation.metrics import evidence_token_support, retrieval_metrics, token_f1
from rag.models import ChatRequest, SearchRequest
from rag.query import resolve_language, validate_output_language


class EvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)
    relevant: dict[str, float] = Field(min_length=1)
    document_ids: list[str] | None = Field(default=None, min_length=1, max_length=20)
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
    include_answers: bool = False

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
        by_chunk = {c.chunk_id: c.document_id for c in service.index.chunks}
        for case in request.cases:
            if case.document_ids and any(
                by_chunk[cid] not in case.document_ids for cid in case.relevant
            ):
                raise ValueError(
                    "Relevant passage annotations fall outside the case's document scope"
                )
        rows = []
        for mode in request.modes:
            for case in request.cases:
                search = service.search(
                    SearchRequest(
                        query=case.question, mode=mode, top_k=10, document_ids=case.document_ids
                    )
                )
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
                            document_ids=case.document_ids,
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
                    try:
                        validate_output_language(
                            answer.answer, resolve_language(case.question, case.language)
                        )
                        row["language_matches"] = True
                    except ValueError:
                        row["language_matches"] = False
                    row["citation_provenance_valid"] = citation_provenance(answer)
                    row["warning"] = answer.warning
                    if request.include_answers:
                        row["answer"] = answer.answer
                        row["citations"] = [c.model_dump() for c in answer.citations]
                        row["question"] = case.question
                    if case.reference_answer:
                        row["answer_token_f1"] = token_f1(answer.answer, case.reference_answer)
                rows.append(row)
        summaries = []
        for mode in request.modes:
            subset = [r for r in rows if r["mode"] == mode]
            numeric = [k for k, v in subset[0].items() if isinstance(v, (int, float))]
            numeric = sorted(
                {
                    k
                    for r in subset
                    for k, v in r.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
            )
            summary = {"mode": mode, **{k: mean(r[k] for r in subset if k in r) for k in numeric}}
            summary["query_latency_p50_ms"] = float(
                np.percentile([r["query_latency_ms"] for r in subset], 50)
            )
            summary["query_latency_p95_ms"] = float(
                np.percentile([r["query_latency_ms"] for r in subset], 95)
            )
            for key in ["language_matches", "citation_provenance_valid"]:
                checked = [r[key] for r in subset if r.get(key) is not None]
                if checked:
                    summary[key + "_rate"] = mean(checked)
                    summary[key + "_count"] = len(checked)
            summary["cases"] = len(subset)
            summaries.append(summary)
        return {
            "run_id": str(uuid4()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {"python": platform.python_version(), "os": platform.system()},
            "configuration": {
                "chunk_size": service.settings.chunk_size,
                "chunk_overlap": service.settings.chunk_overlap,
                "retrieval_depth": service.settings.retrieval_depth,
                "rrf_k": service.settings.rrf_k,
                "ollama_model": service.settings.ollama_model,
                "translation_model": service.settings.translation_model
                or service.settings.ollama_model,
                "max_context_chars": service.settings.max_context_chars,
                "llm_total_timeout": service.settings.llm_total_timeout,
                "max_answer_statements": service.settings.max_answer_statements,
            },
            "documents": [d.model_dump() for d in service.documents()],
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
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    columns = list(dict.fromkeys(k for row in results["rows"] for k in row))
    with (directory / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                if isinstance(v, (dict, list))
                else v
                for k, v in row.items()
            }
            for row in results["rows"]
        )


def citation_provenance(answer):
    """Check identity and exact source substrings; this is NOT an entailment score.

    No citations is unassessed (None), never a vacuous perfect score for refusal.
    """
    if not answer.citations:
        return None
    sources = {h.chunk.chunk_id: h for h in answer.evidence}
    for citation in answer.citations:
        hit = sources.get(citation.chunk_id)
        if (
            hit is None
            or (hit.chunk.document_id, hit.chunk.document_name, hit.chunk.page)
            != (citation.document_id, citation.document_name, citation.page)
            or not citation.quote
            or citation.quote not in hit.chunk.text
        ):
            return False
        if hit.native_page_text is not None and citation.quote not in hit.native_page_text:
            return False
    return True
