"""Run annotated local paper questions and preserve outputs for independent review.

No model answer is a correctness verdict. This tool records methods, language
following and exact citation provenance separately; humans inspect the claims.
It reads an existing corpus and never uploads, deletes or reindexes documents.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rag.config import Settings
from rag.evaluation.benchmark import citation_provenance
from rag.models import ChatRequest
from rag.query import resolve_language, validate_output_language
from rag.service import RAGService


def run_questions(service, cases, output: Path, *, use_llm=True):
    documents = service.documents()
    run_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        record = {
            "id": case["id"],
            "question": case["question"],
            "language": case.get("language", "auto"),
            "document_name": case["document_name"],
            "document_id": case.get("document_id"),
        }
        try:
            matches = [
                d
                for d in documents
                if d.name == case["document_name"]
                and (not case.get("document_id") or d.document_id == case["document_id"])
            ]
            if len(matches) != 1:
                raise ValueError("Document name/identity is missing or ambiguous; specify its ID")
            doc_id = matches[0].document_id
            record["document_id"] = doc_id
            record["extraction_version"] = matches[0].extraction_version
            answer = service.chat(
                ChatRequest(
                    query=case["question"],
                    language=record["language"],
                    document_ids=[doc_id],
                    use_llm=use_llm,
                )
            )
            record.update(
                response=answer.model_dump(), citation_provenance_valid=citation_provenance(answer)
            )
            try:
                validate_output_language(
                    answer.answer, resolve_language(case["question"], record["language"])
                )
                record["language_matches"] = True
            except ValueError:
                record["language_matches"] = False
            print(f"{case['id']}: {answer.generation_mode}, {answer.latency_ms:.0f} ms", flush=True)
        except (ValueError, RuntimeError, KeyError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{case['id']}: ERROR {record['error']}", flush=True)
        records.append(record)
        output.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "created_at_utc": created_at,
                    "use_llm": use_llm,
                    "model": service.settings.ollama_model,
                    "translation_model": service.settings.translation_model
                    or service.settings.ollama_model,
                    "generation_limits": {
                        "context_chars": service.settings.max_context_chars,
                        "total_timeout": service.settings.llm_total_timeout,
                        "max_statements": service.settings.max_answer_statements,
                        "question_coverage_audit": service.settings.question_coverage_audit,
                    },
                    "embedding_model": service.embedder.key,
                    "retrieval": {
                        "mode": "hybrid",
                        "embedding_backend": service.settings.embedding_backend,
                        "reranker_used": False,
                        "configured_reranker_backend": service.settings.reranker_backend,
                    },
                    "documents": [d.model_dump() for d in documents],
                    "records": records,
                    "note": "Question smoke controls. Language/script and exact source provenance checks are not semantic correctness grades.",
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/live-questions.json"))
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    run_questions(RAGService(Settings()), payload["cases"], args.output)


if __name__ == "__main__":
    main()
