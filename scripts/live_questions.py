"""Run annotated local paper questions and preserve outputs for independent review.

No model answer is a correctness verdict. This tool records methods, language
following and exact citation provenance separately; humans inspect the claims.
It reads an existing corpus and never uploads, deletes or reindexes documents.
"""

import argparse
import json
from pathlib import Path

from rag.config import Settings
from rag.evaluation.benchmark import citation_provenance
from rag.models import ChatRequest
from rag.query import resolve_language, validate_output_language
from rag.service import RAGService


def run_questions(service, cases, output: Path):
    documents = {d.name: d.document_id for d in service.documents()}
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        record = {
            "id": case["id"],
            "question": case["question"],
            "language": case.get("language", "auto"),
        }
        try:
            doc_id = documents[case["document_name"]]
            answer = service.chat(
                ChatRequest(
                    query=case["question"],
                    language=record["language"],
                    document_ids=[doc_id],
                    use_llm=True,
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
                    "model": service.settings.ollama_model,
                    "embedding_model": service.embedder.key,
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
