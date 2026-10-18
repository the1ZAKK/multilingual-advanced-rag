"""Run source-verified paper checks in a NEW isolated index and create a review sheet."""

import argparse
import json
from pathlib import Path

from rag.config import Settings
from rag.evaluation.review import ReviewSuite, export_review, verify_sources
from rag.service import RAGService
from scripts.live_questions import run_questions


def run(
    dataset: Path,
    sources: Path,
    output: Path,
    *,
    learned=False,
    context_chars=None,
    max_statements=None,
):
    if output.exists():
        raise ValueError(
            "Output already exists; choose a new directory to preserve prior runs/indexes"
        )
    suite = ReviewSuite.model_validate_json(dataset.read_text(encoding="utf-8-sig"))
    raw = verify_sources(suite, sources)  # No output writes before reference preflight passes.
    settings = Settings()
    updates = {
        "data_dir": output / "index",
        "reranker_backend": "none" if learned else "lexical",
        "embedding_backend": "sentence-transformer" if learned else "hash",
    }
    if context_chars is not None:
        updates["max_context_chars"] = context_chars
    if max_statements is not None:
        updates["max_answer_statements"] = max_statements
    config = Settings.model_validate({**settings.model_dump(), **updates})
    output.mkdir(parents=True)
    (output / "reference.json").write_text(suite.model_dump_json(indent=2), encoding="utf-8")
    service = RAGService(config)
    if learned:
        status = service.generator.status()
        if not status["reachable"] or not status["model_available"]:
            raise ValueError(
                "Learned profile requires the configured local Ollama model; use offline for excerpts"
            )
    for source in suite.sources:
        doc = service.ingest(raw[source.id], source.filename)
        if doc.document_id != source.sha256:
            raise ValueError("Ingestion document hash differs from preflight PDF")
    by_id = {s.id: s for s in suite.sources}
    cases = [
        {
            "id": c.id,
            "question": c.question,
            "language": c.language,
            "document_name": by_id[c.source_id].filename,
            "document_id": by_id[c.source_id].sha256,
        }
        for c in suite.cases
    ]
    result_path = output / "answers.json"
    run_questions(service, cases, result_path, use_llm=learned)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    export_review(suite, result, output / "review.csv")
    print(f"Saved {len(cases)} attempts and ungraded review sheet in {output}")
    print(
        "Review the original PDFs; edit reviewer/verdict/notes only. Valid citations are not a correctness grade."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/evaluation/workflow-review.json")
    )
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["offline", "learned"], default="offline")
    parser.add_argument("--context-chars", type=int)
    parser.add_argument("--max-statements", type=int)
    args = parser.parse_args()
    try:
        run(
            args.dataset,
            args.sources,
            args.output,
            learned=args.profile == "learned",
            context_chars=args.context_chars,
            max_statements=args.max_statements,
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Validation stopped: {exc}\n")


if __name__ == "__main__":
    main()
