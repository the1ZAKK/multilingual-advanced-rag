"""Generate synthetic PDFs and bind demonstration annotations to actual stable chunk IDs."""

import json
from pathlib import Path

import pymupdf

from rag.config import Settings
from rag.evaluation.benchmark import EvaluationRequest, evaluate, export_results
from rag.service import RAGService

DEMO = [
    (
        "retrieval.pdf",
        "What combines dense and lexical search?",
        "Hybrid retrieval combines semantic vector search and BM25 keyword search. Reciprocal Rank Fusion merges ranked lists.",
    ),
    (
        "citations.pdf",
        "How are citations verified?",
        "Page citations identify the document and page of each evidence passage. Exact source quotes make references verifiable.",
    ),
]


def main():
    settings = Settings()
    service = RAGService(settings)
    cases = []
    for name, question, text in DEMO:
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), text, fontsize=11)
        info = service.ingest(pdf.tobytes(no_new_id=True), name)
        pdf.close()
        relevant = {
            c.chunk_id: 1.0 for c in service.index.chunks if c.document_id == info.document_id
        }
        cases.append(
            {
                "id": name,
                "question": question,
                "relevant": relevant,
                "reference_answer": text,
                "language": "en",
            }
        )
    modes = ["dense", "bm25", "hybrid"]
    if settings.reranker_backend != "none":
        modes.append("hybrid-rerank")
    dataset = {
        "description": "Synthetic English pipeline demonstration, NOT a scientific or multilingual benchmark.",
        "cases": cases,
        "modes": modes,
        "generate_answers": True,
        "use_llm": False,
    }
    output = Path("reports/demo")
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    result = evaluate(service, EvaluationRequest.model_validate(dataset))
    export_results(result, output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
