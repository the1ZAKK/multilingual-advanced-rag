import argparse
import json
from pathlib import Path

from rag.config import Settings
from rag.evaluation.benchmark import EvaluationRequest, evaluate, export_results
from rag.service import RAGService

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an indexed corpus with annotated passage IDs"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    request = EvaluationRequest.model_validate(
        json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    )
    export_results(evaluate(RAGService(Settings()), request), args.output)
    print(f"Results written to {args.output}")
