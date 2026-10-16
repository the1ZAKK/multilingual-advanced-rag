import argparse
import json
from pathlib import Path

from rag.config import Settings
from rag.evaluation.ragas_adapter import score_with_ragas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optional local Ollama/Ragas judge; requires .[evaluation,models]"
    )
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/ragas.json"))
    args = parser.parse_args()
    result = score_with_ragas(json.loads(args.samples.read_text(encoding="utf-8")), Settings())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
