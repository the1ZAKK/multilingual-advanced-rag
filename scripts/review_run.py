"""Export or summarize explicit human judgments for a fixed validation run."""

import argparse
import json
from pathlib import Path

from rag.evaluation.review import ReviewSuite, export_review, summarize_reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "summarize"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        suite = ReviewSuite.model_validate_json(
            (args.run / "reference.json").read_text(encoding="utf-8-sig")
        )
        result = json.loads((args.run / "answers.json").read_text(encoding="utf-8-sig"))
        if args.action == "export":
            export_review(suite, result, args.output)
        else:
            if args.reviews is None:
                parser.error("summarize requires --reviews")
            summary = summarize_reviews(suite, result, args.reviews)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False, indent=2))
            print(json.dumps(summary, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Review refused: {exc}\n")


if __name__ == "__main__":
    main()
