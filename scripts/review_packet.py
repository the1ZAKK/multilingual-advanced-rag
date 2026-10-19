"""Create an ungraded, source-bound packet for an actual independent reviewer."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from rag.evaluation.review import ReviewSuite, export_review, review_rows


def packet(dataset, run_path, output):
    if output.exists():
        raise ValueError("Review packet already exists; prior reviews are preserved")
    suite = ReviewSuite.model_validate_json(dataset.read_text(encoding="utf-8-sig"))
    result = deepcopy(json.loads(run_path.read_text(encoding="utf-8-sig")))
    for record in result["records"]:
        record.get("response", {}).pop("evidence", None)
    review_rows(suite, result)  # Exact cases, source IDs and questions before any writes.
    output.mkdir(parents=True)
    (output / "reference.json").write_text(suite.model_dump_json(indent=2), encoding="utf-8")
    (output / "answers.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    export_review(suite, result, output / "review.csv")
    (output / "REVIEW.md").write_text(
        "# Independent source review\n\n"
        "No human verdicts have been entered. Open each original PDF and check every "
        "requested part, relation, number, condition, negation, page and exact quotation. "
        "Arabic script alone does not establish faithful Arabic meaning. Read the full "
        "page, including columns, captions and surrounding qualifications.\n\n"
        "Edit only reviewer, reviewer_type, verdict and notes in review.csv. A person "
        "who actually checked the sources may use reviewer_type=human; AI-assisted "
        "review must use ai_assisted. Use pass, partial or fail, and explain partial/fail. "
        "Do not convert citation checks or generated mode into semantic grades.\n\n"
        "The packet is bound to its exact compact answers and frozen expectations; "
        "retrieved full-context passages are omitted, not replaced. Expected points were "
        "not generation input. Independent human review is still pending.\n\n"
        "After review: python -m scripts.review_run summarize --run PATH_TO_PACKET "
        "--reviews PATH_TO_PACKET/review.csv --output PATH_TO_PACKET/human-summary.json\n",
        encoding="utf-8",
    )
    print(f"Created {len(suite.cases)} ungraded source-review rows; human reviews remain zero")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet(args.dataset, args.run, args.output)


if __name__ == "__main__":
    main()
