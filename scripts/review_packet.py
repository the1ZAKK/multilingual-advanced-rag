"""Create an ungraded, source-bound packet for an actual independent reviewer."""

import argparse
import hashlib
import html
import json
import zipfile
from copy import deepcopy
from pathlib import Path

from rag.evaluation.review import ReviewSuite, export_review, review_rows, verify_sources


def packet(dataset, run_path, output, *, sources=None, archive=None):
    if output.exists():
        raise ValueError("Review packet already exists; prior reviews are preserved")
    suite = ReviewSuite.model_validate_json(dataset.read_text(encoding="utf-8-sig"))
    originals = verify_sources(suite, sources) if sources is not None else None
    if archive is not None and (originals is None or archive.exists()):
        raise ValueError("A new portable archive requires verified original PDFs and a new path")
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
    if originals is not None:
        pdf_dir = output / "PDFs"
        pdf_dir.mkdir()
        for source in suite.sources:
            (pdf_dir / source.filename).write_bytes(originals[source.id])
        source_by_id = {source.id: source for source in suite.sources}
        record_by_id = {record["id"]: record for record in result["records"]}
        sections = []
        for case in suite.cases:
            record = record_by_id[case.id]
            response = record.get("response", {})
            citations = "".join(
                f"<li>{html.escape(c['label'])}, PDF p. {c['page']}: "
                f"<pre dir='auto'>{html.escape(c['quote'])}</pre></li>"
                for c in response.get("citations", [])
            )
            sections.append(
                f"<section><h2>{html.escape(case.id)}</h2>"
                f"<a href='PDFs/{html.escape(source_by_id[case.source_id].filename, quote=True)}'>"
                "Open original PDF</a>"
                f"<h3>Question</h3><p dir='auto'>{html.escape(case.question)}</p>"
                f"<h3>Exact recorded answer ({html.escape(response.get('generation_mode', 'error'))})</h3>"
                f"<pre dir='auto'>{html.escape(response.get('answer', record.get('error', '')))}</pre>"
                f"<p>{html.escape(response.get('warning') or '')}</p>"
                f"<details><summary>Exact citations and PDF pages</summary><ul>{citations}</ul></details>"
                "<details><summary>AI-curated reference guide; verify independently</summary>"
                f"<pre dir='auto'>{html.escape(json.dumps(case.expected_points, ensure_ascii=False, indent=2))}</pre>"
                "</details></section>"
            )
        (output / "START_HERE.html").write_text(
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Ungraded source review</title>"
            "<style>body{font:16px system-ui;max-width:1050px;margin:2em auto;padding:1em}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border-top:1px solid #aaa;margin-top:2em}</style>"
            "<h1>Review the exact new answers against the original PDFs</h1>"
            "<p>These are development controls, not unseen evaluation. No human or AI verdicts are prefilled. "
            "Check every requested fact, language, numeric relationship, quotation and PDF page. "
            "Source provenance and generated mode are not answer-quality acceptance.</p>"
            "<p>Open review.csv in Excel. Edit only reviewer, reviewer_type, verdict and notes. "
            "Use human only for a person's source review; use ai_assisted for AI-assisted judgments. "
            "Use pass, partial or fail and explain errors with PDF page numbers. Save as UTF-8 CSV.</p>"
            f"<p>Source commit: {html.escape(str(result.get('source_code_sha', 'not recorded')))}. "
            f"Run: {html.escape(result['run_id'])}. Model: {html.escape(str(result.get('model', 'not recorded')))}.</p>"
            + "".join(sections)
            + "</html>",
            encoding="utf-8",
        )
        manifest = {
            "source_code_sha": result.get("source_code_sha"),
            "run_id": result["run_id"],
            "human_reviews": 0,
            "verdicts": "ungraded",
            "files": {
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(output))
    print(f"Created {len(suite.cases)} ungraded source-review rows; human reviews remain zero")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--zip", type=Path, dest="archive")
    args = parser.parse_args()
    packet(args.dataset, args.run, args.output, sources=args.sources, archive=args.archive)


if __name__ == "__main__":
    main()
