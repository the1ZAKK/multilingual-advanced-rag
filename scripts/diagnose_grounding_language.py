"""Opt-in dashboard or saved-response quality audit against independent PDF references.

No document/index writes. Output separates language/meaning, citation provenance,
and generation metadata. Live runs use the actual dashboard/HTTP backend.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pymupdf

if __package__ in {None, ""}:
    from grounding_language_oracle import REFERENCE, evaluate_answer
else:
    from scripts.grounding_language_oracle import REFERENCE, evaluate_answer


def pdf_pages(path):
    data = path.read_bytes()
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        pages = [page.get_text(sort=False) for page in pdf]
    return hashlib.sha256(data).hexdigest(), pages


def evaluate_records(records, reference, sources):
    audited = []
    for record in records:
        case = {**record}
        case["independent_quality"] = evaluate_answer(
            record["query"],
            record["response"],
            reference,
            sources,
            expected_language=None
            if record.get("language_setting") == "auto"
            else record.get("language_setting"),
        )
        audited.append(case)
    return audited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, help="Evaluate saved dashboard JSON without generation"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Run real Streamlit AppTest and API"
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("reports/grounding-language/queries.json")
    )
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if bool(args.input) == args.dashboard:
        parser.error("Select exactly one of --input or --dashboard")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    digest, pages = pdf_pages(args.pdf)
    if digest != reference["sha256"]:
        raise ValueError("Independent reference hash does not match the inspected PDF")
    sources = {digest: pages}
    if args.input:
        records = json.loads(args.input.read_text(encoding="utf-8"))
        records = [record for record in records if "response" in record and "query" in record]
    else:
        from streamlit.testing.v1 import AppTest

        cases = json.loads(args.queries.read_text(encoding="utf-8"))
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")
        ).run(timeout=30)
        records = []
        for repeat in range(args.repeats):
            for setting_type in ("auto", "explicit"):
                for case in cases:
                    setting = "auto" if setting_type == "auto" else case["explicit_language"]
                    app.selectbox[0].set_value(setting).run(timeout=30)
                    app.chat_input[0].set_value(case["query"]).run(timeout=300)
                    if app.exception:
                        raise RuntimeError(str(app.exception))
                    result = app.session_state["messages"][-1]["response"]
                    records.append(
                        {
                            "repeat": repeat + 1,
                            "query": case["query"],
                            "language_setting": setting,
                            "response": result,
                        }
                    )
                    partial = evaluate_records(records, reference, sources)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    print(
                        json.dumps(
                            {
                                "repeat": repeat + 1,
                                "language_setting": setting,
                                "query": case["query"],
                                "independent_quality": partial[-1]["independent_quality"],
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )
    audited = evaluate_records(records, reference, sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audited, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(audited),
                "language_passed": sum(
                    r["independent_quality"]["language_following"]["passed"] for r in audited
                ),
                "answer_quality_passed": sum(
                    r["independent_quality"]["answer_quality"]["passed"] for r in audited
                ),
                "citation_validity_passed": sum(
                    r["independent_quality"]["citation_validity"]["passed"] for r in audited
                ),
                "generation_modes": [r["independent_quality"]["generation_mode"] for r in audited],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
