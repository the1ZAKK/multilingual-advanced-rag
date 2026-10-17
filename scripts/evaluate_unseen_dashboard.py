"""Opt-in real dashboard/API evaluation of frozen held-out paper questions.

No production document writes. Uploads must be done to a separate API/index.
Source/quote checks and output-language checks are NOT semantic answer grading;
the independent source reviewers grade the saved answers separately.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pymupdf
from streamlit.testing.v1 import AppTest

from rag.query import resolve_language, validate_output_language


def source_checks(response, sources):
    evidence = {h["chunk"]["chunk_id"]: h["chunk"] for h in response["evidence"]}
    errors = []
    for citation in response["citations"]:
        chunk = evidence.get(citation["chunk_id"])
        if not chunk or any(
            citation[k] != chunk[k] for k in ("document_id", "page", "document_name")
        ):
            errors.append("Citation identity does not match indexed evidence")
            continue
        quote = citation["quote"]
        pages = sources.get(citation["document_id"], [])
        if not quote or quote not in chunk["text"]:
            errors.append("Citation quote is not an exact indexed substring")
        if not 0 < citation["page"] <= len(pages) or quote not in pages[citation["page"] - 1]:
            errors.append("Citation quote is not an exact native PDF page substring")
    markers = set(re.findall(r"\[(S\d+)\]", response["answer"]))
    if markers != {c["label"] for c in response["citations"]}:
        errors.append("Answer markers differ from citation labels")
    return {"passed": not errors, "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--settings", nargs="+", choices=["auto", "explicit"], default=["auto", "explicit"]
    )
    args = parser.parse_args()
    references = [json.loads(path.read_text(encoding="utf-8-sig")) for path in args.reference]
    cases, sources = [], {}
    for ref in references:
        paper = ref.get("paper", ref.get("source", ref))
        path = Path(paper.get("path") or paper.get("source_path") or ref.get("source_path"))
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            sources[digest] = [p.get_text(sort=False) for p in doc]
        for case in ref["cases"]:
            if not args.case or case["id"] in args.case:
                cases.append(
                    {
                        **case,
                        "query": case.get("query", case.get("question")),
                        "explicit_language": case.get(
                            "explicit_language", case.get("language", "en")
                        ),
                    }
                )
    os.environ["RAG_API_URL"] = args.api
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app/streamlit_app.py")).run(timeout=45)
    records = []
    for repeat in range(1, args.repeats + 1):
        for case in cases:
            for requested in args.settings:
                setting = "auto" if requested == "auto" else case["explicit_language"]
                app.selectbox[0].set_value(setting).run(timeout=45)
                app.chat_input[0].set_value(case["query"]).run(timeout=360)
                if app.exception:
                    raise RuntimeError(str(app.exception))
                message = app.session_state["messages"][-1]
                if "response" not in message:
                    record = {
                        "case_id": case["id"],
                        "query": case["query"],
                        "repeat": repeat,
                        "language_setting": setting,
                        "request_failed": True,
                        "dashboard_errors": [item.value for item in app.error],
                    }
                    records.append(record)
                    args.output.write_text(
                        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf8"
                    )
                    print(json.dumps(record, ensure_ascii=True), flush=True)
                    continue
                result = message["response"]
                captions = [item.value for item in app.chat_message[-1].caption]
                displayed = [c for c in captions if re.match(r"^\[S\d+\] ", c)]
                expected = {
                    (c["label"], c["document_id"], c["chunk_id"], c["page"])
                    for c in result["citations"]
                }
                try:
                    validate_output_language(
                        result["answer"], resolve_language(case["query"], setting)
                    )
                    language = {"passed": True}
                except ValueError as exc:
                    language = {"passed": False, "error": str(exc)}
                display = len(displayed) == len(expected) and len(displayed) == len(set(displayed))
                record = {
                    "case_id": case["id"],
                    "query": case["query"],
                    "repeat": repeat,
                    "language_setting": setting,
                    "response": result,
                    "language_check": language,
                    "citation_check": source_checks(result, sources),
                    "display_check": {"passed": display, "sources": displayed},
                    "mode_displayed": f"Generation mode: {result['generation_mode']}" in captions,
                }
                records.append(record)
                args.output.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf8"
                )
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "repeat": repeat,
                            "language": setting,
                            "mode": result["generation_mode"],
                            "warning": result["warning"],
                            "language_pass": language["passed"],
                            "citations_pass": record["citation_check"]["passed"],
                            "display_pass": display,
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
