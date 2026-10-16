"""Opt-in dashboard/PDF checks using per-item, source-annotated regression assertions."""

import json
import re
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

if __package__ in {None, ""}:
    from contribution_oracle import citation_caption_verification, verify_source_answer
else:
    from scripts.contribution_oracle import citation_caption_verification, verify_source_answer

QUESTIONS = [
    "According to the 3LM paper, what problem does the work address, and what are its main contributions? Cite supporting passages and page numbers.",
    "List all three contributions explicitly stated by the authors of the 3LM paper. Number them 1–3 and cite supporting passages and pages. If the retrieved evidence does not contain all three, state which are missing.",
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="baseline")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    directory = Path("reports/3lm-diagnosis")
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    with httpx.Client(timeout=180, trust_env=False) as client:
        for query in QUESTIONS:
            for mode in ("dense", "bm25", "hybrid", "hybrid-rerank"):
                response = client.post(
                    "http://localhost:8000/search",
                    json={
                        "query": query,
                        "mode": mode,
                        "top_k": 10,
                    },
                )
                response.raise_for_status()
                results.append(
                    {"kind": "search", "query": query, "mode": mode, "response": response.json()}
                )
    # AppTest executes the real dashboard and its real HTTP client against the running API.
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")).run(
        timeout=30
    )
    failed = []
    for repeat in range(args.repeats):
        for query in QUESTIONS:
            old_captions = [c.value for c in app.caption if re.match(r"^\[S\d+\] ", c.value)]
            app.chat_input[0].set_value(query).run(timeout=300)
            if app.exception:
                raise RuntimeError(str(app.exception))
            result = app.session_state["messages"][-1]["response"]
            captions = [c.value for c in app.caption if re.match(r"^\[S\d+\] ", c.value)]
            caption_check = citation_caption_verification(captions[len(old_captions) :], result)
            results.append(
                {
                    "kind": "dashboard-chat",
                    "repeat": repeat + 1,
                    "query": query,
                    "response": result,
                    "dashboard_citation_captions": caption_check,
                }
            )
            if args.pdf:
                results[-1]["pdf_verification"] = verify_pdf(result, args.pdf, query)
            if not caption_check["passed"] or (
                args.pdf and not results[-1]["pdf_verification"]["passed"]
            ):
                failed.append({"repeat": repeat + 1, "query": query})
            print(
                json.dumps(
                    {
                        "query": query,
                        "mode": result["generation_mode"],
                        "warning": result["warning"],
                        "answer": result["answer"],
                        "source_captions_deduplicated": caption_check["passed"],
                        "pdf_verified": results[-1].get("pdf_verification", {}).get("passed"),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            (directory / f"{args.phase}.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    if failed:
        raise RuntimeError(f"Dashboard/PDF regression checks failed: {failed}")


def verify_pdf(result, path, query):
    """Verify individual contribution meanings and complete source support on PDF pages 1–2."""
    return verify_source_answer(result, path, require_problem=query == QUESTIONS[0])


if __name__ == "__main__":
    main()
