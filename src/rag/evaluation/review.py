"""Source-verified review cases and explicit reviewer verdicts, separate from model judges."""

import csv
import hashlib
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pymupdf
from pydantic import BaseModel, Field, model_validator


class SourceReference(BaseModel):
    id: str
    filename: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def local_filename(self):
        if Path(self.filename).name != self.filename or "\\" in self.filename:
            raise ValueError("Reference filename must be a single local filename")
        return self


class SourceQuote(BaseModel):
    page: int = Field(ge=1)
    text: str = Field(min_length=1)


class ReviewCase(BaseModel):
    id: str
    source_id: str
    question: str = Field(min_length=1)
    language: Literal["auto", "en", "ar"] = "auto"
    expected_points: list[str] = Field(min_length=1)
    references: list[SourceQuote] = Field(min_length=1)


class ReviewSuite(BaseModel):
    description: str
    sources: list[SourceReference] = Field(min_length=1)
    cases: list[ReviewCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        source_ids = [s.id for s in self.sources]
        case_ids = [c.id for c in self.cases]
        filenames = [s.filename for s in self.sources]
        if (
            len(set(source_ids)) != len(source_ids)
            or len(set(case_ids)) != len(case_ids)
            or len(set(filenames)) != len(filenames)
        ):
            raise ValueError("Source and case IDs must be unique")
        if any(c.source_id not in source_ids for c in self.cases):
            raise ValueError("Case references an unknown source")
        return self


def verify_sources(suite: ReviewSuite, directory: Path) -> dict[str, bytes]:
    """Read-only preflight: verify original PDF hashes and exact reference quotes."""
    data = {}
    for source in suite.sources:
        path = directory / source.filename
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source.sha256:
            raise ValueError(f"Wrong PDF bytes for {source.id}; use the referenced paper version")
        with pymupdf.open(stream=raw, filetype="pdf") as pdf:
            for case in suite.cases:
                if case.source_id != source.id:
                    continue
                for quote in case.references:
                    if quote.page > len(pdf) or quote.text not in pdf[quote.page - 1].get_text():
                        raise ValueError(f"Reference quote/page mismatch for case {case.id}")
        data[source.id] = raw
    return data


EDITABLE = {"reviewer", "reviewer_type", "verdict", "notes"}
VERDICTS = {"pass", "partial", "fail", "not_reviewed"}


def spreadsheet_text(value) -> str:
    text = str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def chat_review_export(messages: list[dict]) -> dict:
    """Export completed local chat turns without inventing expected answers or grades."""
    turns = []
    question = None
    for message in messages:
        if message.get("role") == "user":
            question = message["text"]
        elif (
            message.get("role") == "assistant"
            and question is not None
            and ("response" in message or "error" in message)
        ):
            turns.append(
                {
                    "question": question,
                    "request": deepcopy(message.get("request")),
                    "response": deepcopy(message.get("response")),
                    "error": message.get("error"),
                    "review": {
                        "expected_answer": "",
                        "reviewer": "",
                        "reviewer_type": "",
                        "verdict": "not_reviewed",
                        "notes": "",
                    },
                }
            )
            question = None
    return {
        "schema_version": 1,
        "export_id": str(uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "turns": turns,
        "note": "Local chat review export, including cited passages. Expected answers and semantic judgments must be supplied by a reviewer.",
    }


def review_rows(suite: ReviewSuite, result: dict) -> list[dict]:
    """Bind human reviews to exact run, question, expectations and returned answer."""
    if not result.get("run_id"):
        raise ValueError("Results lack a run_id; rerun the current question runner")
    cases = {c.id: c for c in suite.cases}
    sources = {s.id: s for s in suite.sources}
    records = {r["id"]: r for r in result["records"]}
    if len(records) != len(result["records"]) or set(records) != set(cases):
        raise ValueError("Result IDs must match the review suite exactly, without duplicates")
    rows = []
    for case in suite.cases:
        record = records[case.id]
        source = sources[case.source_id]
        if record["question"] != case.question or record["language"] != case.language:
            raise ValueError(f"Question/language mismatch for {case.id}")
        if record.get("document_id") != source.sha256:
            raise ValueError(f"Source identity mismatch for {case.id}")
        response = record.get("response", {})
        answer = response.get("answer", "")
        identity = {"response": response, "error": record.get("error")}
        fingerprint = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        rows.append(
            {
                "run_id": result["run_id"],
                "case_id": case.id,
                "response_sha256": fingerprint,
                "paper": source.filename,
                "question": case.question,
                "answer_language": case.language,
                "expected_points": json.dumps(case.expected_points, ensure_ascii=False),
                "reference_pages": ", ".join(
                    str(p) for p in sorted({q.page for q in case.references})
                ),
                "mode": response.get("generation_mode", "request-error"),
                "answer": answer,
                "warning_or_error": response.get("warning") or record.get("error") or "",
                "citation_provenance_check": record.get("citation_provenance_valid"),
                "language_check": record.get("language_matches"),
                "reviewer": "",
                "reviewer_type": "",
                "verdict": "not_reviewed",
                "notes": "",
            }
        )
    return [{key: spreadsheet_text(value) for key, value in row.items()} for row in rows]


def export_review(suite: ReviewSuite, result: dict, output: Path):
    rows = review_rows(suite, result)
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_reviews(suite: ReviewSuite, result: dict, path: Path) -> dict:
    expected = {row["case_id"]: row for row in review_rows(suite, result)}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        actual = list(csv.DictReader(handle))
    if (
        len(actual) != len(expected)
        or any("case_id" not in r for r in actual)
        or {r["case_id"] for r in actual} != set(expected)
    ):
        raise ValueError("Review rows must match the run exactly")
    for row in actual:
        baseline = expected[row["case_id"]]
        if set(row) != set(baseline) or any(
            row[key] != value for key, value in baseline.items() if key not in EDITABLE
        ):
            raise ValueError("Immutable review fields changed or belong to a different run")
        if row["verdict"] not in VERDICTS:
            raise ValueError("Unknown human review verdict")
        if row["verdict"] != "not_reviewed" and not row["reviewer"].strip():
            raise ValueError("A completed verdict requires a reviewer")
        if row["verdict"] != "not_reviewed" and row["reviewer_type"] not in {
            "human",
            "ai_assisted",
        }:
            raise ValueError("A completed verdict requires reviewer_type human or ai_assisted")
        if row["verdict"] in {"fail", "partial"} and not row["notes"].strip():
            raise ValueError("Failed/partial reviews require a reproducible explanation")
    counts = Counter(row["verdict"] for row in actual)
    reviewed = len(actual) - counts["not_reviewed"]
    by_type = {}
    for kind in ("human", "ai_assisted"):
        group = [r for r in actual if r["reviewer_type"] == kind and r["verdict"] != "not_reviewed"]
        group_counts = Counter(r["verdict"] for r in group)
        by_type[kind] = {
            "reviewed": len(group),
            "counts": {key: group_counts[key] for key in ("pass", "partial", "fail")},
            "pass_rate": group_counts["pass"] / len(group) if group else None,
        }
    return {
        "run_id": result["run_id"],
        "cases": len(actual),
        "reviewed": reviewed,
        "counts": {key: counts[key] for key in sorted(VERDICTS)},
        "by_reviewer_type": by_type,
        "note": "Recorded judgments on development controls, split by human versus AI-assisted review; not an automatic or unseen-paper accuracy score.",
    }
