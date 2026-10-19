"""Publish bounded source-review outputs without weights, vectors or automatic grades."""

import argparse
import json
import os
from pathlib import Path


def compact(input_path, runtime_path, output, trace=None):
    if output.exists():
        raise ValueError("Refusing to overwrite prior published evaluation output")
    run = json.loads(input_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    if run["model"] != runtime["installed_model"]["name"]:
        raise ValueError("Run/model verification metadata disagree")
    report = {
        k: run[k]
        for k in [
            "run_id",
            "created_at_utc",
            "model",
            "translation_model",
            "generation_limits",
            "embedding_model",
            "retrieval",
            "documents",
        ]
    }
    report.update(
        runtime=runtime,
        source_code_sha=os.environ.get("GITHUB_SHA"),
        independent_human_reviews=0,
        review_status="unreviewed; provenance/language flags are not semantic grades",
    )
    report["records"] = []
    for record in run["records"]:
        item = {k: v for k, v in record.items() if k != "response"}
        if "response" in record:
            item["response"] = {k: v for k, v in record["response"].items() if k != "evidence"}
        report["records"].append(item)
    if trace is not None and trace.exists():
        report["translation_checks"] = []
        report["generation_attempts"] = []
        for line in trace.read_text().splitlines():
            item = json.loads(line)
            if item["stage"] == "generation":
                report["generation_attempts"].append(
                    {
                        "question": item["request"].get("question"),
                        "language": item["request"].get("language"),
                        "validation_error": item["request"].get("validation_error"),
                        "result": item["result"],
                    }
                )
            if item["stage"] in {"ordinary-translation-ar", "ordinary-translation-en"}:
                report["translation_checks"].append(
                    {
                        "stage": item["stage"],
                        "input": item["request"].get("text"),
                        "output": item["result"].get("text"),
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(report['records'])} ungraded outputs for source review")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    compact(args.input, args.runtime, args.output, args.trace)


if __name__ == "__main__":
    main()
