import copy
from pathlib import Path

import pymupdf
import pytest

from rag.config import Settings
from rag.generation.table_answer import table_answer
from rag.ingestion.pdf import parse_pdf
from rag.ingestion.structure import reading_text
from rag.ingestion.tables import extract_result_tables
from rag.models import Chunk, Hit


def _hit(text, page=1, document="other-paper", tables=None):
    return Hit(
        chunk=Chunk(
            chunk_id=f"{document}:p{page}:0-{len(text)}",
            document_id=document,
            document_name="Different research paper.pdf",
            page=page,
            text=text,
            normalized=text.lower(),
            start=0,
            end=len(text),
        ),
        score=1,
    ).model_copy(update={"tables": tables or []})


def _fixture():
    caption = "Table 7: Base models accuracy on reasoning benchmarks."
    native = (
        caption + "\nEnglish Arabic\nTaskA TaskA\nModel Size\n"
        "Sparrow 7B 10.1 20.2\nHeron 8B 11.1 19.4\nCrow 9B 9.5 15.3\n"
        "Machine Translation is a curation process. TaskA is a benchmark."
    )

    def source(text):
        return [{"text": text, "bbox": [10, 20, 30, 40], "word_ids": [1, 2, 3]}]

    columns = [
        {
            "column_id": f"C{i}",
            "path": [language, "TaskA"],
            "source": source(language) + source("TaskA"),
            "language": language,
            "benchmark": "TaskA",
            "condition": None,
            "metric": "accuracy",
        }
        for i, language in enumerate(["English", "Arabic"], 1)
    ]
    rows = []
    for j, (entity, size, values) in enumerate(
        [
            ("Sparrow", "7B", ["10.1", "20.2"]),
            ("Heron", "8B", ["11.1", "19.4"]),
            ("Crow", "9B", ["9.5", "15.3"]),
        ],
        1,
    ):
        rows.append(
            {
                "row_id": f"R{j}",
                "entity": entity,
                "size": size,
                "entity_source": source(entity),
                "size_source": source(size),
                "cells": [
                    {
                        "column_id": column["column_id"],
                        "path": column["path"],
                        "value": value,
                        "source": source(value),
                    }
                    for column, value in zip(columns, values)
                ],
            }
        )
    table = {
        "table_id": "p1:t7",
        "status": "ok",
        "page": 1,
        "caption": caption,
        "native_page_text": native,
        "context": {"variant": "base"},
        "columns": columns,
        "rows": rows,
    }
    return [_hit(native, tables=[table])]


def test_catalog_distinguishes_models_from_benchmarks_and_curation_and_retains_numbers():
    evidence = _fixture()
    result = table_answer(evidence, "en")
    assert result.mode == "source-tables"
    catalog = next(
        line for line in result.answer.splitlines() if line.startswith("Models and sizes")
    )
    assert "Sparrow (7B)" in catalog and "Heron (8B)" in catalog and "Crow (9B)" in catalog
    assert "TaskA" not in catalog and "Machine Translation" not in catalog
    assert "English / TaskA" in result.answer and "Arabic / TaskA" in result.answer
    assert "| Sparrow | 7B | 10.1 | 20.2" in result.answer
    assert "| Heron | 8B | 11.1 | 19.4" in result.answer
    assert "| Crow |" not in result.answer  # Catalog retains non-selected models.
    assert "Metric: accuracy" in result.answer
    assert not any(line.startswith("1.") for line in result.answer.splitlines())
    assert all(c.quote in evidence[int(c.label[1:]) - 1].chunk.text for c in result.citations)


def test_maximum_selection_is_dynamic_instead_of_paper_or_model_specific():
    evidence = _fixture()
    table = evidence[0].tables[0]
    table["rows"][2]["cells"][1]["value"] = "25.7"
    table["rows"][2]["cells"][1]["source"][0]["text"] = "25.7"
    evidence[0].chunk.text = evidence[0].chunk.text.replace("15.3", "25.7")
    table["native_page_text"] = table["native_page_text"].replace("15.3", "25.7")
    result = table_answer(evidence, "en")
    assert "| Crow | 9B | 9.5 | 25.7" in result.answer
    assert "| Sparrow |" not in result.answer


@pytest.mark.parametrize(
    "mutation", ["value", "entity", "size", "path", "missing_cell", "wrong_doc"]
)
def test_corrupted_native_relations_fail_closed_without_weakening_citations(mutation):
    evidence = _fixture()
    table = evidence[0].tables[0]
    row = table["rows"][0]
    if mutation == "value":
        row["cells"][0]["value"] = "83.5"
    elif mutation == "entity":
        row["entity"] = "TaskA"
    elif mutation == "size":
        row["size"] = "540B"
    elif mutation == "path":
        row["cells"][0]["path"] = ["Arabic", "TaskA"]
    elif mutation == "missing_cell":
        row["cells"].pop()
    else:
        table["document_id"] = "different-document"
    assert table_answer(evidence, "en") is None


def test_metric_binding_requires_direct_declaration_and_exact_provenance():
    evidence = _fixture()
    table = evidence[0].tables[0]
    table["caption"] = table["caption"].replace("accuracy on reasoning", "results on code")
    evidence[0].chunk.text = evidence[0].chunk.text.replace(
        "accuracy on reasoning", "results on code"
    )
    table["native_page_text"] = table["native_page_text"].replace(
        "accuracy on reasoning", "results on code"
    )
    for column in table["columns"]:
        column["metric"] = None
    method = "For code benchmarks, pass@1 evaluation was adopted using greedy generation."
    evidence.append(_hit(method, page=2))
    table["metric_bindings"] = [
        {
            "metric": "pass@1",
            "scope": "code",
            "label": "S2",
            "quote": method,
            "native_page_text": method,
        }
    ]
    result = table_answer(evidence, "en")
    assert "Metric: pass@1" in result.answer
    assert any(c.page == 2 and c.quote == method for c in result.citations)
    table["metric_bindings"][0]["quote"] = method.replace("pass@1", "pass@2")
    assert table_answer(evidence, "en") is None
    # An exact quote about unrelated source data does not entail a metric claim.
    table["metric_bindings"][0]["quote"] = evidence[0].chunk.text
    table["metric_bindings"][0]["label"] = "S1"
    assert table_answer(evidence, "en") is None


def test_missing_native_fragment_is_logged_and_no_partial_model_catalog_returned(caplog):
    evidence = _fixture()
    evidence[0].chunk.text = evidence[0].chunk.text.replace("Crow", "(missing entity)")
    assert table_answer(evidence, "en") is None
    assert "Native table fragment missing" in caplog.text


def test_fixed_answer_labels_follow_arabic_while_source_identifiers_remain_unchanged():
    answer = table_answer(_fixture(), "ar").answer
    assert "النماذج وأحجامها" in answer and "صفوف مختارة" in answer
    assert "| النموذج | الحجم | English / TaskA | Arabic / TaskA |" in answer
    assert "Models and sizes" not in answer


def test_subject_breakdown_is_excluded_when_an_overlapping_aggregate_is_available():
    evidence = _fixture()
    original = evidence[0].tables[0]
    for column in original["columns"]:
        column["condition"] = "MCQ" if column["column_id"] == "C1" else "Completion"
    aggregate = copy.deepcopy(original)
    aggregate["table_id"] = "aggregate"
    aggregate["columns"][0]["benchmark"] = "Native"
    aggregate["columns"][1]["benchmark"] = "Synthetic"
    breakdown = copy.deepcopy(original)
    breakdown["table_id"] = "breakdown"
    for i in range(3, 5):
        column = copy.deepcopy(breakdown["columns"][0])
        column.update(column_id=f"C{i}", benchmark=f"Subject{i}", path=[f"Subject{i}", "MCQ"])
        breakdown["columns"].append(column)
    evidence[0] = evidence[0].model_copy(update={"tables": [aggregate, breakdown]})
    result = table_answer(evidence, "en")
    assert result is not None
    assert result.answer.count("Models and sizes") == 1


def test_control_paper_dashboard_rows_are_complete_semantic_tuples_when_available():
    paper = Path(
        r"C:\Users\brimz\Desktop\source papers\14-Chain-of-Thought Prompting Elicits Reasoning.pdf"
    )
    if not paper.exists():
        pytest.skip("User's independent control paper is unavailable")
    data = paper.read_bytes()
    tables = [t for t in extract_result_tables(data) if t["page"] == 21 and t["status"] == "ok"]
    with pymupdf.open(stream=data, filetype="pdf") as document:
        native = reading_text(document[20])
    evidence = [_hit(native, page=21, tables=tables)]
    result = table_answer(evidence, "en")
    assert "GSM8K / standard | GSM8K / CoT" in result.answer
    row = next(line for line in result.answer.splitlines() if line.startswith("| PaLM | 540B |"))
    assert row.startswith("| PaLM | 540B | 17.9 | 56.9 |")
    catalog = next(
        line for line in result.answer.splitlines() if line.startswith("Models and sizes")
    )
    assert all(name in catalog for name in ["UL2", "LaMDA", "GPT", "Codex", "PaLM"])
    assert "GSM8K" not in catalog and "CoT" not in catalog
    assert all(c.quote in native for c in result.citations)
    with pymupdf.open(stream=data, filetype="pdf") as document:
        original_native = document[20].get_text("text", sort=False)
    assert all(c.quote in original_native for c in result.citations)


def test_reordered_uploaded_pdf_quotes_are_contiguous_in_native_and_indexed_text():
    paper = Path(r"C:\Users\brimz\Desktop\source papers\3LM-Bridging Arabic, STEM, and Code.pdf")
    if not paper.exists():
        pytest.skip("User's uploaded source paper is unavailable")
    data = paper.read_bytes()
    tables = [
        t
        for t in extract_result_tables(data)
        if t["status"] == "ok" and t["page"] in {6, 8, 21, 22}
    ]
    _, chunks = parse_pdf(data, paper.name, Settings(chunk_size=900, chunk_overlap=150))
    evidence = []
    for chunk in chunks:
        page_tables = [t for t in tables if t["page"] == chunk.page]
        if page_tables:
            evidence.append(Hit(chunk=chunk, score=1).model_copy(update={"tables": page_tables}))
    result = table_answer(evidence, "en")
    assert result is not None
    assert {c.page for c in result.citations} == {6, 8, 21, 22}
    with pymupdf.open(stream=data, filetype="pdf") as document:
        native_pages = {
            page: document[page - 1].get_text("text", sort=False) for page in {6, 8, 21, 22}
        }
    for citation in result.citations:
        assert citation.quote in native_pages[citation.page]
        assert citation.quote in next(
            hit.chunk.text for hit in evidence if hit.chunk.chunk_id == citation.chunk_id
        )
    for table in tables:
        if table["caption"].splitlines()[0] not in result.answer:
            continue  # Overlapping subject-level tables are deliberately excluded.
        quotes = [c.quote for c in result.citations if c.page == table["page"]]
        for row in table["rows"]:
            for fragment in row["entity_source"] + row["size_source"]:
                assert any(fragment["text"] in quote for quote in quotes)
            for cell in row["cells"]:
                assert any(cell["source"][0]["text"] in quote for quote in quotes)
