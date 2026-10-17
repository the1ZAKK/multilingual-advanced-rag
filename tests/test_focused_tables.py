import copy

import pymupdf
import pytest
from test_table_answer import _fixture

from rag.generation.table_answer import table_answer
from rag.ingestion.tables import extract_result_tables
from rag.query import models_results_question
from rag.retrieval.table_selection import select_tables


def test_caption_above_not_replaced_by_next_table_caption():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Table 2: Accuracy on independent tasks.", fontsize=10)
    for x, label in [(50, "Model"), (220, "TaskA"), (320, "TaskB")]:
        page.insert_text((x, 90), label, fontsize=10)
    for y, name, a, b in [(110, "Oriole", "75.02%", "74.62%"), (130, "Heron", "75.57%", "79.03%")]:
        for x, text in [(50, name), (220, a), (320, b)]:
            page.insert_text((x, y), text, fontsize=10)
    page.insert_text((50, 148), "Table 3: Performance on unrelated tasks.", fontsize=10)
    table = next(t for t in extract_result_tables(doc.tobytes()) if t["status"] == "ok")
    assert table["caption"].startswith("Table 2:")
    assert table["rows"][0]["entity"] == "Oriole"
    assert [c["value"] for c in table["rows"][1]["cells"]] == ["75.57%", "79.03%"]


def test_named_models_keep_weaker_comparison_rows_and_exact_values():
    evidence = _fixture()
    original = copy.deepcopy(evidence[0].tables)
    evidence[0].tables = select_tables(original, "Compare Crow and Heron in Table 7: give scores.")
    result = table_answer(evidence, "en")
    assert "| Crow | 9B | 9.5 | 15.3" in result.answer
    assert "| Heron | 8B | 11.1 | 19.4" in result.answer
    assert "| Sparrow |" not in result.answer
    assert "Focused view" in result.answer
    assert len(original[0]["rows"]) == 3  # Selection never mutates cached source data.


def test_explicit_table_and_condition_never_return_unrelated_table():
    tables = _fixture()[0].tables
    assert select_tables(tables, "Show Table 8 accuracy values") == []
    tables[0]["caption"] += " In zero-shot evaluation."
    assert select_tables(tables, "Table 7 zero-shot scores")
    assert select_tables(tables, "Table 7 few-shot scores") == []


@pytest.mark.parametrize(
    "query",
    [
        "Which models and API temperature and how many demonstrations were used?",
        "ما النماذج ودرجة الحرارة المستخدمة وعدد الامثلة التوضيحية؟",
        "How were numerical model results measured using the prompting protocol?",
    ],
)
def test_protocol_quantities_do_not_force_numerical_table_route(query):
    assert not models_results_question(query)


def test_arabic_table_digits_resolve_to_source_table_number():
    tables = _fixture()[0].tables
    assert select_tables(tables, "قارن نتائج النماذج في الجدول ٧")
    assert select_tables(tables, "قارن نتائج النماذج في الجدول ٨") == []
