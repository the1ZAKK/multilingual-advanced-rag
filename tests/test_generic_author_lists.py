import pymupdf
import pytest

from rag.generation.ollama import authored_contributions
from rag.ingestion.pdf import parse_pdf
from rag.ingestion.structure import contribution_declarations, contribution_markers, reading_text
from rag.models import Hit


@pytest.mark.parametrize(
    "declaration",
    [
        "We highlight three main contributions:",
        "Our key contributions are as follows:",
        "The contributions are three-fold, including:",
    ],
)
def test_inline_author_lists_preserve_first_item_and_count(settings, declaration):
    text = (
        declaration
        + " (i) We release a corpus. (ii) We evaluate baselines. (iii) We report a robustness study."
    )
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 540, 700), text, fontsize=11)
    _, chunks = parse_pdf(
        doc.tobytes(), "IndependentStudy.pdf", settings.model_copy(update={"chunk_size": 900})
    )
    evidence = [Hit(chunk=c, score=1) for c in chunks if c.section == "Contributions"]
    units = authored_contributions(evidence)
    assert len(units) == 3
    assert list(units.values())[0]["text"] == "We release a corpus."
    assert list(units.values())[-1]["text"] == "We report a robustness study."


def test_declaration_does_not_consume_colon_inside_first_item():
    text = "Our contributions are: (1) We introduce a system: it searches documents. (2) We evaluate it."
    declaration = contribution_declarations(text)[0]
    markers = contribution_markers(text, declaration[1])
    assert markers[0][2] == 1 and text[markers[0][1] :].startswith("We introduce")


def test_one_tall_block_in_each_column_is_not_line_interleaved():
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text(
        (50, 70),
        "Left column begins with a research question.\nIt continues with a methodological statement.\nIts third line discusses the experiment.\nIts final line reports the conclusion.",
        fontsize=10,
    )
    page.insert_text(
        (320, 70),
        "Right column contains separate discussion.\nIt must follow the whole first column.\nThis line is independent source context.\nThe final line ends the second column.",
        fontsize=10,
    )
    text = reading_text(page)
    assert text.index("final line reports") < text.index("Right column")


def test_single_column_native_order_keeps_margin_stamp_out_of_method_sentence(settings):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((230, 80), "Abstract", fontsize=12)
    page.insert_text(
        (110, 120),
        "We propose an evaluation framework.\nIt uses a self-\ndeliberation mechanism verified by experts.",
        fontsize=11,
    )
    page.insert_text((25, 170), "arXiv:0000.00000", fontsize=10, rotate=90)
    native = page.get_text(sort=False)
    assert reading_text(page) == native
    assert "self-\ndeliberation" in reading_text(page)
    _, chunks = parse_pdf(doc.tobytes(), "OneColumn.pdf", settings)
    abstract = [c for c in chunks if c.section == "Abstract"]
    assert any("self-\ndeliberation" in c.text for c in abstract)
    assert all(c.text in native for c in abstract)


def test_overlap_does_not_begin_or_end_inside_ordinary_words():
    from rag.ingestion.structure import chunk_spans

    text = " ".join(["methodology", "evaluation", "framework", "reasoning"] * 30)
    spans = chunk_spans(text, 100, 23, [(0, None)])
    assert len(spans) > 5
    for start, end, _ in spans:
        assert end - start <= 100
        assert start == 0 or not (text[start - 1].isalnum() and text[start].isalnum())
        assert end == len(text) or not (text[end - 1].isalnum() and text[end].isalnum())
    assert spans[-1][1] == len(text)
