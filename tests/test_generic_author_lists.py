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


def test_publisher_metadata_does_not_enter_cross_page_contribution_or_weaken_refs():
    from copy import deepcopy

    from rag.generation.ollama import quote_catalog, validate_answer
    from rag.models import Chunk

    heading = "Proceedings of The Research Conference, pages 42–63\nNovember 8-9, 2025 ©2025 Research Association"
    start = (
        "The contributions of this paper are threefold:\nFirst, we release a corpus with careful"
    )
    p1 = start + "\n\n42\n\n" + heading
    p2 = "translation with human verification. Second, we evaluate forty models. Third, we analyze robustness."
    evidence = []
    for page, text, native in ((1, p1, heading + "\n" + start + "\n42"), (2, p2, p2)):
        c = Chunk(
            chunk_id=f"doc:p{page}:0",
            document_id="doc",
            document_name="Study.pdf",
            page=page,
            start=0,
            end=len(text),
            text=text,
            normalized=text,
            section="Contributions",
        )
        evidence.append(Hit(chunk=c, score=1, native_page_text=native))
    catalog = quote_catalog(evidence)
    units = authored_contributions(evidence)
    assert (
        units["First"]["text"]
        == "First, we release a corpus with careful translation with human verification."
    )
    assert all(
        "Proceedings" not in catalog[tuple(ref.split(":"))]
        for unit in units.values()
        for ref in unit["refs"]
    )
    payload = {
        "statements": [
            dict(text=u["text"], citations=[dict(ref_id=r) for r in u["refs"]])
            for u in units.values()
        ]
    }
    answer, citations = validate_answer(
        payload, evidence, question="List all contributions", language="en"
    )
    assert "forty models" in answer and "Proceedings" not in answer
    assert all(c.quote in evidence[int(c.label[1:]) - 1].native_page_text for c in citations)
    bad = deepcopy(payload)
    bad["statements"][0]["citations"].pop()
    with pytest.raises(ValueError, match="Incomplete supporting references"):
        validate_answer(bad, evidence, question="List all contributions", language="en")


def test_unknown_edge_text_and_standalone_numbers_are_not_discarded():
    from rag.generation.ollama import mask_publication_furniture

    text = "First, we evaluate\n42\nwith a caveat in this study."
    assert mask_publication_furniture(text, [(0, len(text), 1, 0)]) == text
