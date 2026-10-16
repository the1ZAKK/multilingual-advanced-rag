import pytest

from rag.ingestion.pdf import InvalidDocument, parse_pdf
from rag.ingestion.text import is_arabic, normalize, tokenize


def test_arabic_normalization():
    assert normalize("إِنَّ الـعَرَبِيَّةَ  لغةٌ") == "ان العربيه لغة".replace("العربيه", "العربية")
    assert tokenize("Arabic, 123; إِلى المَدْرَسة!") == ["arabic", "123", "الي", "المدرسة"]
    assert is_arabic("مرحبا") and not is_arabic("hello")


def test_english_unicode():
    assert normalize("ＦＡＩＳＳ \n Retrieval") == "faiss retrieval"
    assert tokenize("FAISS-BM25") == ["faiss", "bm25"]


def test_original_offsets_overlap(pdf_bytes, settings):
    info, chunks = parse_pdf(pdf_bytes("Semantic retrieval. " * 50), "../paper.pdf", settings)
    assert info.name == "paper.pdf" and len(chunks) > 1
    assert chunks[0].end - chunks[1].start == 20
    assert all(c.text and c.page == 1 and c.chunk_id.startswith(info.document_id) for c in chunks)
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_arabic_pdf(pdf_bytes, settings):
    _, chunks = parse_pdf(
        pdf_bytes("الاسترجاع الهجين يجمع البحث الدلالي والبحث بالكلمات", arabic=True),
        "arabic.pdf",
        settings,
    )
    assert any(is_arabic(c.text) for c in chunks)
    assert "الاسترجاع الهجين يجمع البحث الدلالي" in chunks[0].text


@pytest.mark.parametrize(
    "name,data", [("x.txt", b"%PDF-no"), ("x.pdf", b"not pdf"), ("x.pdf", b"%PDF-invalid")]
)
def test_invalid_pdf(settings, name, data):
    with pytest.raises(InvalidDocument):
        parse_pdf(data, name, settings)


def test_scanned_pdf_guidance(pdf_bytes, settings):
    with pytest.raises(InvalidDocument, match="OCR"):
        parse_pdf(pdf_bytes(blank=True), "scan.pdf", settings)


def test_page_limit(pdf_bytes, settings):
    import pymupdf

    pdf = pymupdf.open(stream=pdf_bytes(), filetype="pdf")
    pdf.new_page()
    with pytest.raises(InvalidDocument, match="pages"):
        parse_pdf(pdf.tobytes(), "two.pdf", settings.model_copy(update={"max_pages": 1}))


def test_size_limit(settings):
    with pytest.raises(InvalidDocument, match="limit"):
        parse_pdf(
            b"%PDF-" + b" " * (1024 * 1024),
            "huge.pdf",
            settings.model_copy(update={"max_file_mb": 1}),
        )


def test_empty_page_warning(pdf_bytes, settings):
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes(), filetype="pdf")
    doc.new_page()
    info, chunks = parse_pdf(doc.tobytes(), "mixed.pdf", settings)
    assert info.pages == 2 and "Page 2" in info.warnings[0]
    assert all(c.page == 1 for c in chunks)
