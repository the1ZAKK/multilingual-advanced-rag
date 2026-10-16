import pymupdf
import pytest

from rag.config import Settings
from rag.service import RAGService


@pytest.fixture
def pdf_bytes():
    def make(
        text="Hybrid retrieval combines dense vectors and BM25 keyword search.",
        blank=False,
        arabic=False,
    ):
        doc = pymupdf.open()
        page = doc.new_page()
        if not blank:
            if arabic:
                from pathlib import Path

                doc.close()
                return (Path(__file__).parent / "fixtures/arabic.pdf").read_bytes()
            else:
                page.insert_textbox(pymupdf.Rect(50, 50, 500, 700), text, fontsize=11)
        data = doc.tobytes()
        doc.close()
        return data

    return make


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "index",
        embedding_backend="hash",
        reranker_backend="lexical",
        chunk_size=100,
        chunk_overlap=20,
    )


@pytest.fixture
def service(settings):
    return RAGService(settings)
