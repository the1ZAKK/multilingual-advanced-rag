import hashlib
from pathlib import PurePath

import pymupdf

from rag.config import Settings
from rag.ingestion.structure import VERSION, chunk_spans, reading_text, section_marks
from rag.ingestion.text import normalize
from rag.models import Chunk, DocumentInfo


class InvalidDocument(ValueError):
    pass


def parse_pdf(data: bytes, filename: str, settings: Settings) -> tuple[DocumentInfo, list[Chunk]]:
    name = PurePath(filename.replace("\\", "/")).name
    if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF-"):
        raise InvalidDocument("Only genuine PDF files with a .pdf extension are accepted")
    if len(data) > settings.max_file_mb * 1024 * 1024:
        raise InvalidDocument(f"PDF exceeds {settings.max_file_mb} MiB limit")
    document_id = hashlib.sha256(data).hexdigest()
    chunks, warnings = [], []
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise InvalidDocument("Encrypted PDF requires an unprotected copy")
            if not 0 < len(pdf) <= settings.max_pages:
                raise InvalidDocument(f"PDF must contain 1–{settings.max_pages} pages")
            pages = len(pdf)
            section = None
            for i, page in enumerate(pdf, 1):
                try:
                    text = reading_text(page)
                except Exception as exc:
                    raise InvalidDocument(
                        f"Text extraction failed on page {i}: {type(exc).__name__}"
                    ) from exc
                if not text.strip():
                    warnings.append(f"Page {i}: no extractable text; scanned pages require OCR")
                    continue
                bold = " ".join(
                    s["text"]
                    for b in page.get_text("dict")["blocks"]
                    for line in b.get("lines", [])
                    for s in line["spans"]
                    if s["flags"] & 16
                )
                sections = section_marks(text, section, bold or None)
                section = sections[-1][1]
                for start, end, chunk_section in chunk_spans(
                    text, settings.chunk_size, settings.chunk_overlap, sections
                ):
                    original = text[start:end]
                    if original.strip():
                        chunks.append(
                            Chunk(
                                chunk_id=f"{document_id}:p{i}:{start}-{end}",
                                document_id=document_id,
                                document_name=name,
                                page=i,
                                text=original,
                                normalized=normalize(original),
                                start=start,
                                end=end,
                                section=chunk_section,
                            )
                        )
    except InvalidDocument:
        raise
    except Exception as exc:
        raise InvalidDocument(f"Cannot parse PDF: {type(exc).__name__}") from exc
    if not chunks:
        raise InvalidDocument(
            "PDF has no extractable text. Run Arabic/English OCR before uploading."
        )
    return DocumentInfo(
        document_id=document_id,
        name=name,
        pages=pages,
        chunks=len(chunks),
        warnings=warnings,
        extraction_version=VERSION,
    ), chunks
