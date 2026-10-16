from typing import Literal

from pydantic import BaseModel, Field

from rag.response_modes import ResponseMode


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    text: str
    normalized: str
    start: int
    end: int
    section: str | None = None


class DocumentInfo(BaseModel):
    document_id: str
    name: str
    pages: int
    chunks: int
    warnings: list[str] = Field(default_factory=list)
    duplicate: bool = False
    extraction_version: str = "legacy-fixed-v1"


class Hit(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    # Ephemeral geometry-bound source data; never replaces indexed native quotes.
    tables: list[dict] = Field(default_factory=list)
    # Retained PDF provenance, ephemeral and excluded from API payloads/indexes.
    native_page_text: str | None = Field(default=None, exclude=True)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["dense", "bm25", "hybrid", "hybrid-rerank"] = "hybrid"
    top_k: int = Field(5, ge=1, le=50)


class SearchResponse(BaseModel):
    hits: list[Hit]
    latency_ms: float
    rerank_latency_ms: float = 0
    embedding_backend: str
    reranker_backend: str


class ChatRequest(SearchRequest):
    language: Literal["auto", "ar", "en"] = "auto"
    use_llm: bool = True


class Citation(BaseModel):
    label: str
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    quote: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    evidence: list[Hit]
    generation_mode: ResponseMode
    warning: str | None = None
    latency_ms: float
