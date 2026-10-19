from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load cache/client variables as well as typed RAG settings; injected values take precedence.
load_dotenv(override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")
    data_dir: Path = Path("data/index")
    embedding_backend: Literal["sentence-transformer", "hash"] = "sentence-transformer"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_revision: str | None = None
    device: str = "cpu"
    chunk_size: int = Field(900, ge=50, le=12000)
    chunk_overlap: int = Field(150, ge=0)
    max_file_mb: int = Field(25, ge=1, le=200)
    max_pages: int = Field(500, ge=1, le=5000)
    max_upload_files: int = Field(10, ge=1, le=100)
    retrieval_depth: int = Field(30, ge=1, le=200)
    bm25_casefold: bool = True
    rrf_k: int = Field(60, ge=1)
    reranker_backend: Literal["none", "lexical", "cross-encoder"] = "none"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_revision: str | None = None
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
    translation_model: str | None = Field(default=None, min_length=1)
    llm_timeout: float = Field(120, gt=0)
    llm_total_timeout: float = Field(150, gt=0, le=600)
    llm_context_tokens: int = Field(8192, ge=2048, le=32768)
    question_coverage_audit: bool = False  # Experimental: small judges can falsely reject.
    max_answer_statements: int = Field(4, ge=1, le=20)
    max_context_chars: int = Field(12000, ge=1000, le=64000)
    min_dense_score: float = Field(0.25, ge=-1, le=1)

    @model_validator(mode="after")
    def overlap_valid(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self
