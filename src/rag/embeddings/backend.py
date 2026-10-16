import hashlib
from typing import Protocol

import numpy as np

from rag.config import Settings
from rag.ingestion.text import tokenize


class Embedder(Protocol):
    key: str

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray: ...


class HashEmbedder:
    """Offline pipeline baseline. NOT multilingual semantic embeddings."""

    key = "hash-v1:384"

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                h = hashlib.blake2b(token.encode(), digest_size=8).digest()
                matrix[i, int.from_bytes(h[:4], "little") % 384] += 1 if h[4] % 2 else -1
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.key = f"st:{settings.embedding_model}:{settings.embedding_revision or 'default'}"
        self._model = None

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.settings.device,
                    revision=self.settings.embedding_revision,
                    trust_remote_code=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Embedding model unavailable. Install .[models], cache the model, "
                    "or explicitly select RAG_EMBEDDING_BACKEND=hash for offline tests. "
                    "Do not reuse semantic indexes with the hash backend."
                ) from exc
        kwargs = {}
        if query and "Qwen3-Embedding" in self.settings.embedding_model:
            kwargs["prompt"] = (
                "Instruct: Retrieve research passages relevant to the question.\nQuery: "
            )
        return np.asarray(
            self._model.encode(
                texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False, **kwargs
            ),
            dtype=np.float32,
        )


def make_embedder(settings: Settings) -> Embedder:
    return (
        HashEmbedder()
        if settings.embedding_backend == "hash"
        else SentenceTransformerEmbedder(settings)
    )
