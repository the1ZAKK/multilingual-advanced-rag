import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rag.embeddings.backend import HashEmbedder, SentenceTransformerEmbedder


def test_hash_stable_normalized():
    embedder = HashEmbedder()
    first = embedder.encode(["Arabic العربية retrieval"])
    second = embedder.encode(["Arabic العربية retrieval"])
    assert np.array_equal(first, second)
    assert np.linalg.norm(first[0]) == pytest.approx(1, abs=1e-6)


def test_sentence_transformer_query_instruction(settings, monkeypatch):
    calls = []

    class Model:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    config = settings.model_copy(update={"embedding_model": "Qwen/Qwen3-Embedding-0.6B"})
    embedder = SentenceTransformerEmbedder(config)
    embedder.encode(["document"])
    output = embedder.encode(["question"], query=True)
    assert calls[0][1]["trust_remote_code"] is False
    assert calls[0][1]["device"] == "cpu"
    assert "prompt" not in calls[1][1]
    assert calls[2][1]["prompt"].endswith("Query: ")
    assert calls[2][1]["normalize_embeddings"] is True
    assert output.shape == (1, 4)
