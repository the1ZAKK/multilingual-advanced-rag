import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from rag.ingestion.text import tokenize
from rag.models import Chunk, Hit


class SearchIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, *, bm25_casefold: bool = True):
        self.chunks = chunks
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.faiss = faiss.IndexFlatIP(vectors.shape[1])
        if chunks:
            self.faiss.add(self.vectors)
        # Positive IDF variant avoids negative BM25 on tiny corpora with ubiquitous terms.
        self.bm25_casefold = bm25_casefold
        self.tokens = [tokenize(c.text, casefold=bm25_casefold) for c in chunks]
        self.bm25 = BM25Okapi(self.tokens) if chunks and any(self.tokens) else None
        if self.bm25:
            import math

            n = len(chunks)
            for term in self.bm25.idf:
                df = sum(term in words for words in self.bm25.doc_freqs)
                self.bm25.idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def dense(self, vector: np.ndarray, k: int, document_ids: set[str] | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        params = None
        if document_ids is not None:
            positions = np.asarray(
                [i for i, chunk in enumerate(self.chunks) if chunk.document_id in document_ids],
                dtype=np.int64,
            )
            if not len(positions):
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(positions))
        scores, indices = self.faiss.search(
            np.ascontiguousarray(vector, dtype=np.float32), min(k, len(self.chunks)), params=params
        )
        return [
            Hit(chunk=self.chunks[int(i)], score=float(s), dense_score=float(s))
            for s, i in zip(scores[0], indices[0])
            if i >= 0
        ]

    def sparse(self, query: str, k: int, document_ids: set[str] | None = None) -> list[Hit]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query, casefold=self.bm25_casefold))
        positions = [
            i
            for i, c in enumerate(self.chunks)
            if document_ids is None or c.document_id in document_ids
        ]
        order = sorted(positions, key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]), bm25_score=float(scores[i]))
            for i in order[:k]
            if scores[i] > 0
        ]


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    if k < 1:
        raise ValueError("RRF k must be positive")
    combined: dict[str, Hit] = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            cid = hit.chunk.chunk_id
            if cid in seen:
                continue
            seen.add(cid)
            if cid not in combined:
                combined[cid] = Hit(chunk=hit.chunk, score=0, rrf_score=0)
            result = combined[cid]
            result.score += 1 / (k + rank)
            result.rrf_score = result.score
            if hit.dense_score is not None:
                result.dense_score = hit.dense_score
            if hit.bm25_score is not None:
                result.bm25_score = hit.bm25_score
    return sorted(combined.values(), key=lambda h: (-h.score, h.chunk.chunk_id))
