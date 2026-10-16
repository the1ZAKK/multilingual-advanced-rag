from time import perf_counter

from rag.config import Settings
from rag.ingestion.text import tokenize
from rag.models import Hit


class Reranker:
    def __init__(self, settings: Settings):
        self.settings, self.model = settings, None

    def rerank(self, query: str, hits: list[Hit]) -> tuple[list[Hit], float]:
        start = perf_counter()
        backend = self.settings.reranker_backend
        if backend == "none" or not hits:
            return hits, 0
        if backend == "lexical":
            q = set(tokenize(query))
            scores = [len(q & set(tokenize(h.chunk.text))) / max(len(q), 1) for h in hits]
        else:
            if self.model is None:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(
                    self.settings.reranker_model,
                    device=self.settings.device,
                    revision=self.settings.reranker_revision,
                    trust_remote_code=False,
                )
            scores = self.model.predict([(query, h.chunk.text) for h in hits], batch_size=8)
        output = []
        for hit, score in zip(hits, scores, strict=True):
            copy = hit.model_copy(deep=True)
            copy.rerank_score = float(score)
            copy.score = float(score)
            output.append(copy)
        return sorted(output, key=lambda h: (-h.score, h.chunk.chunk_id)), (
            perf_counter() - start
        ) * 1000
