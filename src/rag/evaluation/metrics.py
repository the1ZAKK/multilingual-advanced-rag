import math

from rag.ingestion.text import tokenize


def retrieval_metrics(
    ranked: list[str], relevant: dict[str, float], ks=(1, 5, 10)
) -> dict[str, float]:
    if not relevant or any(not math.isfinite(v) or v <= 0 or v > 10 for v in relevant.values()):
        raise ValueError("Provide finite relevance grades in (0, 10]")
    ranked = list(dict.fromkeys(ranked))
    output = {}
    for k in ks:
        if k < 1:
            raise ValueError("K must be positive")
        found = sum(cid in relevant for cid in ranked[:k])
        output[f"recall@{k}"] = found / len(relevant)
        output[f"precision@{k}"] = found / k
        dcg = sum(
            (2 ** relevant.get(cid, 0) - 1) / math.log2(i + 2) for i, cid in enumerate(ranked[:k])
        )
        ideal = sum(
            (2**grade - 1) / math.log2(i + 2)
            for i, grade in enumerate(sorted(relevant.values(), reverse=True)[:k])
        )
        output[f"ndcg@{k}"] = dcg / ideal if ideal else 0
    output["mrr"] = next((1 / i for i, cid in enumerate(ranked, 1) if cid in relevant), 0)
    return output


def token_f1(answer: str, reference: str) -> float:
    """Lexical correctness proxy, not a semantic correctness judgment."""
    from collections import Counter

    a, b = Counter(tokenize(answer)), Counter(tokenize(reference))
    common = sum((a & b).values())
    return 2 * common / (sum(a.values()) + sum(b.values())) if a or b else 1.0


def evidence_token_support(answer: str, contexts: list[str]) -> float:
    """Lexical grounding proxy; never call this a validated faithfulness score."""
    a = set(tokenize(answer))
    c = set(tokenize(" ".join(contexts)))
    return len(a & c) / len(a) if a else 0
