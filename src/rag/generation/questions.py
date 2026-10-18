"""Conservative structural decomposition, without model-generated assumptions."""

import re


def question_parts(question: str) -> list[str]:
    """Split only explicit coordinated interrogatives, preserving original wording.

    Ordinary conjunctions (models and datasets, or how X and Y work) are not
    independent questions. This is a formatting contract, not a coverage judge.
    """
    parts = re.split(
        r"\s*,?\s+and\s+(?=(?:what|which|how|why|where|when|who|does|do|is|are)\b)"
        r"|[،,]?\s+و(?=(?:ما\s|ماذا\s|كيف\s|هل\s|لماذا\s|أي\s))"
        r"|(?<=[?؟])\s+(?=Distinguish\b|Explain\b|مي[ّ]?ز\b|اشرح\b)",
        question.strip(),
        flags=re.IGNORECASE,
    )
    return parts if 1 < len(parts) <= 4 and all(p.strip() for p in parts) else [question]


def flatten_parts(payload, parts):
    """Refuse omitted/extra slots before the normal grounding validators run."""
    keys = [f"part_{i}" for i in range(1, len(parts) + 1)]
    if not isinstance(payload, dict) or set(payload) != set(keys):
        raise ValueError("Multipart answer must answer every supplied question part")
    statements = []
    for key in keys:
        item = payload[key]
        if not isinstance(item, dict) or set(item) != {"text", "citations"}:
            raise ValueError("Invalid multipart answer slot")
        statements.append(item)
    return {"statements": statements}


def focus_sources(sources, question, limit=8):
    """Prioritize existing quote IDs for a part; never rewrite or create evidence.

    Inverse document frequency suppresses repeated paper names. No-overlap queries
    (including cross-language queries) retain the full original evidence.
    """
    from collections import Counter
    from math import log

    stop = {
        "what",
        "which",
        "how",
        "why",
        "where",
        "when",
        "who",
        "does",
        "do",
        "is",
        "are",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "to",
        "its",
        "it",
        "this",
        "that",
        "from",
        "with",
        "for",
        "by",
        "as",
    }

    def words(text):
        return {w.casefold() for w in re.findall(r"[^\W\d_]+", text) if w.casefold() not in stop}

    quotes = [(s, q, words(q["quote"])) for s in sources for q in s["quotes"]]
    counts = Counter(w for _, _, tokens in quotes for w in tokens)
    query = words(question)
    arabic_query = {w for w in query if re.search(r"[\u0600-\u06ff]", w)}
    if arabic_query and not any(arabic_query & tokens for _, _, tokens in quotes):
        # A shared Latin paper name is not a cross-language relevance signal.
        return sources
    ranked = sorted(
        [
            (sum(log(1 + len(quotes) / counts[w]) for w in sorted(query & tokens)), i, s, q)
            for i, (s, q, tokens) in enumerate(quotes)
        ],
        key=lambda r: (-r[0], r[1]),
    )
    if not ranked or ranked[0][0] == 0:
        return sources
    focused = {}
    for score, _, source, quote in ranked[:limit]:
        if score <= 0:
            continue
        label = source["label"]
        if label not in focused:
            focused[label] = {**source, "quotes": []}
        focused[label]["quotes"].append(quote)
    return list(focused.values())
