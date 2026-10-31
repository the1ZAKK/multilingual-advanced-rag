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
        r"|(?<=[?؟])\s+(?=Distinguish\b|Explain\b|Compare\b|مي[ّ]?ز\b|اشرح\b|قارن\b)",
        question.strip(),
        flags=re.IGNORECASE,
    )
    parts = [facet for part in parts for facet in count_parts(part)]
    # The caller enforces its statement budget. Never hide excess requirements
    # by collapsing them back into a single slot.
    return parts if all(p.strip() for p in parts) else [question]


def count_parts(question: str) -> list[str]:
    """Expand explicit coordinated counts without inventing expected values.

    This is syntax, not a paper-specific checklist or a semantic parser. It also
    works for sensors, sites, patients, etc. Non-enumerations stay untouched.
    Original wording remains available as original_question to the model.
    """
    english = re.search(r"\bhow many\s+", question, re.IGNORECASE)
    arabic = re.search(r"كم\s+(?:عدد\s+)?", question)
    match = english or arabic
    if match is None:
        return [question]
    prefix, tail = question[: match.start()], question[match.end() :]
    if english:
        body = re.split(r"\b(?:are|were|is|was|have|has|do|does|did)\b|[?؟]", tail, 1)[0]
        nouns = re.split(r",\s*(?:and\s+)?|\s+and\s+", body.strip(), flags=re.IGNORECASE)
    else:
        body = re.split(r"\s+(?:التي|الذي|الذين|تم|جرى)\b|[?؟،]", tail, 1)[0]
        nouns = re.split(r"\s+و(?=\S)|،\s*(?:و)?", body.strip())
    nouns = [noun.strip(" ,،") for noun in nouns if noun.strip(" ,،")]
    # Avoid treating a long predicate or a mechanism as a list of count nouns.
    if len(nouns) < 2 or any(len(noun.split()) > 6 for noun in nouns):
        return [question]
    return [f"{prefix}how many {noun}?" if english else f"{prefix}كم عدد {noun}؟" for noun in nouns]


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
