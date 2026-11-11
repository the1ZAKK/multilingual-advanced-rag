"""Source-derived evidence plans for explicit factual facets, never reference answers."""

import re

from rag.generation.numerical import (
    ROLES,
    clean_numbers,
    count_bindings,
    strategy_bindings,
    strategy_mentions,
)

PREDICATES = {
    "evaluate": r"\bevaluat\w*|\bassess\w*|\btest(?:ed|ing)\b|تقييم|قيم|اختبر",
    "generate": r"\bgenerat\w*|\bproduc\w*|توليد|ينتج|انتاج",
}

# Ordinary count facts are grounded in English before any Arabic translation.
# These are generic entity names, not document-specific phrases or quantities.
ENGLISH_COUNT_ENTITIES = {
    "model": r"(?:models?|[Ll][Ll][Mm]s?)",
    "provider": r"providers?",
    "task": r"tasks?",
    "question": r"questions?",
    "evaluation": r"(?:evaluations?|test[ \t]+cases?)",
}


def requested_counts(question):
    match = re.search(r"\bhow many\s+|كم\s+(?:عدد\s+)?", question, re.IGNORECASE)
    if not match:
        return set()
    tail = clean_numbers(question[match.end() :])
    return {role for role, pattern in ROLES.items() if re.search(rf"(?:{pattern})(?!\w)", tail)}


def plan_facet(sources, question, original_question=None):
    """Prefer source facts of the requested role/relation, keeping exact quote IDs.

    Counts and candidate comparisons come from the supplied quotations only.
    Ambiguous candidate values remain alternatives for the ordinary validators;
    this plan cannot certify support or a complete answer.
    """
    original_question = original_question or question
    normalized = clean_numbers(question)
    roles = requested_counts(question)
    deictic = bool(
        re.search(r"\b(?:distinguish|compare)\b|ميز|قارن", normalized)
        and re.search(r"\b(?:quantities|counts)\b|الكميتين|الكميات|العددين", normalized)
    )
    if deictic:
        roles = requested_counts(original_question)
    quotes = [(source, quote) for source in sources for quote in source["quotes"]]
    chosen = []
    contract = None
    if roles:
        candidates = [
            (source, quote, role, value)
            for source, quote in quotes
            for role, value in sorted(count_bindings(quote["quote"]))
            if role in roles
        ]
        predicates = {
            name
            for name, pattern in PREDICATES.items()
            if re.search(pattern, clean_numbers(original_question))
        }
        for role in roles:
            matching = [item for item in candidates if item[2] == role]
            preferred = [
                item
                for item in matching
                if any(
                    re.search(PREDICATES[name], clean_numbers(item[1]["quote"]))
                    for name in predicates
                )
            ]
            chosen.extend(preferred or matching)
        if chosen:
            contract = {
                "kind": "quantity-distinction" if deictic else "count",
                "requested_entities": sorted(roles),
                "source_candidates": [
                    {"entity": role, "quantity": value, "ref_id": quote["ref_id"]}
                    for _, quote, role, value in chosen
                ],
                "answer_style": (
                    "Distinguish every originally requested quantity, retaining each entity and count."
                    if deictic
                    else "Start with the quantity and its counted entity; use a short factual phrase. "
                    "Retain every necessary bound or condition. Do not add a new claim about "
                    "the benchmark's definition, or repeat unrequested counts."
                ),
            }
    elif re.search(r"\bcompare\b|قارن", normalized):
        requested = {role for _, _, role in strategy_mentions(question)}
        chosen = [
            (source, quote, None, None)
            for source, quote in quotes
            if strategy_bindings(quote["quote"])
            and (
                len(requested) != 2
                or {role for role, _ in strategy_bindings(quote["quote"])} == requested
            )
        ]
        if chosen:
            contract = {
                "kind": "condition-comparison",
                "answer_style": "State both conditions with their own accuracies and the reported ratio. "
                "Use one sentence; do not substitute other study counts or another comparison.",
            }
    elif re.search(r"\bwhat\s+score\b|(?:ما|وما)\s+(?:نتيجة|درجة|دقة)\b", normalized):
        chosen = [
            (source, quote, None, None)
            for source, quote in quotes
            if re.search(r"\d[\d.]*\s*%", clean_numbers(quote["quote"]))
            and re.search(r"\bscore\b|\baccura\w*|دقة|درجة|نتيجة", clean_numbers(quote["quote"]))
        ]
        if chosen:
            contract = {
                "kind": "score",
                "answer_style": "Give the requested score and its precise model/condition scope. "
                "Do not replace it with another score from the same quote.",
            }
    elif re.search(
        r"\b(?:what|which)\s+(?:(?:question|data|response|output|input)\s+)?"
        r"(?:languages?(?=\s+(?:and|is|are|was|were|does|do)\b|[?؟])|format\b)"
        r"|(?:ما|وما)\s+(?:لغة|صيغة)\b",
        normalized,
    ):
        language = bool(re.search(r"\blanguage\b|لغة", normalized))
        form = bool(re.search(r"\bformat\b|صيغة", normalized))
        for source, quote in quotes:
            text = re.sub(r"-\s*\n\s*", "", quote["quote"])
            language_fact = re.search(
                r"(?is:\b(?:questions?|data(?:set)?s?|corpus|content)\b.{0,160}\bin\s+)"
                r"[A-Z][A-Za-z]+",
                text,
            )
            format_fact = re.search(
                r"\b(?:multiple[- ]choice|open[- ]ended|free[- ](?:text|form)|yes/no|true/false)\b",
                text,
                re.IGNORECASE,
            )
            if (not language or language_fact) and (not form or format_fact):
                chosen.append((source, quote, None, None))
        contract = {
            "kind": "attribute",
            "answer_style": "Give all requested attributes as a concise source-supported phrase. "
            "Do not add a definition, dataset size, background, or other unrequested facts.",
        }
    if not chosen:
        return contract, sources
    selected = {quote["ref_id"] for _, quote, _, _ in chosen}
    # Repeated overlap quotes are alternatives, not a requirement to cite twice.
    seen = set()
    scoped = []
    for source in sources:
        matches = []
        for quote in source["quotes"]:
            if quote["ref_id"] in selected and quote["quote"] not in seen:
                matches.append(quote)
                seen.add(quote["quote"])
        if matches:
            scoped.append({**source, "quotes": matches})
    return contract, scoped


def count_text_pattern(contract):
    if not contract or contract["kind"] != "count":
        return None
    alternatives = set()
    for item in contract["source_candidates"]:
        values = {item["quantity"]}
        values |= {f"{int(value):,}" for value in values if value.isdecimal()}
        entity = ENGLISH_COUNT_ENTITIES.get(item.get("entity"))
        for value in values:
            prefix = re.escape(value) + r"[ \t]+"
            if entity:
                # Retain room for bounds and source qualifiers. Pair quantities
                # with their own entity, and keep this intermediate stage in
                # English even when the original user question is Arabic.
                prefix += r"(?:[A-Za-z][A-Za-z-]*[ \t]+){0,6}" + entity
                prefix += r"(?:[ \t]+.+|[.!?])?"
            else:
                prefix += r".+"
            alternatives.add(prefix)
    if not alternatives:
        return None
    # Ollama's grammar converter requires both anchors and does not support
    # regex shorthand escapes such as \s. Unsupported patterns are ignored.
    # This decoding hint never replaces numeric/NLI checks.
    # Permit the source's bounds; the unchanged semantic audit must check them.
    return (
        r"^(?:(?:over|more than|at least|at most|less than|under|up to)[ \t]+)?(?:"
        + "|".join(sorted(alternatives))
        + r")$"
    )
