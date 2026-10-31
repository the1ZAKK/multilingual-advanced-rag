"""Necessary numerical relation checks; source text and semantic audits stay authoritative.

No document identities or expected answer values live here. The vocabulary is
measurement roles and common experimental conditions, shared across papers.
Unrecognized relations still need the ordinary model-based semantic audit.
"""

import re
import unicodedata

from rag.ingestion.text import normalize

NUMBER = r"\d+(?:\.\d+)?"
ROLES = {
    "model": r"models?|llms?|(?:ال)?نماذج|(?:ال)?نموذج(?:ا)?",
    "provider": r"providers?|(?:ال)?مقدمي(?:ها|ن)?|مزود(?:ي|ين|ا)?(?:\s+الخدمات)?",
    "task": r"tasks?|(?:ال)?مهام|(?:ال)?مهمة",
    "question": r"questions?|(?:ال)?اسئلة|(?:ال)?سؤال(?:ا)?",
    "evaluation": r"(?:individual\s+)?(?:test\s+cases?|evaluations?)|"
    r"(?:ال)?حالات\s+(?:ال)?اختبار|حالة\s+اختبار|تقييم(?:ات|ا)?",
}
STRATEGIES = {
    "few-cot": r"few[- ]?shot\s+(?:chain[- ]?of[- ]?thought|cot)|"
    r"سلسلة\s+(?:ال)?تفكير\s+بامثلة\s+قليلة",
    "zero-cot": r"zero[- ]?shot\s+(?:chain[- ]?of[- ]?thought|cot)",
    "few": r"few[- ]?shot|بامثلة\s+قليلة",
    "zero": r"zero[- ]?shot|(?:دون|بدون)\s+امثلة",
}


def clean_numbers(text):
    text = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    text = text.replace("\u066b", ".").replace("\u066a", "%")
    text = re.sub(
        r"(?<!\d)\d{1,3}(?:[,\u066c]\d{3})+(?!\d)",
        lambda m: re.sub(r"[,\u066c]", "", m[0]),
        text,
    )
    text = re.sub(r"(?<=[\u0600-\u06ff])(?=\d)", " ", text)
    return normalize(re.sub(r"-\s*\n\s*", "", text))


def atoms(text):
    return set(re.findall(rf"(?<![\w.]){NUMBER}(?!\d|\.\d)(?=[x×]\b|\W|$)", clean_numbers(text)))


def count_bindings(text):
    text = clean_numbers(text)
    result = set()
    for role, expression in ROLES.items():
        for match in re.finditer(
            rf"(?<![\w.])({NUMBER})\s+(?:(?:evaluated|distinct|individual|multiple[- ]?choice|"
            rf"number[- ]reading|large language|الفردية|مختلفة)\s+){{0,3}}(?:{expression})(?![\w-])",
            text,
        ):
            result.add((role, match[1]))
    return result


def strategy_bindings(text):
    """Bind two explicitly named prompting conditions to their percentage pair.

    Handles both 'A 90% versus B 30%' and 'A versus B (90% vs 30%)'.
    Ambiguous lists/ranges return no bindings, never a guessed association.
    """
    text = clean_numbers(text)
    matches = []
    for strategy, expression in STRATEGIES.items():
        for match in re.finditer(expression, text):
            if not any(match.start() < end and start < match.end() for start, end, _ in matches):
                matches.append((match.start(), match.end(), strategy))
    matches.sort()
    percentages = list(re.finditer(rf"({NUMBER})\s*%", text))
    if len(matches) != 2 or len(percentages) != 2 or matches[0][2] == matches[1][2]:
        return set()
    if matches[0][0] < percentages[0].start() and (
        matches[1][0] < percentages[1].start()
        or re.search(r"\b(?:vs|versus|compared)\b|مقابل", text)
    ):
        return {(matches[i][2], percentages[i][1]) for i in (0, 1)}
    return set()


def validate_numeric_support(claim, premises):
    """Reject swapped roles and unrelated numerical citations before model judging."""
    claimed = atoms(claim)
    supplied = set().union(*(atoms(p) for p in premises))
    if not claimed <= supplied:
        raise ValueError("Numerical grounding failed: unsupported number")
    for index, premise in enumerate(premises, 1):
        values = atoms(premise)
        if claimed and values and not values & claimed:
            raise ValueError(
                f"Irrelevant numerical citation {index}: supports a different comparison"
            )
    for extractor, label in ((count_bindings, "counted entity"), (strategy_bindings, "condition")):
        expected = set().union(*(extractor(p) for p in premises))
        for role, value in extractor(claim):
            alternatives = {v for r, v in expected if r == role}
            if alternatives and value not in alternatives:
                raise ValueError(f"Numerical grounding failed: swapped {label} {role}")


def validate_count_slot(question, answer, premises):
    """A recognized count facet cannot silently be replaced by another count.

    Only source-expressed quantities activate this check. A question with no
    source-bound count still requires the ordinary evidence/semantic checks.
    """
    if not re.search(r"\bhow many\b|كم\s+(?:عدد\s+)?", question, re.IGNORECASE):
        return
    source = set().union(*(count_bindings(p) for p in premises))
    normalized = clean_numbers(question)
    requested = {
        role
        for role, expression in ROLES.items()
        if re.search(rf"(?:{expression})(?!\w)", normalized)
    }
    answered = count_bindings(answer)
    for role in requested:
        values = {value for r, value in source if r == role}
        if values and not any((role, value) in answered for value in values):
            raise ValueError(f"Incomplete quantitative answer: missing requested {role} count")


def validate_comparison_slot(question, answer, premises):
    if not re.search(r"\bcompare\b|قارن", question, re.IGNORECASE):
        return
    pairs = [p for p in premises if strategy_bindings(p)]
    expected = set().union(*(strategy_bindings(p) for p in pairs))
    # Conflicting source values are ambiguous, not a license to pick an answer.
    if not expected or any(len({v for r, v in expected if r == role}) > 1 for role, _ in expected):
        return
    if not expected <= strategy_bindings(answer):
        raise ValueError(
            "Incomplete quantitative comparison: preserve condition-to-accuracy mapping"
        )
    ratios = set().union(
        *(set(re.findall(rf"({NUMBER})\s*[x×]\b", clean_numbers(p))) for p in pairs)
    )
    if len(ratios) == 1 and not ratios <= atoms(answer):
        raise ValueError("Incomplete quantitative comparison: missing source-reported ratio")
