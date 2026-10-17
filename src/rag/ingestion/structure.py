"""Reading-order text offsets and bounded sentence/section-aware windows."""

import re

from rag.ingestion.text import normalize

VERSION = "columns-sentences-v3"

_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "واحدة": 1,
    "اثنتان": 2,
    "اثنين": 2,
    "ثلاث": 3,
    "ثلاثة": 3,
    "اربع": 4,
    "اربعة": 4,
    "خمس": 5,
    "خمسة": 5,
    "ست": 6,
    "ستة": 6,
    "سبع": 7,
    "سبعة": 7,
    "ثمان": 8,
    "ثمانية": 8,
    "تسع": 9,
    "تسعة": 9,
    "عشر": 10,
    "عشرة": 10,
}


def contribution_declarations(text: str) -> list[tuple[int, int, int | None]]:
    """Explicit author-list declarations, never a findings/methodology list.

    A count is taken from the declaration, not supplied by a question. None
    means the source announces a list without specifying its count.
    """
    declarations = []
    pattern = (
        r"(?im)(?:\b(?:our|the|this\s+paper['’]s)\s+(?:(?:main|key|primary|novel)\s+)?contributions\b"
        r"|\bwe\s+(?:make|highlight|summarize|summarise|present)\b[^\n:]{0,55}\bcontributions\b)[^.!?]{0,220}?"
        r"(?:\b(?:fold|following|follows|including|include|consist\s+of)\b|:)"
    )
    for match in re.finditer(pattern, text):
        declaration = normalize(match[0])
        count_match = re.search(
            r"\b("
            + "|".join(_COUNT_WORDS)
            + r"|\d{1,2})\s*(?:[- ]?fold|(?:(?:main|key|primary)\s+)?contributions?\b)",
            declaration,
        )
        count = None
        if count_match:
            word = count_match[1]
            count = _COUNT_WORDS.get(word, int(word) if word.isdecimal() else None)
        declarations.append((match.start(), match.end(), count))
    # Preserve original offsets even though Arabic normalization removes marks.
    for match in re.finditer(r"(?m)^[^\n]{0,220}", text):
        declaration = match[0].split(":", 1)[0]
        line = normalize(declaration)
        if re.search(r"(?:مساهمات|اسهامات)(?:نا)?\b", line) and re.search(
            r"كما يلي|النحو التالي|(?:ثلاث|اربع|خمس|ست|سبع|ثمان|تسع|عشر|[\d])", line
        ):
            count_match = re.search(r"\b(" + "|".join(_COUNT_WORDS) + r"|\d{1,2})\b", line)
            word = count_match[1] if count_match else ""
            count = _COUNT_WORDS.get(word, int(word) if word.isdecimal() else None)
            declarations.append((match.start(), match.start() + len(declaration), count))
    return sorted(set(declarations))


def contribution_markers(text: str, start: int = 0) -> list[tuple[int, int, int | None, str]]:
    """List markers with original offsets and source ordinal, where declared."""
    markers = []
    for match in re.finditer(
        r"(?<!\w)(?:\((\d{1,2})\)|(\d{1,2})[)])\s+|(?m:^\s*(\d{1,2})\.\s+)", text[start:]
    ):
        number = int(next(group for group in match.groups() if group))
        markers.append((start + match.start(), start + match.end(), number, "number"))
    romans = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")
    for number, roman in enumerate(romans, 1):
        for match in re.finditer(rf"(?<!\w)(?:\({roman}\)|{roman}\))\s+", text[start:], re.I):
            markers.append((start + match.start(), start + match.end(), number, "roman"))
    ordinals = (
        "First",
        "Second",
        "Third",
        "Fourth",
        "Fifth",
        "Sixth",
        "Seventh",
        "Eighth",
        "Ninth",
        "Tenth",
    )
    for number, ordinal in enumerate(ordinals, 1):
        for match in re.finditer(rf"\b{ordinal}\s*[,،:]\s*", text[start:], re.IGNORECASE):
            markers.append((start + match.start(), start + match.end(), number, "ordinal"))
    arabic_ordinals = (
        "اولا",
        "ثانيا",
        "ثالثا",
        "رابعا",
        "خامسا",
        "سادسا",
        "سابعا",
        "ثامنا",
        "تاسعا",
        "عاشرا",
    )
    marks = r"[\u0610-\u061a\u064b-\u065f\u0670]*"
    for number, word in enumerate(arabic_ordinals, 1):
        pattern = "".join((r"[اأإآ]" if char == "ا" else char) + marks for char in word)
        for match in re.finditer(rf"(?<!\w){pattern}(?:\s*[,،:]\s*|\s+)", text[start:]):
            markers.append((start + match.start(), start + match.end(), number, "ordinal"))
    for match in re.finditer(r"(?m)^[ \t]*[-*•▪–][ \t]+", text[start:]):
        markers.append((start + match.start(), start + match.end(), None, "bullet"))
    return sorted(markers)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    # Keep source characters unchanged, including PDF line-end hyphenation.
    ends = [m.end() for m in re.finditer(r"[.!?؟](?=\s|$)", text)]
    spans, start = [], 0
    for end in ends + [len(text)]:
        while start < end and text[start].isspace():
            start += 1
        if end > start:
            spans.append((start, end))
        start = end
    return spans


def reading_text(page) -> str:
    native = page.get_text("text", sort=False)
    arabic = len(re.findall(r"[\u0600-\u06ff]", native))
    latin = len(re.findall(r"[A-Za-z]", native))
    if arabic > latin:
        return native  # Never geometrically reverse RTL glyph order.
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    middle = page.rect.width / 2
    left = [b for b in blocks if b[2] <= middle and b[2] - b[0] > page.rect.width * 0.2]
    right = [b for b in blocks if b[0] >= middle and b[2] - b[0] > page.rect.width * 0.2]
    left_column = len(left) >= 2 or any(b[3] - b[1] > 35 for b in left)
    right_column = len(right) >= 2 or any(b[3] - b[1] > 35 for b in right)
    if not (left_column and right_column):
        return page.get_text("text", sort=True)
    # Full-width titles/tables divide the page into bands. Within a band read down
    # the entire left column, then down the right, never across matching baselines.
    spanning = sorted([b for b in blocks if b[0] < middle < b[2]], key=lambda b: b[1])
    remaining = [b for b in blocks if not (b[0] < middle < b[2])]
    ordered = []
    for span in spanning:
        band = [b for b in remaining if b[1] < span[1]]
        ordered.extend(sorted(band, key=lambda b: (b[0] >= middle, b[1], b[0])))
        remaining = [b for b in remaining if b not in band]
        ordered.append(span)
    ordered.extend(sorted(remaining, key=lambda b: (b[0] >= middle, b[1], b[0])))
    return "\n".join(b[4] for b in ordered)


def section_marks(
    text: str, inherited: str | None, bold_text: str | None = None
) -> list[tuple[int, str | None]]:
    marks = [(0, inherited)]
    pattern = r"(?m)^(?:\d{1,2}(?:\.\d+)*[ \t]*\n|\d{1,2}(?:\.\d+)*[ \t]+|[A-Z](?:\.\d+)*[ \t\n]+)([A-Z][^\n.!?]{2,65})\n"
    for match in re.finditer(pattern, text):
        if bold_text is None or match.group(1).strip() in bold_text:
            marks.append((match.start(), match.group(0).strip().replace("\n", " ")))
    for match in re.finditer(r"(?m)^(Abstract|References|Appendix)\s*\n", text):
        marks.append((match.start(), match.group(1)))
    for start, _, _ in contribution_declarations(text):
        current = max((mark for mark in marks if mark[0] <= start), key=lambda m: m[0])[1]
        if not current or not re.match(r"^(?:Appendix|References|[A-Z](?:\.\d+)*\b)", current):
            marks.append((start, "Contributions"))
    return sorted(marks, key=lambda m: m[0])


def chunk_spans(
    text: str, size: int, overlap: int, sections: list[tuple[int, str | None]]
) -> list[tuple[int, int, str | None]]:
    sentences = [end for _, end in sentence_spans(text)]
    paragraphs = [m.end() for m in re.finditer(r"\n\s*\n", text)]
    boundaries = sorted(set(sentences + paragraphs + [len(text)]))
    result = []
    for index, (section_start, section) in enumerate(sections):
        stop = sections[index + 1][0] if index + 1 < len(sections) else len(text)
        start = section_start
        while start < stop:
            limit = min(start + size, stop)
            choices = [p for p in boundaries if start + size // 2 <= p <= limit]
            end = limit if limit == stop else max(choices, default=limit)
            if text[start:end].strip():
                result.append((start, end, section))
            if end == stop:
                break
            # Begin overlap at a sentence boundary, with a hard size bound on
            # pathological single sentences. This guarantees forward progress.
            candidates = [p for p in boundaries if end - overlap <= p < end]
            start = max(start + 1, min(candidates, default=end - overlap))
    return result
