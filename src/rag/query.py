"""Question intent and answer language checks shared by retrieval and generation.

These helpers describe a user's request; they do not establish claim entailment.
Source quotations and identifiers remain unchanged and are validated separately.
"""

import re
import unicodedata
from collections.abc import Iterable

from rag.ingestion.text import normalize


def _arabic_letters(text: str) -> int:
    return sum(
        unicodedata.category(char).startswith("L")
        and ("\u0600" <= char <= "\u06ff" or "\u0750" <= char <= "\u077f")
        for char in unicodedata.normalize("NFKC", text)
    )


_CONTRIBUTION_AR = re.compile(
    r"(?<!\w)[وفبل]{0,2}(?:ال)?(?:مساهم(?:ات|ة|ه)|اسهام(?:ات)?)"
    r"(?:ها|هم|هن|ه|نا|كما|كم|ك)?(?!\w)"
)
_DOCUMENT_AR = r"(?:الورقة|البحث|الدراسة|العمل|المؤلفين|الباحثين|الكتاب)"


def contribution_question(question: str) -> bool:
    """Recognize authored contributions, including Arabic inflection/clitics.

    A definition of the word and a physical variable's contribution to a score
    are different requests. No paper title or fixed contribution count is used.
    """
    text = normalize(question)
    if re.search(
        r"\b(?:define|meaning of|word|term)\s+[\"'“”]?contributions?\b|"
        r"(?:معني|تعريف)\s+(?:كلمة\s+|مصطلح\s+)?(?:ال)?(?:مساهمة|اسهام)",
        text,
    ):
        return False
    if _CONTRIBUTION_AR.search(text):
        return True
    if re.search(rf"\b{_DOCUMENT_AR}\b.*\b(?:يساهم|تساهم|يسهم|تسهم)\b", text):
        return True
    if re.search(r"\bcontributions\b", text):
        return True
    if re.search(r"\bcontribution\b", text):
        return bool(
            re.search(r"\b(?:paper|authors?|work|study|research|main|novel|original)\b", text)
        )
    return bool(re.search(r"\b(?:paper|authors?|work|study)\b.*\bcontribut(?:e|es|ed)\b", text))


def asks_problem(question: str) -> bool:
    text = normalize(question)
    return bool(
        re.search(r"\b(?:problem|gap|challenge|limitation)\b|\baddress(?:es|ed)?\b", text)
        or re.search(
            r"(?<!\w)[وفبل]{0,2}(?:ال)?(?:مشكلة|مشكلات|مشاكل|فجوة|تحدي|تحديات)(?!\w)|"
            r"\b(?:تعالج|يعالج|تتناول|يتناول)\b",
            text,
        )
    )


def explicit_author_list_question(question: str) -> bool:
    """An exhaustive declared author list, distinct from a thematic summary."""
    if not contribution_question(question):
        return False
    text = normalize(question)
    return bool(
        re.search(
            r"\b(?:explicit(?:ly)?|author[- ]stated|all|every|actual count|complete list)\b|"
            r"\b(?:list|enumerate)\b.{0,45}\b(?:one|two|three|four|five|six|\d+)\b|"
            r"\b(?:صراحة|صريح|صريحة|صريحا|جميع|كل|كامل|كاملة|عددها|العدد الفعلي)\b|"
            r"(?:يذكر|ذكر|يسرد|سرد).{0,25}(?:المؤلف|الباحث)",
            text,
        )
    )


def models_results_question(question: str) -> bool:
    """Identify requests for evaluated models and quantitative performance."""
    text = normalize(question)
    if (
        verification_question(question)
        or contribution_question(question)
        or re.search(
            r"\b(?:define|definition|meaning|what (?:is|does).{0,25}(?:mean|metric))\b|"
            r"(?:ما معني|تعريف|ما المقصود|كيف (?:تحسب|يحسب))",
            text,
        )
    ):
        return False
    if re.search(r"\btable\s+\d+\b|\bالجدول\s+\d+\b", text) and re.search(
        r"\b(?:scores?|results?|values?|f1|accuracy|compare|comparison|numbers?)\b|"
        r"(?:درجات|نتائج|قيم|دقة|مقارنة)",
        text,
    ):
        return True
    # Quantities in an experimental protocol (temperature, examples, decoding)
    # are not automatically comparative model-performance tables.
    protocol = re.search(
        r"\b(?:temperature|demonstrations?|api|decoding|prompting protocol)\b|"
        r"درجة الحرارة|درجات الحرارة|واجهة برمجة|امثلة توضيحية|امثلة.{0,12}تلقين",
        text,
    )
    performance = re.search(
        r"\b(?:accuracy|f1|pass@|scores?|compare|comparison|outperform)\b|"
        r"دقة|مقارنة|تفوق|درجات الاداء",
        text,
    )
    if protocol and not performance:
        return False
    if asks_problem(question) or re.search(
        r"^(?:according to .{0,100},\s*)?(?:how|why)\b|"
        r"\b(?:methodology|pipeline|adaptation|metric definition)\b|"
        r"^(?:كيف|لماذا)\b|(?:طريقة حساب|تعريف المقياس)",
        text,
    ):
        return False
    models = re.search(r"\b(?:models?|llms?|systems?)\b", text) or re.search(
        r"(?<!\w)[وفبل]{0,2}(?:ال)?(?:نماذج|نموذج|انظمة|نظم)(?!\w)", text
    )
    results = re.search(
        r"\b(?:results?|scores?|performance|accuracy|metrics?|numerical|quantitative)\b",
        text,
    ) or re.search(
        r"(?<!\w)[وفبل]{0,2}(?:ال)?"
        r"(?:نتائج|نتيجة|اداء|دقة|درجات|درجة|مقاييس|ارقام|رقمية|كمية)"
        r"(?:ها|هم|هن|ه|نا|كما|كم|ك)?(?!\w)",
        text,
    )
    return bool(models and results)


def verification_question(question: str) -> bool:
    """Recognize a claim-verification request, independently of its subject."""
    text = normalize(question).lstrip(" ¿?؟")
    return bool(
        re.match(r"(?:does|do|did|is|are|was|were|has|have|had|can|could|will|would)\b", text)
        or re.search(r"\b(?:verify|check|confirm|establish)\b.*\b(?:whether|if|that)\b", text)
        or re.match(r"هل\b|ا(?:يمكن|ليس|لا|يقدم|تقدم|يقترح|تقترح|يدرب|تدرب)\b", text)
        or re.search(r"\b(?:تحقق|تاكد|اثبت)\b.*(?:ما اذا|ان|هل)\b", text)
    )


def claim_target(question: str) -> str | None:
    """Extract an explicitly queried introduction/training object when clear.

    This is a retrieval hint, never a positive-claim or absence validator. Complex
    passive constructions and multiple alternatives return None rather than
    guessing the relation being asked about.
    """
    if not verification_question(question):
        return None
    text = normalize(question, casefold=False).strip()
    english = re.search(
        r"\b(?:introduce[sd]?|propose[sd]?|train(?:s|ed)?|develop(?:s|ed)?|"
        r"present(?:s|ed)?|build|built|provide[sd]?|include[sd]?)\s+(.+)",
        text,
        re.IGNORECASE,
    )
    arabic = re.search(
        r"\b(?:يقدم|تقدم|قدمت?|يقترح|تقترح|اقترحت?|يدرب|تدرب|دربت?|"
        r"يطور|تطور|طورت?|يطرح|تطرح|طرحت?|يتضمن|تتضمن|يستحدث|تستحدث)\s+(.+)",
        text,
    )
    match = english or arabic
    if match is None:
        return None
    target = match[1]
    # Relation/method qualifiers stay in the full question for semantic audit;
    # this object-only retrieval hint need not put every clause in one quote.
    target = re.split(
        r"\busing\b|\b(?:باستخدام|باستعمال)\b", target, maxsplit=1, flags=re.IGNORECASE
    )[0]
    if arabic and not english:
        target = re.sub(r"^(?:هذه\s+)?(?:الورقة|ورقة|البحث|الدراسة|العمل)\s+", "", target)
        target = re.sub(r"^[A-Za-z0-9][\w.+/-]*\s+", "", target)
    target = re.split(
        r"[?؟.;\n]|\b(?:cite|citation|supporting|according to|in the paper|as claimed)\b|"
        r"\b(?:اذكر|استشهد|استشهاد|مع ذكر|وفقا)\b",
        target,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    target = re.sub(r"^(?:an?|the)\s+", "", target, flags=re.IGNORECASE).strip(" ,،:؛")
    if not target or re.search(r"\b(?:or|whether)\b|\bاو\b", target, re.IGNORECASE):
        return None
    return target


def numbered_list_question(question: str) -> bool:
    """Number only when numbering/enumeration is requested, not all answers."""
    text = normalize(question)
    return bool(
        re.search(
            r"\b(?:numbered\s+list|number\s+(?:them|the\s+\w+)|enumerate)\b|"
            r"\blist\b.{0,25}\b(?:all\s+)?(?:three|four|five|\d+)\b",
            text,
        )
        or re.search(r"\b(?:مرقمة|مرقمه|رقمها|رقمهم)\b|\bرقم\s+(?:ال|هذه)", text)
    )


def _prose(text: str, identifiers: Iterable[str] = ()) -> str:
    for name in identifiers:
        if name:
            text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", " ", text)
    text = re.sub(r"https?://\S+|\[[^\[\]]*\]|`[^`]+`", " ", text)
    text = re.sub(r"\((?:pages?|الصفحات?|صفحات?)\s+[\d\s,،–-]+\)", " ", text)
    # Model IDs, acronyms, CamelCase names and snake_case identifiers are allowed
    # in either language. Ordinary title-cased words remain prose: accepting all
    # capitals/title case would permit an entire English answer to evade checks.
    prose_spans = [
        match.span() for match in re.finditer(r"\b(?:[A-Z]{2,}\s+){2,}[A-Z]{2,}\b", text)
    ]

    def identifier(match):
        if any(start <= match.start() < end for start, end in prose_spans):
            return match[0]
        return " "

    text = re.sub(
        r"(?<![A-Za-z0-9_])(?:[A-Z]{2,}[A-Z0-9+_-]*|"
        r"[A-Za-z][A-Za-z0-9_./+-]*\d[A-Za-z0-9_./+-]*|"
        r"[A-Z][a-z]+(?:[A-Z][a-zA-Z]*)+[+]?|"
        r"[A-Za-z]+_[A-Za-z_]+)(?![A-Za-z0-9_])",
        identifier,
        text,
    )
    return text


def resolve_language(question: str, requested: str = "auto") -> str:
    if requested not in {"auto", "ar", "en"}:
        raise ValueError("Requested language must be auto, ar or en")
    if requested != "auto":
        return requested
    text = _prose(question)
    arabic = _arabic_letters(text)
    latin = len(re.findall(r"[A-Za-z]", text))
    if arabic and (re.match(r"\W*[\u0600-\u06ff]", text) or arabic >= max(2, latin * 0.6)):
        return "ar"
    return "en"


def validate_output_language(text: str, language: str, *, identifiers: Iterable[str] = ()) -> None:
    """Reject prose in the wrong language; keep citations/proper IDs separate.

    Script checks enforce language following, not translation faithfulness or
    claim entailment. Those require separate semantic checks. An answer cannot
    pass by adding a short Arabic heading before an English factual paragraph.
    """
    if language not in {"ar", "en"}:
        raise ValueError("Output language must be ar or en")
    prose = _prose(text, identifiers)
    arabic = _arabic_letters(prose)
    latin = len(re.findall(r"[A-Za-z]", prose))
    if language == "ar":
        if not arabic or (latin >= 12 and latin > arabic * 0.5):
            raise ValueError("Output language mismatch: expected Arabic answer prose")
        # Check each actual line/paragraph too, so a long Arabic introduction
        # cannot conceal a separate untranslated English contribution item.
        for line in prose.splitlines():
            words = re.findall(r"[A-Za-z]+", line)
            if len(words) >= 4 and len("".join(words)) > _arabic_letters(line) * 0.5:
                raise ValueError("Output language mismatch: untranslated English answer item")
    elif arabic >= 4 and arabic > latin * 0.1:
        raise ValueError("Output language mismatch: expected English answer prose")


def overview_question(question: str) -> bool:
    """Broad paper/dataset overview, not a score or a particular method detail."""
    text = normalize(question)
    if (
        verification_question(question)
        or models_results_question(question)
        or contribution_question(question)
    ):
        return False
    if re.search(
        r"\b(?:summari[sz]e|overview|research problem|research gap|abstract)\b|لخص|ملخص|المشكلة البحثية",
        text,
    ):
        return True
    return bool(
        re.search(r"\b(?:what is|what are|describe|introduce)\b|ما هو|ما هي|عرف", text)
        and re.search(
            r"\b(?:benchmark|dataset|corpus|paper|study|tasks?|questions?)\b|معيار|مجموعة بيانات|ورقة|دراسة",
            text,
        )
    )
