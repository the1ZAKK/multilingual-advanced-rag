import re
import unicodedata

_ARABIC_MARKS = re.compile("[\\u0610-\\u061a\\u064b-\\u065f\\u0670\\u06d6-\\u06ed]")
_ARABIC_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ـ": ""})


def normalize(text: str, *, casefold: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ARABIC_MAP)
    text = _ARABIC_MARKS.sub("", text)
    return " ".join((text.casefold() if casefold else text).split())


def tokenize(text: str, *, casefold: bool = True) -> list[str]:
    return re.findall(r"[^\W_]+", normalize(text, casefold=casefold), flags=re.UNICODE)


def is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06ff]", unicodedata.normalize("NFKC", text)))
