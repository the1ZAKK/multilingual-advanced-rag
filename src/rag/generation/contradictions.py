"""A conservative veto for explicit English same-subject polarity conflicts.

This is not general natural-language inference. In particular it does not infer
paper-wide absence or override a semantic rejection. Ambiguous overlap can cause
abstention; distinct subjects and future-tense predicates are left to the auditor.
"""

import re


def reject_polarity_conflicts(statements):
    texts = [item.get("text", "") for item in statements if isinstance(item, dict)]
    for negative in texts:
        match = re.match(
            r"(.+?)\s+(?:does\s+not|doesn['’]t|do\s+not|don['’]t)\s+(\w+)\s+(.+)", negative, re.I
        )
        if not match:
            continue
        subject, verb, obj = match.groups()
        subject = re.sub(r"\s+", " ", subject.strip()).casefold()
        # Only the main assertion is compared. A following future-work clause
        # does not turn a present-tense positive assertion into a future one.
        obj = re.split(r"[,;]|\b(?:but|whereas|while)\b", obj, maxsplit=1, flags=re.I)[0]
        object_words = set(re.findall(r"\w+", obj.casefold())) - {
            "the",
            "a",
            "an",
            "of",
            "to",
            "from",
            "in",
            "and",
            "or",
            "its",
            "their",
        }
        for positive in texts:
            if positive == negative:
                continue
            pattern = rf"^{re.escape(subject)}\s+{re.escape(verb)}(?:s|es)?\s+(.+)"
            asserted = re.match(pattern, re.sub(r"\s+", " ", positive.strip()), re.I)
            if asserted:
                clause = re.split(
                    r"[,;]|\b(?:but|whereas|while)\b", asserted[1], maxsplit=1, flags=re.I
                )[0]
                words = set(re.findall(r"\w+", clause.casefold()))
                if len(words & object_words) >= 4:
                    raise ValueError(
                        "Conflicting positive and negative assertions for the same subject/action"
                    )
