"""Query-focused views over extracted table data; never invent cells or relations."""

from __future__ import annotations

import copy
import re

from rag.ingestion.text import normalize, tokenize


def _digits(text):
    return text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))


def _key(text):
    return re.sub(r"[^\w]", "", _digits(normalize(text)))


def _contains(query, phrase):
    return len(_key(phrase)) >= 3 and _key(phrase) in _key(query)


def select_tables(
    tables: list[dict], query: str, *, max_tables=2, max_rows=6, max_columns=6
) -> list[dict]:
    """Focus table ID, prompting condition, named model and named column.

    Explicit constraints fail closed. Broad queries return bounded representative
    views and label omissions, rather than implying a full-corpus model census.
    Original source fragments and cell IDs survive selection unchanged.
    """
    if not query.strip():
        return [t for t in tables if t.get("status") == "ok"]
    candidates = [t for t in tables if t.get("status") == "ok" and t.get("rows")]
    q = _digits(normalize(query))
    requested = set(re.findall(r"(?:\btable|الجدول)\s*(\d+)\b", q))
    if requested:
        candidates = [
            t
            for t in candidates
            if (m := re.search(r"(?:Table|الجدول)\s*(\d+)", t.get("caption", ""), re.I))
            and m[1] in requested
        ]
    conditions = []
    for pattern in [r"zero[- ]shot", r"few[- ]shot", r"zero[- ]cot", r"few[- ]cot"]:
        if re.search(pattern, q):
            conditions.append(pattern)
    if re.search("دون امثلة|صفر امثلة", q):
        conditions.append(r"zero[- ]shot")
    if re.search("امثلة قليلة|قليل الامثلة|بضع امثلة", q):
        conditions.append(r"few[- ]shot")
    if conditions:
        candidates = [
            t
            for t in candidates
            if any(
                re.search(p, normalize(t.get("caption", "")))
                or any(re.search(p, normalize(" ".join(c["path"]))) for c in t["columns"])
                for p in conditions
            )
        ]
    named = {
        row["entity"]
        for table in candidates
        for row in table["rows"]
        if _contains(q, row["entity"])
    }
    if named:
        candidates = [t for t in candidates if any(r["entity"] in named for r in t["rows"])]
    stop = {
        "table",
        "model",
        "models",
        "results",
        "scores",
        "performance",
        "the",
        "and",
        "what",
        "which",
        "paper",
    }
    tokens = set(tokenize(q)) - stop

    def rank(table):
        words = set(
            tokenize(
                table.get("caption", "")
                + " "
                + " ".join(" ".join(c["path"]) for c in table["columns"])
            )
        )
        return (-len(tokens & words), table["page"], table["table_id"])

    candidates = sorted(candidates, key=rank)
    output = []
    for original in candidates[:max_tables]:
        table = copy.deepcopy(original)
        total_rows, total_columns = len(table["rows"]), len(table["columns"])
        rows = [r for r in table["rows"] if r["entity"] in named] if named else table["rows"]
        # For broad questions choose a bounded sample of column leaders. A
        # requested comparison keeps every named row, including weaker models.
        if not named:
            leaders = set()
            for col in table["columns"]:
                cells = [
                    (
                        r["row_id"],
                        float(
                            next(
                                c["value"] for c in r["cells"] if c["column_id"] == col["column_id"]
                            ).rstrip("%")
                        ),
                    )
                    for r in rows
                ]
                best = max(v for _, v in cells)
                leaders.update(rid for rid, value in cells if value == best)
            rows = [r for r in rows if r["row_id"] in leaders]
        table["rows"] = rows[:max_rows]
        # A literal requested benchmark/column takes precedence over a broad view.
        specific = [
            c
            for c in table["columns"]
            if any(
                _contains(q, part)
                for part in c["path"]
                if normalize(part)
                not in {"english", "arabic", "average", "score", "standard", "cot"}
            )
        ]
        columns = specific or table["columns"]
        columns = columns[:max_columns]
        table["columns"] = columns
        keep = {c["column_id"] for c in columns}
        for row in table["rows"]:
            row["cells"] = [c for c in row["cells"] if c["column_id"] in keep]
        table["render_all_rows"] = True
        table["selection"] = {
            "source_rows": total_rows,
            "source_columns": total_columns,
            "shown_rows": len(table["rows"]),
            "shown_columns": len(columns),
            "source_tables": len(candidates),
            "shown_tables": min(len(candidates), max_tables),
        }
        output.append(table)
    return output
