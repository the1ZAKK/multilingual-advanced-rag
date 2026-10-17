"""Render coordinate-bound result tuples without asking an LLM to interpret cells."""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from difflib import SequenceMatcher

from rag.generation.ollama import GeneratedAnswer, validate_answer
from rag.models import Hit

logger = logging.getLogger(__name__)


def _display(value):
    text = str(value).replace("\n", " ").strip()
    # Table structure is ours; document cells/captions must remain plain text.
    return re.sub(r"([\\`*_{}\[\]<>|])", r"\\\1", text)


def _aggregate_tables(tables):
    """Prefer aggregate score tables over overlapping subject-level breakdowns.

    This is a relation-shape rule, independent of paper names, table numbers,
    model names, benchmark names, or values. Other prompting conditions remain
    separate and keep their printed hierarchy.
    """
    aggregate_variants = {
        (t.get("document_id"), t.get("context", {}).get("variant"))
        for t in tables
        if {str(c.get("condition", "")).lower() for c in t["columns"]} == {"mcq", "completion"}
        and len({c.get("benchmark") for c in t["columns"]}) <= 2
    }
    return [
        t
        for t in tables
        if not (
            {str(c.get("condition", "")).lower() for c in t["columns"]} == {"mcq", "completion"}
            and len({c.get("benchmark") for c in t["columns"]}) > 2
            and (t.get("document_id"), t.get("context", {}).get("variant")) in aggregate_variants
        )
    ]


def _table_fragments(table):
    fragments = []
    for column in table["columns"]:
        fragments += column["source"]
    for row in table["rows"]:
        fragments += row["entity_source"] + row["size_source"]
        if " ".join(f["text"] for f in row["entity_source"]) != row["entity"]:
            raise ValueError("Table model name conflicts with native entity fragments")
        if row.get("size") and " ".join(f["text"] for f in row["size_source"]) != row["size"]:
            raise ValueError("Table size conflicts with native size fragments")
        columns = {c["column_id"]: c for c in table["columns"]}
        if len(row["cells"]) != len(columns) or {cell["column_id"] for cell in row["cells"]} != set(
            columns
        ):
            raise ValueError("Incomplete result row")
        for cell in row["cells"]:
            if (
                cell["column_id"] not in columns
                or cell["path"] != columns[cell["column_id"]]["path"]
            ):
                raise ValueError("Result cell conflicts with column hierarchy")
            if len(cell["source"]) != 1 or cell["source"][0]["text"] != cell["value"]:
                raise ValueError("Result value conflicts with exact native cell fragment")
            if not math.isfinite(float(cell["value"].rstrip("%"))):
                raise ValueError("Non-finite result value")
            fragments += cell["source"]
    return fragments


def _native_quotes(indexed, native):
    """Unchanged common substrings; reordered extraction is never a native quote."""
    if indexed in native:
        return [indexed]
    matcher = SequenceMatcher(None, indexed, native, autojunk=False)
    return [
        indexed[match.a : match.a + match.size].strip()
        for match in matcher.get_matching_blocks()
        if match.size >= 12 and indexed[match.a : match.a + match.size].strip()
    ]


def _native_sources(table, evidence):
    native = table.get("native_page_text")
    if not isinstance(native, str) or not native:
        raise ValueError("Original native PDF page text is required for table provenance")
    candidates = [
        (f"S{i}", hit.chunk)
        for i, hit in enumerate(evidence, 1)
        if hit.chunk.page == table["page"]
        and (not table.get("document_id") or hit.chunk.document_id == table["document_id"])
    ]
    if not candidates or len({chunk.document_id for _, chunk in candidates}) != 1:
        raise ValueError("Table page does not resolve to one source document")
    fragments = _table_fragments(table)
    # Every printed caption line is original source data, not reconstructed prose.
    fragments += [{"text": line} for line in table.get("caption", "").splitlines() if line.strip()]
    quotes = OrderedDict()
    common = {label: _native_quotes(chunk.text.strip(), native) for label, chunk in candidates}
    for fragment in fragments:
        text = fragment.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Empty native table fragment")
        matches = [
            (label, chunk) for label, chunk in candidates if text in chunk.text and text in native
        ]
        if not matches:
            raise ValueError("Native table fragment missing from indexed source evidence")
        # Ambiguous repeated tokens are covered by all matching exact source
        # passages. Geometry in the trusted extraction determines relationships;
        # substring occurrence alone never proves a row/value association.
        for label, _ in matches:
            covering = [quote for quote in common[label] if text in quote]
            # A printed word is itself an exact native/source substring when a
            # reading-order boundary splits the larger native block.
            for quote in covering or [text]:
                quotes[label, quote] = {"label": label, "quote": quote}
    labels = sorted({label for label, _ in quotes}, key=lambda label: int(label[1:]))
    return labels, quotes


def _maxima_rows(table):
    leaders = set()
    for column in table["columns"]:
        col = column["column_id"]
        values = [
            (
                row["row_id"],
                float(
                    next(cell["value"] for cell in row["cells"] if cell["column_id"] == col).rstrip(
                        "%"
                    )
                ),
            )
            for row in table["rows"]
        ]
        maximum = max(value for _, value in values)
        leaders.update(row_id for row_id, value in values if value == maximum)
    return [row for row in table["rows"] if row["row_id"] in leaders]


def _verify_metric_declaration(binding):
    text = re.sub(r"-\s*\n\s*", "", binding["quote"])
    text = re.sub(r"\s+", " ", text).lower()
    metric = binding["metric"].lower()
    if metric == "accuracy":
        declared = re.search(
            r"\baccuracy\s+(?:(?:is|was)\s+)?(?:computed|measured|reported)\b", text
        )
    elif metric == "pass@1":
        declared = re.search(r"pass\s*@\s*1\s+evaluation\s+(?:was|is)\s+(?:adopted|used)\b", text)
    else:
        declared = False
    if not declared:
        raise ValueError("Metric binding lacks a supported direct metric declaration")
    if re.search(r"\b(?:not|never|no)\b[^.;]{0,45}" + re.escape(metric), text):
        raise ValueError("Metric binding contains conflicting negated metric declaration")


def table_answer(evidence: list[Hit], language: str) -> GeneratedAnswer | None:
    """Return source-derived model catalogs/results, with strict quote provenance.

    Callers supply trusted extraction metadata and original indexed page passages.
    Missing or conflicting fragments reject that entire table, rather than
    silently completing it from a model's memory. This renderer does not infer
    higher performance, metric units, or paper-wide model counts from bare cells.
    """
    tables, seen = [], set()
    for hit in evidence:
        for table in getattr(hit, "tables", []) or []:
            key = (table.get("document_id", hit.chunk.document_id), table.get("table_id"))
            if key in seen or table.get("status") != "ok" or not table.get("rows"):
                continue
            seen.add(key)
            tables.append({**table, "document_id": key[0]})
    if not tables:
        return None
    tables = _aggregate_tables(tables)
    accepted = []
    quotes = OrderedDict()
    for table in tables:
        try:
            table_quotes = OrderedDict()
            labels, native_quotes = _native_sources(table, evidence)
            table_quotes.update(native_quotes)
            metric_names = {c.get("metric") for c in table["columns"] if c.get("metric")}
            for binding in table.get("metric_bindings", []):
                _verify_metric_declaration(binding)
                if metric_names and binding["metric"] not in metric_names:
                    raise ValueError("Methods metric conflicts with the table's explicit metric")
                label, quote = binding["label"], binding["quote"]
                if not re.fullmatch(r"S[1-9]\d*", label):
                    raise ValueError("Invalid metric binding source label")
                position = int(label[1:]) - 1
                if position >= len(evidence) or quote not in evidence[position].chunk.text:
                    raise ValueError("Metric binding quote is not exact indexed evidence")
                if evidence[position].chunk.document_id != table["document_id"]:
                    raise ValueError("Metric binding belongs to a different document")
                metric_names.add(binding["metric"])
                native = binding.get("native_page_text")
                if not isinstance(native, str) or not native:
                    raise ValueError("Metric binding requires original native PDF page text")
                declared_quotes = []
                for exact in _native_quotes(quote, native):
                    try:
                        _verify_metric_declaration({**binding, "quote": exact})
                        declared_quotes.append(exact)
                    except ValueError:
                        continue
                if not declared_quotes:
                    raise ValueError(
                        "No contiguous native/source quote establishes the metric declaration"
                    )
                for exact in declared_quotes:
                    table_quotes[label, exact] = {"label": label, "quote": exact}
                if label not in labels:
                    labels.append(label)
            accepted.append((table, labels, sorted(metric_names)))
            quotes.update(table_quotes)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            logger.warning("table_answer_rejected table=%s reason=%s", table.get("table_id"), exc)
            continue
    if not accepted:
        return None
    ar = language == "ar"
    model_label, size_label = ("النموذج", "الحجم") if ar else ("Model", "Size")
    catalog_label = (
        "النماذج وأحجامها في صفوف الجدول" if ar else "Models and sizes in the table rows"
    )
    representative_label = (
        "صفوف مختارة بحسب أعلى قيمة معروضة في كل عمود، مع التعادلات"
        if ar
        else "Representative rows selected by column maxima, including ties"
    )
    lines = []
    for table, labels, metrics in accepted:
        markers = " ".join(f"[{label}]" for label in labels)
        caption = _display(table["caption"])
        page_label = "الصفحة" if ar else "page"
        lines.append(f"**{caption}** ({page_label} {table['page']}) {markers}")
        groups = OrderedDict()
        for row in table["rows"]:
            groups.setdefault(row["entity"], [])
            if row.get("size") and row["size"] not in groups[row["entity"]]:
                groups[row["entity"]].append(row["size"])
        catalog = "; ".join(
            _display(entity) + (" (" + ", ".join(sizes) + ")" if sizes else "")
            for entity, sizes in groups.items()
        )
        lines.append(f"{catalog_label}: {catalog} {markers}")
        metric_label = "المقياس" if ar else "Metric"
        if metrics:
            lines.append(f"{metric_label}: {', '.join(metrics)} {markers}")
        else:
            text = (
                "تُعرض القيم كما في المصدر؛ لم يحدد تعليق الجدول مقياسًا"
                if ar
                else "Values retain the source scale; the caption does not name a metric"
            )
            lines.append(f"{text} {markers}")
        selection = table.get("selection")
        if selection:
            note = (
                f"عرض محدد: {selection['shown_rows']} من {selection['source_rows']} صفوف، "
                f"و{selection['shown_columns']} من {selection['source_columns']} أعمدة. "
                "ليس هذا حصرًا شاملًا للنماذج أو النتائج في الورقة."
                if ar
                else f"Focused view: {selection['shown_rows']} of {selection['source_rows']} rows and "
                f"{selection['shown_columns']} of {selection['source_columns']} columns; "
                f"{selection['shown_tables']} of {selection['source_tables']} matching tables. "
                "This is not an exhaustive paper-wide model or result list."
            )
            lines.append(note)
        else:
            lines.append(f"{representative_label}: {markers}")
        headers = [model_label, size_label] + [
            _display(" / ".join(c["path"])) for c in table["columns"]
        ]
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in table["rows"] if table.get("render_all_rows") else _maxima_rows(table):
            values = {c["column_id"]: c["value"] for c in row["cells"]}
            fields = [_display(row["entity"]), _display(row.get("size") or "-")]
            fields += [values[c["column_id"]] for c in table["columns"]]
            fields[-1] += " " + markers
            table_lines.append("| " + " | ".join(fields) + " |")
        lines.append("\n".join(table_lines))
    answer = "\n\n".join(lines).strip()
    # Reuse the strict existing citation verifier without interpreting Markdown
    # rows as prose sentences. Actual row entailment comes from trusted geometry.
    proof_markers = " ".join(f"[{label}]" for label, _ in quotes)
    _, citations = validate_answer(
        {
            "answer": f"Verified source table fragments {proof_markers}",
            "citations": list(quotes.values()),
        },
        evidence,
    )
    if set(re.findall(r"\[S\d+\]", answer)) != {f"[{c.label}]" for c in citations}:
        raise ValueError("Rendered table source markers do not match validated citations")
    return GeneratedAnswer(answer, citations, mode="source-tables")
