"""Conservative coordinate-bound extraction of numerical model-result tables.

Printed cells, not linear PDF token order, establish row/column relationships.
The return value is document data, never a prompt or an answer. Ambiguous tables
return diagnostics and no trusted rows. Synthesized relations are deliberately
separate from exact native source fragments used for citation verification.
"""

from __future__ import annotations

import re
from statistics import median

import pymupdf

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?$")
_SIZE = re.compile(r"^\d+(?:\.\d+)?[BMT](?:-[A-Z]?\d+(?:\.\d+)?[BMT])?$", re.I)
_LANGUAGES = {"english": "English", "arabic": "Arabic"}
_CONDITIONS = {"mcq", "completion", "standard", "cot", "zero", "zero-cot", "few", "few-cot"}
_TABLE_LABEL = r"Table\s+(?:\d+|[IVXLCDM]+)"


def _cx(word):
    return (word[0] + word[2]) / 2


def _cy(word):
    return (word[1] + word[3]) / 2


def _box(words):
    return [min(w[0] for w in words), min(w[1] for w in words),
            max(w[2] for w in words), max(w[3] for w in words)]


def _source(words):
    """Each fragment is a native printed word, never a reconstructed row quote."""
    return [{"text": w[4], "bbox": list(w[:4]), "word_ids": list(w[5:8])} for w in words]


def _lines(words, tolerance=2.0):
    lines = []
    for w in sorted(words, key=lambda w: (_cy(w), w[0])):
        if lines and abs(_cy(w) - median(_cy(v) for v in lines[-1])) <= tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w[0]) for line in lines]


def _joined_headers(words):
    groups = []
    for word in sorted(words, key=lambda w: w[0]):
        if groups and word[0] - groups[-1][-1][2] < 3:
            groups[-1].append(word)
        else:
            groups.append([word])
    return [{"text": " ".join(w[4] for w in group), "bbox": _box(group),
             "source": _source(group)} for group in groups]


def _header_center(header):
    return (header["bbox"][0] + header["bbox"][2]) / 2


def _caption_blocks(page):
    result = []
    for block in page.get_text("blocks"):
        text = block[4].strip()
        if re.match(r"^" + _TABLE_LABEL + r"(?:\s*[:.]|\s*\n)", text, re.I):
            result.append({"text": text, "bbox": list(block[:4]), "block_id": block[5]})
    return result


def _caption(captions, bbox, header_y):
    eligible = [c for c in captions if min(c["bbox"][2], bbox[2])
                > max(c["bbox"][0], bbox[0])]
    # A caption above the table is often multi-line; compare its lower edge.
    def distance(c):
        b = c["bbox"]
        return min(abs(b[3] - header_y), abs(b[1] - bbox[3]))
    return min(eligible, key=distance, default=None)


def _result_caption_role(caption):
    """A formatted example is document data, not evidence of evaluated models."""
    if not caption:
        return "No caption establishes an evaluation-result table role"
    description = re.sub(r"^" + _TABLE_LABEL + r"(?:\s*[:.]|\s*\n)\s*", "", caption, flags=re.I)
    description = " ".join(description.split())
    illustrative = (
        r"^(?:(?:an?|the)\s+)?(?:sample\s+)?"
        r"(?:examples?|illustrations?|illustrative|hypothetical|demonstrations?|templates?|prompts?)\b|"
        r"\b(?:prompt|instruction|response|output)\s+(?:templates?|examples?|schemas?|demonstrations?)\b|"
        r"\b(?:illustrative|hypothetical)\b|"
        r"\b(?:examples?|demonstrations?|templates?)\s+(?:of|for)\s+(?:a\s+|an\s+|the\s+)?"
        r"(?:prompt|instruction|format|table|schema|response|output)\b"
    )
    if re.search(illustrative, description, re.I):
        return "Illustrative or prompt/template table does not establish evaluated-model results"
    if not re.search(
        r"\b(?:results?|accuracy|performance|scores?|benchmarks?|evaluation|evaluated|"
        r"versus|vs|loss|perplexity|latency|precision|recall|f1(?:-score)?)\b", description, re.I
    ):
        return "Caption does not establish a supported evaluation-result table role"
    return None


def _column_parents(leaves, parent_words, grids, first_y):
    parents = _joined_headers(parent_words)
    if not parents or len(parents) >= len(leaves):
        return [None] * len(leaves)
    result = []
    for leaf in leaves:
        x = _header_center(leaf)
        cell = next((b for b in grids if b and b[0] <= x <= b[2]
                     and b[1] <= first_y <= b[3]), None)
        in_cell = [p for p in parents if cell and cell[0] <= _header_center(p) <= cell[2]]
        if len(in_cell) == 1:
            result.append(in_cell[0])
        elif len(in_cell) > 1:
            result.append(None)
        else:
            result.append(min(parents, key=lambda p: abs(_header_center(p) - x)))
    return result


def _roles(path, caption):
    language = next((_LANGUAGES[p.lower()] for p in path if p.lower() in _LANGUAGES), None)
    condition = next((p for p in path if p.lower() in _CONDITIONS), None)
    benchmark = next((p for p in reversed(path) if p.lower() not in
                      {*_LANGUAGES, *_CONDITIONS, "average", "score"}), None)
    if "average" in (p.lower() for p in path):
        benchmark = "Average"
    metric = None
    if re.search(r"accuracy", caption, re.I):
        metric = "accuracy"
    elif re.search(r"pass\s*@\s*1", caption, re.I):
        metric = "pass@1"
    elif re.search(r"\bF1(?:[- ]score)?\b", caption, re.I):
        metric = "F1"
    return {"language": language, "benchmark": benchmark, "condition": condition,
            "metric": metric}


def _entity_for_row(row_words, model_words, model_x, grids, row_y):
    """Use merged-cell geometry, or an original native text block for unruled tables."""
    cell = next((b for b in grids if b and b[0] <= model_x <= b[2]
                 and b[1] <= row_y <= b[3]), None)
    if cell:
        group = [w for w in model_words if cell[1] - 1 <= _cy(w) <= cell[3] + 1]
        lines = _lines(group)
        if not lines:
            return None
        if len(lines) == 1:
            chosen = lines[0]
        else:
            chosen = min(lines, key=lambda line: abs(median(_cy(w) for w in line) - row_y))
    else:
        inline = [w for w in model_words if abs(_cy(w) - row_y) < 2.5]
        if inline:
            chosen = sorted(inline, key=lambda w: w[0])
        else:
            blocks = {w[5] for w in row_words}
            group = [w for w in model_words if w[5] in blocks]
            lines = _lines(group)
            if len(lines) != 1:
                return None
            chosen = lines[0]
    return {"text": " ".join(w[4] for w in chosen), "source": _source(chosen)}


def _header_structure(words, model_word, header_words):
    """Resolve a short printed header band, including vertically centered Model.

    A top-level label beyond all lower subcolumns is a standalone result column,
    rather than a parent of unrelated benchmark subcolumns. No model names or
    paper table numbers enter this geometry rule.
    """
    band = [w for w in words if w[0] > model_word[2] + 5
            and abs(_cy(w) - _cy(model_word)) <= 12
            and not _NUMBER.fullmatch(w[4]) and w[4].lower() not in {"model", "size"}
            and (w[5] == model_word[5] or abs(_cy(w) - _cy(model_word)) <= 2)]
    levels = _lines(band)
    if len(levels) < 2:
        return _joined_headers([w for w in header_words if w[4].lower() not in {"model", "size"}
                                and w[0] > model_word[2] + 5]), [], []
    leaves = _joined_headers(levels[-1])
    upper = _joined_headers([w for line in levels[:-1] for w in line])
    standalone = [h for h in upper if _header_center(h) > leaves[-1]["bbox"][2] + 4]
    standalone_ids = {tuple(f["word_ids"]) for h in standalone for f in h["source"]}
    parents = [w for line in levels[:-1] for w in line if tuple(w[5:8]) not in standalone_ids]
    return leaves + standalone, parents, standalone


def _extract_candidate(page, page_number, header_words, model_word, grid, captions, index):
    words = page.get_text("words")
    header_y = _cy(model_word)
    if grid and grid.bbox[3] < header_y + 15:
        # Partial ruling may enclose only the header, not the result rows.
        grid = None
    size_header = next((w for w in header_words if w[4].lower() == "size"), None)
    leaves, band_parents, standalone = _header_structure(words, model_word, header_words)
    record = {"table_id": f"p{page_number}:t{index}", "page": page_number,
              "caption": "", "caption_provenance": None, "bbox": [], "columns": [],
              "native_page_text": page.get_text("text", sort=False),
              "rows": [], "context": {}, "status": "ambiguous", "diagnostics": []}
    if len(leaves) < 2:
        record["diagnostics"].append("Fewer than two independently identifiable result columns")
        return record
    first_x = leaves[0]["bbox"][0]
    left = min(model_word[0], size_header[0] if size_header else model_word[0]) - 5
    right = grid.bbox[2] if grid else max(h["bbox"][2] for h in leaves) + 5
    bottom = grid.bbox[3] + 1 if grid else page.rect.height
    # Captions and subsequent Model header lines delimit independent tables.
    stops = [c["bbox"][1] for c in captions if c["bbox"][1] > header_y + 5
             and c["bbox"][0] < right and c["bbox"][2] > left]
    stops += [w[1] - 25 for w in words if w[4].lower() == "model"
              and w[0] >= left - 1 and _cy(w) > header_y + 25]
    if stops:
        bottom = min(bottom, min(stops))
    data_words = [w for w in words if left <= _cx(w) <= right
                  and header_y + 5 < _cy(w) < bottom]
    numeric_lines = [line for line in _lines(data_words)
                     if sum(bool(_NUMBER.fullmatch(w[4])) and w[0] >= first_x - 3
                            for w in line) >= 2]
    if not numeric_lines:
        record["diagnostics"].append("No aligned numerical rows under result headers")
        return record
    # A long gap is prose or a new region, rather than a continuation of a table.
    gaps = [median(_cy(w) for w in b) - median(_cy(w) for w in a)
            for a, b in zip(numeric_lines, numeric_lines[1:])]
    if gaps:
        typical_gap = median(gaps)
        for j, gap in enumerate(gaps):
            if gap > max(25, typical_gap * 3):
                numeric_lines = numeric_lines[:j + 1]
                break
    first_y = median(_cy(w) for w in numeric_lines[0])
    last_y = median(_cy(w) for w in numeric_lines[-1])
    grids = grid.cells if grid else []
    above = [line for line in _lines(words) if header_y - 28 < median(_cy(w) for w in line)
             < header_y - 4 and sum(w[0] >= first_x - 30 for w in line) >= 2]
    parent_words = band_parents
    if above and not parent_words and not standalone:
        parent_words = [w for w in above[-1] if first_x - 30 <= _cx(w) <= right
                        and not _NUMBER.fullmatch(w[4])
                        and w[5] not in {c["block_id"] for c in captions}]
    grouped_leaves = [h for h in leaves if h not in standalone]
    parents = _column_parents(grouped_leaves, parent_words, grids, first_y) + [None] * len(standalone)
    record["bbox"] = [left, min([w[1] for w in parent_words] or [model_word[1]]), right,
                       max(w[3] for w in numeric_lines[-1])]
    caption = _caption(captions, record["bbox"], header_y)
    role_error = _result_caption_role(caption["text"] if caption else "")
    if role_error:
        record["diagnostics"].append(role_error)
        if caption:
            record["caption"] = caption["text"]
            record["caption_provenance"] = caption
        return record
    if caption:
        record["caption"] = caption["text"]
        record["caption_provenance"] = caption
        variant = re.search(r"\b(base|instruct(?:ion[- ]tuned)?)\s+models\b",
                            caption["text"], re.I)
        if variant:
            record["context"]["variant"] = variant.group(1).lower()
        if re.search(r"values\s+in\s*%|percentage", caption["text"], re.I):
            record["context"]["unit"] = "%"
    centers = [_header_center(h) for h in leaves]
    boundaries = [(a + b) / 2 for a, b in zip(centers, centers[1:])]
    for i, (leaf, parent) in enumerate(zip(leaves, parents), 1):
        path = ([parent["text"]] if parent else []) + [leaf["text"]]
        record["columns"].append({"column_id": f"C{i}", "index": i, "path": path,
                                  "source": (parent["source"] if parent else [])
                                  + leaf["source"], **_roles(path, record["caption"])})
    if len({tuple(c["path"]) for c in record["columns"]}) != len(leaves):
        record["diagnostics"].append("Repeated leaf columns lack unambiguous grouped headers")
        return record
    # Size values are printed separately even when their column has no Size heading.
    sizes = [w for w in data_words if _SIZE.fullmatch(w[4])
             and left <= _cx(w) < first_x]
    if size_header:
        size_x = _cx(size_header)
        sizes = [w for w in sizes if abs(_cx(w) - size_x) < max(20, first_x - size_x)]
    else:
        # A parameter token inside a complete inline model name remains part of
        # that name. Only rows with absent family labels justify an unheaded size
        # column, as in a vertically merged family/size table.
        all_inline = all(any(w[0] < first_x - 3 and not _NUMBER.fullmatch(w[4])
                            and not _SIZE.fullmatch(w[4]) for w in line)
                         for line in numeric_lines)
        if all_inline:
            sizes = []
    size_left = min((w[0] for w in sizes), default=first_x)
    model_left = size_header[2] + 3 if size_header and size_header[0] < model_word[0] else left
    model_right = first_x if size_header and size_header[0] < model_word[0] else size_left - 1
    model_words = [w for w in data_words if model_left <= w[0] < model_right
                   and (not sizes or (not _NUMBER.fullmatch(w[4]) and not _SIZE.fullmatch(w[4])))
                   and first_y - 4 <= _cy(w) <= last_y + 5]
    parsed = []
    for j, line in enumerate(numeric_lines, 1):
        y = median(_cy(w) for w in line)
        numbers = [w for w in line if _NUMBER.fullmatch(w[4]) and w[0] >= first_x - 3]
        ordered = [[] for _ in leaves]
        for w in numbers:
            col = sum(_cx(w) > bound for bound in boundaries)
            ordered[col].append(w)
        if any(len(values) != 1 for values in ordered):
            record["diagnostics"].append(f"Row {j}: missing or multiple values in result column")
            continue
        entity = _entity_for_row(line, model_words, _cx(model_word), grids, y)
        if entity is None:
            record["diagnostics"].append(f"Row {j}: model identity lacks a geometric/native-block link")
            continue
        row_sizes = [w for w in sizes if abs(_cy(w) - y) < 2.5]
        if len(row_sizes) > 1:
            record["diagnostics"].append(f"Row {j}: ambiguous size association")
            continue
        size = row_sizes[0][4] if row_sizes else None
        cells = []
        for column, values in zip(record["columns"], ordered):
            word = values[0]
            cells.append({"fact_id": f"{record['table_id']}:R{j}:{column['column_id']}",
                          "column_id": column["column_id"], "path": column["path"],
                          "value": word[4], "source": _source([word]),
                          **{k: column[k] for k in ("benchmark", "language", "condition", "metric")}})
        parsed.append({"row_id": f"R{j}", "entity": entity["text"], "entity_type": "model",
                       "size": size, "entity_source": entity["source"],
                       "size_source": _source(row_sizes), "source": _source(line), "cells": cells})
    if record["diagnostics"]:
        # A partially linked table must never silently become a complete model list.
        record["rows"] = []
        return record
    record["rows"] = parsed
    record["status"] = "ok"
    return record


def extract_result_tables(data: bytes) -> list[dict]:
    """Extract supported model-result layouts; retain rejected-layout diagnostics.

    Supports ruled grouped rows and unruled rows whose native text blocks link
    family labels to sizes and scores. Other layouts fail closed. Metric labels
    remain unknown unless explicitly stated in the caption; callers may bind
    additional source-verified declarations without treating quotes as entailment.
    No PDF bytes, corpus records, or backups are modified.
    """
    result = []
    with pymupdf.open(stream=data, filetype="pdf") as document:
        for page_number, page in enumerate(document, 1):
            words = page.get_text("words")
            headers = []
            for line in _lines(words):
                model = next((w for w in line if w[4] == "Model"), None)
                # This deliberately limits supported layouts to result headers,
                # rather than classifying prose containing "model" as tables.
                if model:
                    leaves, _, _ = _header_structure(words, model, line)
                    if len(leaves) >= 2:
                        headers.append((line, model))
            if not headers:
                continue
            try:
                grids = page.find_tables().tables
            except Exception:
                grids = []
            captions = _caption_blocks(page)
            for i, (header, model) in enumerate(headers, 1):
                grid = next((t for t in grids if t.bbox[0] <= _cx(model) <= t.bbox[2]
                             and t.bbox[1] - 2 <= _cy(model) <= t.bbox[3]), None)
                result.append(_extract_candidate(page, page_number, header, model,
                                                 grid, captions, i))
    return result
