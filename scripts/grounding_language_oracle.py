"""Offline, independently annotated semantic regressions; never used to produce answers.

The paper-specific reference is test data, not a production answer. Relation checks are
conservative: an unfamiliar paraphrase is a manual-review failure, not a proof of falsity.
Exact citation provenance is reported separately and is never treated as entailment.
"""

import json
import re
import unicodedata
from pathlib import Path

if __package__ in {None, ""}:
    from contribution_oracle import evaluate_author_items
else:
    from scripts.contribution_oracle import evaluate_author_items

REFERENCE = Path(__file__).resolve().parents[1] / "tests/fixtures/grounding-language-reference.json"
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize(text):
    text = unicodedata.normalize("NFKC", text).translate(_DIGITS).casefold()
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    text = text.translate(str.maketrans("أإآىؤئ", "ااايوي"))
    text = re.sub(r"(?<=\w)-\s*\n(?=\w)", "", text)
    return " ".join(text.split())


def prose_only(answer):
    # Source blockquotes and source-name table cells do not determine answer language.
    lines = [line for line in answer.splitlines() if not line.lstrip().startswith((">", "|"))]
    return re.sub(r"\[S\d+\]|\([^)]*(?:pages?|ص)[^)]*\)", " ", "\n".join(lines))


def language_check(answer, expected):
    body = prose_only(answer)
    ar = len(re.findall(r"[\u0621-\u064a]", body))
    en = len(re.findall(r"[A-Za-z]", body))
    fraction = ar / max(1, ar + en)
    return {
        "expected": expected,
        "arabic_prose_fraction": round(fraction, 4),
        "passed": ar >= 30 and fraction >= 0.60
        if expected == "ar"
        else en >= 20 and fraction <= 0.30,
        "note": "Checks explanatory prose, excluding source blockquotes and result-table cells.",
    }


def numbered_items(answer):
    normalized_digits = answer.translate(_DIGITS)
    marks = list(re.finditer(r"(?m)^\s*(\d+)[.)\-]\s*", normalized_digits))
    if marks:
        return [
            (
                int(m.group(1)),
                normalized_digits[
                    m.end() : marks[i + 1].start() if i + 1 < len(marks) else None
                ].strip(),
            )
            for i, m in enumerate(marks)
        ]
    # Also allow meaningful Arabic ordinal headings; numbering is required only here.
    marks = list(
        re.finditer(
            r"(?m)^\s*(?:المساهمة\s+)?(الأولى|الاولي|الأول|الاول|الثانية|الثاني|الثالثة|الثالث)\s*[:.)\-]",
            answer,
        )
    )
    ordinals = {
        "الأولى": 1,
        "الاولي": 1,
        "الأول": 1,
        "الاول": 1,
        "الثانية": 2,
        "الثاني": 2,
        "الثالثة": 3,
        "الثالث": 3,
    }
    return [
        (
            ordinals[m.group(1)],
            answer[m.end() : marks[i + 1].start() if i + 1 < len(marks) else None].strip(),
        )
        for i, m in enumerate(marks)
    ]


def _relation(text, action, obj, window=180):
    return bool(
        re.search(
            f"(?:{action}).{{0,{window}}}(?:{obj})|(?:{obj}).{{0,{window}}}(?:{action})", text
        )
    )


def _coordinated(text, left, right, window=120):
    """Require a joint assertion rather than nearby independent domain tokens."""
    for first, second in ((left, right), (right, left)):
        for start in re.finditer(first, text):
            for end in re.finditer(second, text[start.end() : start.end() + window]):
                between = text[start.end() : start.end() + end.start()]
                if re.search(r"(?:^|\s)(?:و|وكذلك|وايضا|بالاضافة\s+الي|فضلا\s+عن)\s*$", between):
                    return True
    return False


def _scientific_programming_capabilities(text):
    science = r"(?:ال)?علمي(?:ة)?|العلوم|stem"
    programming = r"(?:ال)?برمجي(?:ة)?"
    # Adjectives must qualify capabilities/performance or domains. A loose bag
    # of science/programming words can misdescribe a geographical region.
    return (
        _coordinated(text, science, programming)
        and _relation(text, r"قدرات|اداء|امكانيات|مجال|ميدان|ميادين", science, 100)
    ) or (
        _coordinated(text, science, r"البرمجة|برمجة|توليد\s+(?:الشيفرة|الشفرة|الكود)")
        and _relation(text, r"مجال|ميدان|ميادين", science, 100)
    )


def contribution_quality(answer, expected_language):
    items = numbered_items(answer)
    if expected_language != "ar":
        legacy = evaluate_author_items(answer)
        return {"kind": "author-contributions", **legacy}
    annotations = []
    for position, (number, raw) in enumerate(items[:3], 1):
        text = normalize(prose_only(raw))
        checks = {
            "asserted_not_negated": not re.search(
                r"(?:لم|لن|ليس|ليست|لا)\s+(?:تقدم|قدم|يقيم|قيم|تقيم|يحلل|تحلل|يجري|اجري|يشمل|تشمل|تراجع|يراجع)|(?:دون|بدون)\s+(?:ترجم|تحقق|مراجع|اختبار|تقييم|تحليل)",
                text,
            ),
            "arabic_explanation": language_check(raw, "ar")["passed"],
        }
        if position == 1:
            checks.update(
                {
                    "introduced_three_benchmark_suite": _relation(
                        text,
                        r"تقديم|تقدم|قدم|انشاء|بناء|تطوير|طرح|اتاح|اطلاق|نعرض|عرض",
                        r"(?:ثلاث(?:ة)?|3)\s+(?:معايير|اختبارات|مجموعات|مقاييس|حزم)",
                    ),
                    "stem_and_code": _coordinated(
                        text,
                        r"stem|علوم|العلمي|علمية",
                        r"توليد\s+(?:الشيفر|الشفر|الكود)|(?:ال)?برمج|الاكواد",
                    ),
                    "native_arabic_curation": _relation(
                        text,
                        r"جمع|تنسيق|انتقا|تنقيح|اعداد|مصادر|محتو",
                        r"عربي.{0,30}(?:اصيل|اصلي|طبيعي)|(?:اصيل|اصلي|طبيعي).{0,30}عربي",
                        100,
                    ),
                    "synthetic_generation": _relation(
                        text, r"توليد|مولد|انتاج", r"اصطناعي|اصطناعيا|تركيبي|مصطنع", 65
                    ),
                    "careful_translation_with_human_verification": _relation(
                        text,
                        r"ترجم",
                        r"بشر.{0,40}(?:تحقق|تدقيق|مراجع|فحص)|(?:تحقق|تدقيق|مراجع|فحص).{0,40}بشر",
                        100,
                    ),
                }
            )
        elif position == 2:
            checks.update(
                {
                    "evaluation_of_language_models": _relation(
                        text, r"تقييم|قيم|اختبار|تقويم|قياس|فحص", r"نماذج|نموذج|llms?|models?", 180
                    ),
                    "over_forty_models": bool(
                        re.search(
                            r"(?:اكثر\s+من|ما\s+يزيد\s+(?:علي|عن)|يتجاوز|فوق)\s*(?:40|اربعين)|40\s*\+",
                            text,
                        )
                    ),
                    "arabic_and_multilingual_models": _relation(
                        text, r"عربي", r"متعدد.{0,25}لغ|متعددة.{0,25}لغ", 70
                    ),
                    "scientific_and_programming_capabilities": _scientific_programming_capabilities(
                        text
                    ),
                    "evaluation_not_one_benchmark_component": not re.search(
                        r"^(?:تقديم|تطوير|انشاء|بناء|طرح)?\s*(?:معيار|مجموعة|اختبار|مقياس).{0,45}(?:اصطناعي|ترجم|اصيل)",
                        text,
                    ),
                }
            )
        else:
            checks.update(
                {
                    "cross_task_correlation_analysis": _relation(
                        text, r"تحليل|حلل|دراس|فحص", r"ارتباط|ترابط", 120
                    )
                    and _relation(
                        text, r"ارتباط|ترابط", r"بين\s+المهام|عبر\s+المهام|مختلف.{0,20}مهام", 90
                    ),
                    "robustness_testing": _relation(
                        text,
                        r"اختبار|اختبارات|فحص|تقييم|تحليل",
                        r"متانة|المتانة|قوة\s+التحمل|صمود|ثبات|robustness",
                        100,
                    ),
                    "model_behavior_insights": _relation(
                        text,
                        r"روي|رؤي|فهم|كشف|استنتاج|رؤية|معرف|سلوك",
                        r"سلوك.{0,25}نماذج|سلوك.{0,25}نموذج|كيف.{0,30}(?:تتصرف|تعمل).{0,20}نماذج",
                        90,
                    ),
                    "relationships_between_cognitive_capabilities": _relation(
                        text,
                        r"علاق|ترابط|ارتباط|تفاعل",
                        r"قدرات.{0,20}(?:معرفي|ادراكي)|(?:معرفي|ادراكي).{0,20}قدرات|مهارات.{0,20}معرفي",
                        100,
                    ),
                    "joint_behavior_and_capability_findings": _coordinated(
                        text,
                        r"سلوك\s+(?:ال)?(?:نماذج|نموذج)",
                        r"(?:ال)?(?:علاق|ترابط|ارتباط)",
                        120,
                    ),
                    "analysis_not_translated_code_component": not re.search(
                        r"^(?:تقديم|تطوير|انشاء|بناء|طرح)?\s*(?:معيار|مجموعة|اختبار|مقياس).{0,45}(?:ترجم|كود|شيفر|شفر)",
                        text,
                    ),
                }
            )
        annotations.append(
            {
                "number": number,
                "text": raw,
                "assertions": checks,
                "missing_or_wrong": [k for k, v in checks.items() if not v],
                "passed": all(checks.values()),
            }
        )
    numbering = [number for number, _ in items]
    return {
        "kind": "author-contributions",
        "numbering": numbering,
        "exactly_three_authored_items": numbering == [1, 2, 3],
        "items": annotations,
        "passed": numbering == [1, 2, 3] and all(item["passed"] for item in annotations),
        "note": "Annotated action/object/qualifier relations per authored contribution; unrecognized paraphrases require manual review.",
    }


def family(model):
    value = re.sub(r"[^a-z0-9]", "", normalize(model))
    for prefix, output in [
        ("qwen25", "qwen2.5"),
        ("qwen3", "qwen3"),
        ("jais", "jais"),
        ("qcrifanar", "fanar"),
        ("fanar", "fanar"),
        ("llama", "llama"),
        ("acegpt", "acegpt"),
        ("gemma", "gemma"),
        ("silma", "silma"),
        ("aya", "aya"),
        ("c4ai", "c4ai"),
        ("allam", "allam"),
        ("yehia", "yehia"),
    ]:
        if value.startswith(prefix):
            return output
    return value


def _model_key(model):
    value = re.sub(r"[^a-z0-9]", "", normalize(model))
    return (
        value.removeprefix("qcri").removesuffix("1")
        if "fanar" in value and "instruct" in value
        else value.removeprefix("qcri")
    )


def markdown_tables(answer):
    tables, lines, start, last_table_end = [], answer.splitlines(), 0, 0
    while start < len(lines):
        if not lines[start].lstrip().startswith("|"):
            start += 1
            continue
        end = start
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        cells = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in lines[start:end]
        ]
        rows = [
            row for row in cells[1:] if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in row)
        ]
        if cells:
            tables.append(
                {
                    "headers": cells[0],
                    "rows": rows,
                    "context": "\n".join(lines[last_table_end:start]),
                    "line": start + 1,
                }
            )
        start = end
        last_table_end = end
    return tables


def _column(header, context):
    value, surrounding = normalize(header), normalize(context)
    if "native" in value or "synthetic" in value:
        return {
            "benchmark": "Native" if "native" in value else "Synthetic",
            "language": "Arabic",
            "condition": "MCQ"
            if "mcq" in value
            else "Completion"
            if "completion" in value
            else None,
            "metric": "accuracy",
        }
    lang = "English" if "english" in value else "Arabic" if "arabic" in value else None
    if "average" in value and lang:
        return {
            "benchmark": "EvalPlus suite average",
            "language": lang,
            "condition": "code generation",
            "metric": "pass@1",
        }
    for benchmark in ("HumanEval+", "HumanEval", "MBPP+", "MBPP"):
        if benchmark.casefold() in value:
            if not lang:
                lang = (
                    "Arabic"
                    if "arabic" in surrounding and "english" not in surrounding
                    else "English"
                    if "english" in surrounding and "arabic" not in surrounding
                    else None
                )
            return {
                "benchmark": benchmark,
                "language": lang,
                "condition": "code generation",
                "metric": "pass@1",
            }
    return None


def numeric_table_claims(answer):
    claims, role_entities, unparsed = [], [], []
    for table in markdown_tables(answer):
        headers = table["headers"]
        model_columns = [
            i
            for i, h in enumerate(headers)
            if normalize(h) in {"model", "models", "model / size", "model and size", "model (size)"}
        ]
        if not model_columns:
            continue
        mi = model_columns[0]
        si = next(
            (
                i
                for i, h in enumerate(headers)
                if normalize(h) in {"size", "parameters", "parameter size"}
            ),
            None,
        )
        context = table["context"]
        pages = [int(p) for p in re.findall(r"\bpages?\s*(\d+)", context, re.I)]
        for row in table["rows"]:
            if len(row) != len(headers):
                unparsed.append({"line": table["line"], "reason": "ragged table row", "row": row})
                continue
            model = re.sub(r"\[S\d+\]", "", row[mi]).strip()
            role_entities.append(model)
            size = row[si] if si is not None else ""
            size_match = re.search(r"\b\d+(?:\.\d+)?[BM](?:-A\d+B)?\b", size or model, re.I)
            size = size_match.group(0) if size_match else None
            if si is None and size:
                model = re.sub(re.escape(size), "", model, flags=re.I).strip(" /()-")
            caption = next(
                (
                    line
                    for line in reversed(context.splitlines())
                    if re.search(r"\bTable\s+\d+", line, re.I)
                ),
                "",
            )
            variant_text = normalize(model + " " + caption)
            variant = (
                "base"
                if re.search(r"base|\bpt\b", variant_text)
                else "instruct"
                if re.search(r"instruct|\bit\b|chat|instruction", variant_text)
                else None
            )
            for i, cell in enumerate(row):
                number = re.sub(r"\[S\d+\]", "", cell).strip()
                if i in {mi, si} or not re.fullmatch(r"\d+(?:\.\d+)?%?", number):
                    continue
                column = _column(headers[i], context)
                if not column:
                    unparsed.append(
                        {
                            "line": table["line"],
                            "reason": "missing benchmark/language/condition header",
                            "header": headers[i],
                            "value": cell,
                        }
                    )
                    continue
                claims.append(
                    {
                        "model": model,
                        "size": size,
                        "model_variant": variant,
                        **column,
                        "value": number.rstrip("%"),
                        "pages": pages,
                        "labels": list(
                            dict.fromkeys(re.findall(r"\[(S\d+)\]", caption + " " + " ".join(row)))
                        ),
                        "metric_explicit": bool(
                            re.search(
                                r"accuracy|accuracies|pass\s*@\s*1",
                                normalize(context + " " + headers[i]),
                            )
                        ),
                    }
                )
    return claims, role_entities, unparsed


def _role_list_entities(answer):
    entities = []
    lines = answer.splitlines()
    model_list = False
    for line in lines:
        text = normalize(line)
        if re.search(
            r"(?:models?|families|variants).{0,35}(?:evaluated|include|were)|evaluated.{0,35}(?:models?|families)",
            text,
        ):
            model_list = True
        if model_list and re.match(r"^\s*(?:\d+[.)]|[-*])\s+", line):
            entities.append(re.sub(r"^\s*(?:\d+[.)]|[-*])\s+", "", line).split("[S", 1)[0])
        elif model_list and not line.strip():
            model_list = False
        inline = re.search(
            r"(?:models?|families|variants)\s*(?:were\s+)?(?:evaluated\s*)?(?:include|including|are)\s+(.+)|evaluated.{0,25}models?,\s*including\s+(.+)",
            line,
            re.I,
        )
        if inline:
            listing = next(group for group in inline.groups() if group is not None).split("[S", 1)[
                0
            ]
            entities.extend(
                part.strip(" .*") for part in re.split(r",|\band\b", listing) if part.strip(" .* ")
            )
    return entities


def models_results_quality(answer, reference):
    gold = reference["numeric_reference_tuples"]
    claims, role_entities, unparsed = numeric_table_claims(answer)
    role_entities.extend(_role_list_entities(answer))
    wrong_roles = []
    for entity in role_entities:
        clean = normalize(entity).strip(" .*:")
        if re.search(r"humaneval|mbpp|machine\s+translation", clean) or re.fullmatch(
            r"[\d., %]+|\d+(?:\.\d+)?[bm](?:-a\d+b)?", clean
        ):
            wrong_roles.append(entity)
        for curation in reference.get("curation_roles", []):
            if (
                curation["evaluated_in_results_tables"] is False
                and normalize(curation["entity"]) == clean
            ):
                wrong_roles.append(entity)
    wrong_roles = list(dict.fromkeys(wrong_roles))
    # Natural explanatory text establishes family coverage only where it assigns evaluation roles.
    model_prose = " ".join(
        line
        for line in prose_only(answer).splitlines()
        if re.search(r"models?|families|evaluat|variants|base|instruct", line, re.I)
    )
    expected = {family(x["model"]) for x in reference["evaluated_model_entries"]}
    present = {family(entity) for entity in role_entities}
    for entry in reference["evaluated_model_entries"]:
        candidate = family(entry["model"])
        token = re.escape(candidate).replace(r"\.", r"[.\s-]?")
        if re.search(token, normalize(model_prose)):
            present.add(candidate)
    missing = sorted(expected - present)
    checks = []
    for claim in claims:
        matches = [
            fact
            for fact in gold
            if family(fact["model"]) == family(claim["model"])
            and normalize(fact["size"]) == normalize(claim["size"] or "")
            and all(
                fact[field] == claim[field]
                for field in ("benchmark", "language", "condition", "metric")
            )
            and (claim["model_variant"] is None or claim["model_variant"] == fact["model_variant"])
        ]
        exact_model = [
            fact for fact in matches if _model_key(fact["model"]) == _model_key(claim["model"])
        ]
        generic_family = _model_key(claim["model"]) == _model_key(family(claim["model"]))
        matches = exact_model or (matches if generic_family else [])
        supported = [
            fact
            for fact in matches
            if float(fact["value"]) == float(claim["value"])
            and (not claim["pages"] or fact["page"] in claim["pages"])
        ]
        checks.append(
            {
                "claim": claim,
                "source_matches": supported,
                "passed": bool(supported) and claim["metric_explicit"] and bool(claim["size"]),
                "reason": None
                if supported
                else "Numerical value or model/size/benchmark/language/condition relation is unsupported by the independent PDF reference",
            }
        )
    covered_benchmarks = {claim["benchmark"] for claim in claims}
    covered_languages = {claim["language"] for claim in claims if claim["metric"] == "pass@1"}
    variants = {claim["model_variant"] for claim in claims}
    sufficient = (
        len(claims) >= 8
        and {"Native", "Synthetic"}.issubset(covered_benchmarks)
        and any(c["metric"] == "pass@1" for c in claims)
        and {"English", "Arabic"}.issubset(covered_languages)
        and {"base", "instruct"}.issubset(variants)
    )
    return {
        "kind": "models-results",
        "evaluated_families_missing": missing,
        "entities_wrongly_classified_as_models": wrong_roles,
        "numeric_claims": checks,
        "unparsed_numeric_relations": unparsed,
        "substantial_stem_and_code_results": sufficient,
        "passed": not missing
        and not wrong_roles
        and sufficient
        and bool(checks)
        and all(c["passed"] for c in checks)
        and not unparsed,
        "note": "Requires model-role coverage and bound numerical relations, not answer-wide model names or numeral presence. Natural result prose not parsed by this conservative oracle needs manual review.",
    }


def prm_quality(answer, result, reference):
    text = normalize(prose_only(answer))
    sentences = re.split(r"(?<=[.!?])\s+|\n", prose_only(answer))
    affirmative = []
    for raw in sentences:
        sentence = normalize(raw)
        has_claim = _relation(
            sentence,
            r"introduc\w*|present\w*|propos\w*|develop\w*",
            r"(?:arabic.{0,45}(?:prm|process reward model)|(?:prm|process reward model).{0,45}arabic)",
            100,
        )
        negated = re.search(
            r"not|no direct|insufficient|doesn.t|cannot|can.t|whether|does the|no evidence",
            sentence,
        )
        if has_claim and not negated:
            affirmative.append(raw)
    direct = reference.get("direct_claim_evidence", [])
    cited_direct = any(
        e.get("claim") == "introduces_arabic_prm"
        and e.get("direct") is True
        and any(
            c.get("document_id") == e.get("document_id")
            and c.get("page") == e.get("page")
            and c.get("quote") == e.get("quote")
            for c in result.get("citations", [])
        )
        for e in direct
    )
    insufficient = bool(
        re.search(
            r"(?:retrieved|available|provided|supporting).{0,50}evidence.{0,65}(?:does not establish|insufficient|not sufficient|does not support)|(?:insufficient|no direct|not enough).{0,50}(?:retrieved|available|provided|supporting).{0,30}(?:evidence|passages)|cannot.{0,35}establish.{0,90}(?:retrieved|available|provided).{0,30}evidence",
            text,
        )
    )
    distinguishes_absence = bool(
        re.search(
            r"(?:not|does not|cannot).{0,45}(?:prove|proof|establish|conclude).{0,90}(?:absence|absent|anywhere|whole paper|entire paper|throughout|full paper)|not proof.{0,80}absence|not.{0,30}proof.{0,80}(?:paper|document)",
            text,
        )
    )
    absolute_negative = bool(
        re.search(
            r"(?:paper|3lm).{0,35}(?:never|does not|doesn't).{0,30}(?:introduce|mention|discuss).{0,65}(?:prm|process reward)|(?:nowhere|anywhere).{0,50}(?:paper|document).{0,40}(?:prm|reward)",
            text,
        )
    )
    positive_supported = bool(affirmative) and cited_direct
    refusal_supported = (
        not affirmative
        and insufficient
        and distinguishes_absence
        and not absolute_negative
        and not direct
    )
    return {
        "kind": "prm-claim",
        "positive_claims": affirmative,
        "direct_positive_claim_evidence": cited_direct,
        "retrieved_evidence_insufficiency_explained": insufficient,
        "absence_vs_proof_of_absence_distinguished": distinguishes_absence,
        "unqualified_whole_paper_negative": absolute_negative,
        "passed": positive_supported or refusal_supported,
        "note": "Direct-claim annotations are independently reviewed source evidence; matching an unrelated exact quote never establishes entailment.",
    }


def cot_quality(answer, reference):
    text = normalize(prose_only(answer))
    control = reference.get("control_reference", {})
    table_relations = []
    for table in markdown_tables(answer):
        headers = [normalize(header) for header in table["headers"]]
        if "model" not in headers or "size" not in headers:
            continue
        mi, si = headers.index("model"), headers.index("size")
        standard = next(
            (i for i, header in enumerate(headers) if "gsm8k" in header and "standard" in header),
            None,
        )
        cot = next(
            (
                i
                for i, header in enumerate(headers)
                if "gsm8k" in header and re.search(r"\bcot\b|chain.of.thought", header)
            ),
            None,
        )
        if standard is None or cot is None:
            continue
        for row in table["rows"]:
            if (
                len(row) == len(headers)
                and normalize(row[mi]) == "palm"
                and normalize(row[si]) == "540b"
            ):
                values = [
                    re.sub(r"\[S\d+\]", "", row[i]).strip().rstrip("%") for i in (standard, cot)
                ]
                table_relations.append(
                    {
                        "model": row[mi],
                        "size": row[si],
                        "benchmark": "GSM8K",
                        "standard": values[0],
                        "chain-of-thought": values[1],
                        "passed": values
                        == [
                            control.get("standard", "17.9"),
                            control.get("chain-of-thought", "56.9"),
                        ],
                    }
                )
    # A relation, not a bag of numbers: same model+size+task, direction and prompt conditions.
    subject = bool(
        re.search(r"palm.{0,20}540\s*b.{0,140}gsm8k|gsm8k.{0,140}palm.{0,20}540\s*b", text)
    )
    ordered = bool(
        re.search(
            r"(?:from\s+)?17[.]9\s*%?.{0,70}(?:standard|without|baseline).{0,90}(?:to|versus|vs[.]?)?\s*56[.]9\s*%?.{0,70}(?:chain.of.thought|cot)|(?:standard|without|baseline).{0,55}17[.]9\s*%?.{0,90}(?:chain.of.thought|cot).{0,55}56[.]9|from\s+17[.]9\s*%?\s+to\s+56[.]9\s*%?.{0,80}(?:chain.of.thought|cot)",
            text,
        )
    )
    not_reversed = not bool(
        re.search(r"(?:standard|baseline).{0,35}56[.]9|(?:chain.of.thought|cot).{0,35}17[.]9", text)
    )
    no_forced_list = not numbered_items(answer)
    prose_passed = subject and ordered and not_reversed
    table_passed = bool(table_relations) and all(relation["passed"] for relation in table_relations)
    return {
        "kind": "other-paper-control",
        "model_size_task_relation": subject,
        "standard_to_cot_values": ordered,
        "no_reversed_conditions": not_reversed,
        "no_forced_contribution_list": no_forced_list,
        "table_numeric_relations": table_relations,
        "passed": (prose_passed or table_passed) and no_forced_list,
        "source_reference": reference.get("control_reference", {}),
    }


def citation_validity(result, reference, source_pages=None):
    citations = result.get("citations", [])
    evidence = {entry["chunk"]["chunk_id"]: entry["chunk"] for entry in result.get("evidence", [])}
    if isinstance(source_pages, list):
        source_pages = {reference["sha256"]: source_pages}
    source_pages = source_pages or {}
    checks = []
    for citation in citations:
        pages = source_pages.get(citation.get("document_id"), [])
        page = citation.get("page")
        quote = citation.get("quote", "")
        pdf_exact = (
            isinstance(page, int)
            and 1 <= page <= len(pages)
            and bool(quote)
            and quote in pages[page - 1]
        )
        chunk = evidence.get(citation.get("chunk_id"))
        chunk_exact = (
            chunk is not None
            and bool(quote)
            and quote in chunk["text"]
            and chunk.get("document_id") == citation.get("document_id")
            and chunk.get("page") == page
        )
        checks.append(
            {
                "label": citation.get("label"),
                "page": page,
                "exact_pdf_quote": pdf_exact,
                "exact_selected_chunk_quote": chunk_exact,
                "chunk_check_available": bool(evidence),
                "passed": pdf_exact and (chunk_exact or not evidence),
            }
        )
    labels = {c.get("label") for c in citations}
    markers = set(re.findall(r"\[(S\d+)\]", result.get("answer", "")))
    return {
        "checks": checks,
        "markers_match_citations": markers == labels,
        "nonempty": bool(citations),
        "pdf_sources_available": bool(source_pages),
        "passed": bool(source_pages) and all(c["passed"] for c in checks) and markers == labels,
        "note": "Provenance only; does not establish entailment or answer completeness.",
    }


def source_support_alignment(quality, result, reference):
    """Link independently interpreted claims to the relevant cited author/table spans.

    The gold establishes source meaning/relations; matching a quote is only the final
    provenance link to that annotation and cannot make wrong prose become entailed.
    """
    citations = result.get("citations", [])
    if quality["kind"] == "author-contributions":
        aligned = {}
        for number, text in numbered_items(result.get("answer", "")):
            labels = set(re.findall(r"\[(S\d+)\]", text))
            fragments = reference.get("contribution_source_fragments", {}).get(str(number), [])
            aligned[str(number)] = bool(fragments) and all(
                any(
                    c.get("label") in labels
                    and c.get("document_id") == reference["sha256"]
                    and c.get("page") == fragment["page"]
                    and normalize(fragment["text"]) in normalize(c.get("quote", ""))
                    for c in citations
                )
                for fragment in fragments
            )
        return {"per_authored_item": aligned, "passed": len(aligned) == 3 and all(aligned.values())}
    if quality["kind"] == "models-results":
        aligned = []
        for annotation in quality["numeric_claims"]:
            claim = annotation["claim"]
            support = any(
                c.get("document_id") == reference["sha256"]
                and c.get("page") == fact["page"]
                and (not claim["labels"] or c.get("label") in claim["labels"])
                and any(
                    float(n) == float(fact["value"])
                    for n in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", c.get("quote", ""))
                )
                for fact in annotation["source_matches"]
                for c in citations
            )
            aligned.append({"claim": claim, "source_table_page_and_value_cited": support})
        return {
            "numeric_claim_support": aligned,
            "passed": bool(aligned)
            and all(item["source_table_page_and_value_cited"] for item in aligned),
            "note": "Independent gold verifies table-cell relations; cited source page/value then links the claim to that reviewed table. Exact quote identity alone is not an entailment check.",
        }
    if quality["kind"] == "other-paper-control":
        control = reference["control_reference"]
        citations = [
            c
            for c in citations
            if c.get("document_id") == reference["sha256"] and c.get("page") == control["page"]
        ]
        supported = all(
            any(str(control[condition]) in c.get("quote", "") for c in citations)
            for condition in ("standard", "chain-of-thought")
        )
        return {
            "reviewed_control_table_values_cited": supported,
            "passed": supported,
            "note": "Independent source table establishes the PaLM540B/GSM8K standard/CoT relation; exact native citations link to that reviewed source page.",
        }
    return {
        "passed": quality["passed"],
        "note": "PRM uses independently annotated direct-claim evidence; control uses independently reviewed model/task/condition relations.",
    }


def evaluate_answer(question, result, reference=None, source_pages=None, expected_language=None):
    reference = reference or json.loads(REFERENCE.read_text(encoding="utf-8"))
    normalized = normalize(question)
    expected_language = expected_language or (
        "ar" if re.search(r"[\u0621-\u064a]", question) else "en"
    )
    if reference.get("control_reference"):
        quality = cot_quality(result.get("answer", ""), reference)
    elif "process reward" in normalized or "prm" in normalized:
        quality = prm_quality(result.get("answer", ""), result, reference)
    elif "مساهم" in normalized or "contribution" in normalized:
        quality = contribution_quality(result.get("answer", ""), expected_language)
    elif "models" in normalized and "results" in normalized:
        quality = models_results_quality(result.get("answer", ""), reference)
    else:
        quality = {
            "kind": "manual-review",
            "passed": False,
            "note": "This question is outside the independently annotated regression cases.",
        }
    support = source_support_alignment(quality, result, reference)
    quality["source_support_alignment"] = support
    quality["passed"] = quality["passed"] and support["passed"]
    language = language_check(result.get("answer", ""), expected_language)
    citations = citation_validity(result, reference, source_pages)
    return {
        "language_following": language,
        "answer_quality": quality,
        "citation_validity": citations,
        "generation_mode": result.get("generation_mode"),
        "warning": result.get("warning"),
        "passed": language["passed"] and quality["passed"] and citations["passed"],
        "note": "Generation mode is descriptive metadata, never an answer-quality criterion.",
    }
