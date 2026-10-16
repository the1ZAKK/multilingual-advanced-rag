"""Semantic regressions: valid source quotes can accompany an incorrect answer."""

import copy
import json
from pathlib import Path

import pytest

from scripts.grounding_language_oracle import (
    citation_validity,
    contribution_quality,
    cot_quality,
    evaluate_answer,
    language_check,
    models_results_quality,
    prm_quality,
)

FIXTURES = Path(__file__).parent / "fixtures"
ARABIC = [
    "تقديم ثلاثة معايير تقييم شاملة في مجالات العلوم وتوليد الشيفرة، بُنيت من محتوى "
    "عربي أصيل وتوليد اصطناعي وترجمة دقيقة مع تحقق بشري.",
    "إجراء تقييم واسع لأكثر من ٤٠ نموذجًا لغويًا عربيًا ومتعدد اللغات لقياس القدرات "
    "العلمية والبرمجية.",
    "إجراء تحليل للارتباطات بين المهام واختبارات المتانة، لاستخلاص رؤى عن سلوك "
    "النماذج والعلاقات بين قدراتها المعرفية.",
]
PARAPHRASE = [
    "طرح ثلاث مجموعات اختبار للعلوم والبرمجة، مع بناء المحتوى من مصادر عربية أصلية "
    "وتوليد الأسئلة اصطناعيا، وترجمة يفحصها البشر.",
    "فحص ما يزيد على أربعين نموذجا عربيا ومتعدد اللغات لتقييم الأداء العلمي والبرمجي.",
    "دراسة الترابط عبر المهام وفحص ثبات النماذج، لفهم سلوك النماذج والعلاقات بين "
    "قدراتها الإدراكية.",
]
REFUSAL = (
    "The retrieved evidence does not establish that the paper introduces an Arabic process "
    "reward model. Available passages describe benchmark evaluation [S1] (page 1). "
    "This is not proof of absence throughout the full paper."
)


@pytest.fixture
def reference():
    return json.loads((FIXTURES / "grounding-language-reference.json").read_text(encoding="utf-8"))


def answer_for(items=ARABIC):
    return "\n".join(f"{i}. {item} [S1] (ص 1-2)." for i, item in enumerate(items, 1))


@pytest.fixture
def source_result(reference):
    text = "We introduce a benchmark suite for Arabic STEM and code evaluation."
    result = {
        "answer": REFUSAL,
        "generation_mode": "ollama",
        "citations": [
            {
                "label": "S1",
                "document_id": reference["sha256"],
                "chunk_id": "source:p1",
                "page": 1,
                "quote": text,
            }
        ],
        "evidence": [
            {
                "chunk": {
                    "chunk_id": "source:p1",
                    "document_id": reference["sha256"],
                    "page": 1,
                    "text": text,
                }
            }
        ],
    }
    return result, [text]


@pytest.mark.parametrize("items", [ARABIC, PARAPHRASE])
def test_complete_arabic_authored_meanings_allow_nonliteral_paraphrases(items):
    answer = answer_for(items)
    checks = contribution_quality(answer, "ar")
    assert checks["passed"], checks
    assert language_check(answer, "ar")["passed"]


@pytest.mark.parametrize(
    ("position", "old", "new", "missing"),
    [
        (0, "محتوى عربي أصيل", "محتوى مترجم", "native_arabic_curation"),
        (0, "وتوليد اصطناعي", "", "synthetic_generation"),
        (0, "وترجمة دقيقة مع تحقق بشري", "", "careful_translation_with_human_verification"),
        (1, "أكثر من ٤٠", "٤٠ فقط", "over_forty_models"),
        (1, "ومتعدد اللغات", "", "arabic_and_multilingual_models"),
        (1, "والبرمجية", "", "scientific_and_programming_capabilities"),
        (2, "للارتباطات بين المهام", "لدرجات الأداء", "cross_task_correlation_analysis"),
        (2, "واختبارات المتانة", "", "robustness_testing"),
        (2, "عن سلوك النماذج و", "عن ", "model_behavior_insights"),
        (2, "والعلاقات بين قدراتها المعرفية", "", "relationships_between_cognitive_capabilities"),
    ],
)
def test_each_authored_action_and_qualifier_is_required_in_its_own_item(
    position, old, new, missing
):
    items = ARABIC.copy()
    items[position] = items[position].replace(old, new)
    result = contribution_quality(answer_for(items), "ar")
    assert not result["passed"]
    assert missing in result["items"][position]["missing_or_wrong"]


def test_components_and_four_items_do_not_replace_three_authored_contributions():
    components = [
        "تقديم معيار عربي أصيل للعلوم وتوليد الشيفرة.",
        "تقديم معيار اصطناعي لأسئلة العلوم والبرمجة.",
        "تقديم معيار مترجم لتوليد الشيفرة مع تحقق بشري.",
        "نشر مجموعة البيانات مجانا.",
    ]
    check = contribution_quality(answer_for(components), "ar")
    assert not check["passed"] and not check["exactly_three_authored_items"]
    assert not contribution_quality(answer_for(components[:3]), "ar")["passed"]


def test_correct_english_contributions_still_fail_arabic_language_request(reference, source_result):
    result, pages = source_result
    result["answer"] = (
        "1. They introduce three comprehensive STEM and code benchmarks, constructed rigorously "
        "with native Arabic content curation, synthetic generation and translations checked by "
        "human reviewers [S1].\n"
        "2. They evaluate over 40 Arabic and multilingual language models in scientific and "
        "programming domains [S1].\n"
        "3. They analyze cross-task correlations and test robustness, revealing model behavior "
        "and relationships between cognitive capabilities [S1]."
    )
    check = evaluate_answer(reference["questions"][0], result, reference, pages)
    assert check["citation_validity"]["passed"]
    assert not check["language_following"]["passed"]
    assert not check["passed"]


def test_swapping_duplicate_items_and_negating_qualifiers_fail():
    assert not contribution_quality(answer_for([ARABIC[1], ARABIC[0], ARABIC[2]]), "ar")["passed"]
    assert not contribution_quality(answer_for([ARABIC[0], ARABIC[1], ARABIC[1]]), "ar")["passed"]
    items = ARABIC.copy()
    items[0] = items[0].replace("مع تحقق بشري", "دون تحقق بشري")
    assert not contribution_quality(answer_for(items), "ar")["passed"]


@pytest.mark.parametrize(
    ("position", "old", "new", "missing"),
    [
        (0, "العلوم وتوليد الشيفرة", "STEM توليد الشيفرة", "stem_and_code"),
        (1, "القدرات العلمية والبرمجية", "قدرات النماذج اللغوية العربية في علمي و برمجة المنطقة المحددة",
         "scientific_and_programming_capabilities"),
        (2, "النماذج والعلاقات", "النماذج وأو العلاقات", "joint_behavior_and_capability_findings"),
    ],
)
def test_source_terms_do_not_certify_broken_domain_or_conclusion_relations(
    position, old, new, missing
):
    # Observed translation defects retained every important individual keyword,
    # yet lost a conjunction, changed a domain role, or weakened AND into AND/OR.
    items = ARABIC.copy()
    items[position] = items[position].replace(old, new)
    check = contribution_quality(answer_for(items), "ar")
    assert not check["passed"]
    assert missing in check["items"][position]["missing_or_wrong"]


def test_independently_reviewed_source_complete_arabic_paraphrases_preserve_relations():
    items = [
        "أولًا، نقدم ثلاثة معايير تقييم شاملة تشمل مجالات STEM و توليد الشيفرة، أُنشئت باستخدام منهجيات صارمة تضمن أصالة و جودة من تنظيم المحتوى العربي الأصيل إلى توليد اصطناعي و ترجمة دقيقة مع تحقق بشري.",
        "ثانيًا، نجري تقييم واسع النطاق لـ أكثر من 40 نموذجًا لغويًا كبيرًا متطورًا من النماذج العربية والمتعددة اللغات، مما يوفر أشمل تقييم لـ قدرات النماذج اللغوية العربية في المجالات العلمية والبرمجية حتى الآن.",
        "ثالثًا، نجري تحليل شامل بما في ذلك ارتباطات بين المهام و اختبار المتانة، مما يكشف رؤى حول سلوك النموذج و علاقة بين قدرات معرفية مختلفة في النماذج اللغوية الكبيرة العربية.",
    ]
    assert contribution_quality(answer_for(items), "ar")["passed"]


def results_answer(reference):
    families = (
        "Evaluated model families include Qwen2.5, Qwen3, Jais, Fanar, Llama, AceGPT, Gemma, "
        "SILMA, aya-expanse, c4ai, ALLaM and Yehia. These include base and instruction-tuned variants.\n"
    )
    selected = [
        (1, "Qwen2.5", "72B"),
        (8, "Qwen2.5-Instruct", "72B"),
        (3, "Qwen3-Base", "14B"),
        (10, "Qwen3", "32B"),
    ]
    lines = [families]
    for number, model, size in selected:
        table = next(t for t in reference["table_rows"] if t["table"] == number)
        row = next(r for r in table["rows"] if r["model"] == model and r["size"] == size)
        if number in {1, 8}:
            headers = [
                "Native / MCQ",
                "Native / Completion",
                "Synthetic / MCQ",
                "Synthetic / Completion",
            ]
            metric = "accuracy percentages"
        else:
            headers = [
                f"{language} / {benchmark}"
                for language in ("English", "Arabic")
                for benchmark in ("HumanEval", "HumanEval+", "MBPP", "MBPP+")
            ] + ["Average / English", "Average / Arabic"]
            metric = "pass@1 percentages"
        lines.extend(
            [
                f"\nTable {number}, {table['variant']} models, {metric} [S1] (page {table['page']}).",
                "| Model | Size | " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * (len(headers) + 2)) + " |",
                "| " + " | ".join([model, size, *row["values"]]) + " |",
            ]
        )
    return "\n".join(lines)


def test_model_roles_and_exact_bound_numeric_relations_pass(reference):
    check = models_results_quality(results_answer(reference), reference)
    assert check["passed"], check
    assert len(check["numeric_claims"]) == 28


def test_metric_catalog_spacing_and_last_cell_markers_keep_all_numeric_relations(reference):
    answer = results_answer(reference)
    lines = []
    for line in answer.splitlines():
        if line.startswith("Table "):
            lines.extend(
                [
                    line,
                    "",
                    "Models and sizes in the table rows: source-derived catalog [S1]",
                    "",
                    "Metric: " + ("pass@1" if "pass@1" in line else "accuracy") + " [S1]",
                    "",
                    "Representative rows selected by column maxima, including ties [S1]",
                    "",
                ]
            )
        elif line.startswith("| Qwen"):
            lines.append(line[:-1].rstrip() + " [S1] |")
        else:
            lines.append(line)
    checks = models_results_quality("\n".join(lines), reference)
    assert checks["passed"], checks
    assert len(checks["numeric_claims"]) == 28
    assert all(check["claim"]["labels"] == ["S1"] for check in checks["numeric_claims"])


def test_models_query_cannot_pass_on_family_keywords_or_numeral_presence(reference):
    answer = "Evaluated model families include Qwen2.5 Qwen3 Jais Fanar Llama AceGPT Gemma SILMA aya c4ai ALLaM Yehia. Results were 94.45, 73.1, 69.3, 80.0."
    check = models_results_quality(answer, reference)
    assert not check["passed"]
    assert not check["substantial_stem_and_code_results"]


@pytest.mark.parametrize("entity", ["HumanEval+", "MBPP+", "Machine Translation", "865"])
def test_benchmarks_processes_and_figure_counts_are_not_evaluated_models(reference, entity):
    answer = results_answer(reference) + f"\nThe following models were evaluated:\n1. {entity}\n"
    check = models_results_quality(answer, reference)
    assert check["entities_wrongly_classified_as_models"] == [entity]
    assert not check["passed"]


def test_observed_inline_models_list_cannot_make_sizes_numbers_and_curation_models_entities(
    reference,
):
    answer = "1. The 3LM paper evaluated several models, including GPT-4o, HumanEval+, MBPP+, Machine Translation, Llama-3.1, 8B, 67.6, 63.8, QWEN, Gemma-3-pt, AceGPT-v2, and others [S1].\n2. Results were reported for code generation [S1]."
    check = models_results_quality(answer, reference)
    bad = check["entities_wrongly_classified_as_models"]
    assert all(
        entity in bad
        for entity in ["GPT-4o", "HumanEval+", "MBPP+", "Machine Translation", "8B", "67.6", "63.8"]
    )
    assert not check["passed"]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("| Qwen3-Base | 14B |", "| Qwen3-Base | 8B |"),
        ("73.1 | 69.3", "69.3 | 73.1"),
        ("70.7 | 63.4", "63.4 | 70.7"),
        ("| Qwen2.5-Instruct | 72B |", "| Qwen2.5-Instruct | 7B |"),
        ("94.45 | 62.31", "99.99 | 62.31"),
    ],
)
def test_results_are_relations_not_exact_number_keywords(reference, old, new):
    answer = results_answer(reference)
    assert old in answer
    check = models_results_quality(answer.replace(old, new), reference)
    assert not check["passed"]
    assert any(not c["passed"] for c in check["numeric_claims"])


def test_wrong_explicit_model_variant_not_repaired_by_family_match(reference):
    answer = results_answer(reference).replace("| Qwen3-Base | 14B |", "| Qwen3-Instruct | 14B |")
    assert not models_results_quality(answer, reference)["passed"]


def test_missing_metrics_or_families_are_reported(reference):
    no_metric = results_answer(reference).replace("pass@1 percentages", "code results")
    assert not models_results_quality(no_metric, reference)["passed"]
    no_family = results_answer(reference).replace("SILMA, ", "")
    assert "silma" in models_results_quality(no_family, reference)["evaluated_families_missing"]


def test_benchmark_names_are_allowed_in_benchmark_roles(reference):
    answer = (
        results_answer(reference)
        + "\nHumanEval+ and MBPP+ are benchmarks; Machine Translation is the construction process."
    )
    assert models_results_quality(answer, reference)["passed"]


def test_prm_refusal_quality_is_independent_from_mode_and_citation_validity(
    reference, source_result
):
    result, pages = source_result
    for mode in ("ollama", "extractive", "evidence-refusal", "anything"):
        result["generation_mode"] = mode
        check = evaluate_answer(reference["questions"][2], result, reference, pages)
        assert check["passed"]
        assert check["generation_mode"] == mode


def test_valid_exact_benchmark_quote_does_not_entail_positive_prm(reference, source_result):
    result, pages = source_result
    result["answer"] = "Yes. The paper introduces an Arabic process reward model [S1] (page 1)."
    check = evaluate_answer(reference["questions"][2], result, reference, pages)
    assert check["citation_validity"]["passed"]
    assert not check["answer_quality"]["passed"]
    assert check["answer_quality"]["positive_claims"]


@pytest.mark.parametrize(
    "answer",
    [
        "The paper never introduces an Arabic process reward model [S1].",
        "There is no PRM anywhere in the paper [S1].",
        "The retrieved evidence does not establish an Arabic process reward model [S1].",
    ],
)
def test_absence_of_retrieved_support_cannot_be_universal_negative_or_omit_scope(
    reference, source_result, answer
):
    result, _ = source_result
    result["answer"] = answer
    assert not prm_quality(answer, result, reference)["passed"]


def test_positive_prm_claim_passes_for_another_direct_source_not_blanket_refusal(
    source_result, reference
):
    result, pages = source_result
    direct_quote = (
        "We introduce an Arabic process reward model trained to score individual solution steps."
    )
    result["citations"][0]["quote"] = direct_quote
    result["evidence"][0]["chunk"]["text"] = direct_quote
    result["answer"] = (
        "The paper explicitly introduces an Arabic process reward model for scoring solution steps [S1] (page 1)."
    )
    alternate = copy.deepcopy(reference)
    alternate["direct_claim_evidence"] = [
        {
            "claim": "introduces_arabic_prm",
            "direct": True,
            "document_id": reference["sha256"],
            "page": 1,
            "quote": direct_quote,
        }
    ]
    check = evaluate_answer(reference["questions"][2], result, alternate, [direct_quote])
    assert check["passed"]
    result["answer"] = REFUSAL
    assert not evaluate_answer(reference["questions"][2], result, alternate, [direct_quote])[
        "answer_quality"
    ]["passed"]


def test_an_invalid_second_quote_is_not_hidden_by_duplicate_label(reference, source_result):
    result, pages = source_result
    result["citations"].append(
        {**result["citations"][0], "quote": "Fabricated reward model evidence"}
    )
    checks = citation_validity(result, reference, pages)
    assert checks["checks"][0]["passed"]
    assert not checks["checks"][1]["passed"]
    assert not checks["passed"]


def test_quote_in_pdf_but_outside_selected_chunk_is_invalid(reference, source_result):
    result, pages = source_result
    pages[0] += " This is a second actual PDF sentence."
    result["citations"][0]["quote"] = "This is a second actual PDF sentence."
    checks = citation_validity(result, reference, pages)
    assert checks["checks"][0]["exact_pdf_quote"]
    assert not checks["checks"][0]["exact_selected_chunk_quote"]
    assert not checks["passed"]


def test_complete_arabic_prose_needs_full_author_statement_support_per_item(
    reference, source_result
):
    result, pages = source_result
    result["answer"] = answer_for()
    check = evaluate_answer(reference["questions"][0], result, reference, pages)
    assert check["citation_validity"]["passed"]
    assert check["language_following"]["passed"]
    assert all(item["passed"] for item in check["answer_quality"]["items"])
    assert not check["answer_quality"]["source_support_alignment"]["passed"]
    assert not check["passed"]


def test_correct_numeric_relations_with_irrelevant_exact_quotes_lack_support(
    reference, source_result
):
    result, pages = source_result
    result["answer"] = results_answer(reference)
    check = evaluate_answer(reference["questions"][1], result, reference, pages)
    assert check["citation_validity"]["passed"]
    assert check["answer_quality"]["substantial_stem_and_code_results"]
    assert all(item["passed"] for item in check["answer_quality"]["numeric_claims"])
    assert not check["answer_quality"]["source_support_alignment"]["passed"]
    assert not check["passed"]


def test_real_other_paper_control_does_not_require_three_contributions():
    answer = "For PaLM 540B on GSM8K, standard prompting scores 17.9%, while chain-of-thought prompting scores 56.9% (Table 2, page 21)."
    check = cot_quality(
        answer,
        {
            "control_reference": {
                "page": 21,
                "model": "PaLM",
                "size": "540B",
                "task": "GSM8K",
                "standard": "17.9",
                "CoT": "56.9",
            }
        },
    )
    assert check["passed"]
    assert not cot_quality(
        answer.replace("17.9", "TEMP").replace("56.9", "17.9").replace("TEMP", "56.9"), {}
    )["passed"]
    assert not cot_quality("1. " + answer + "\n2. A second claim\n3. A third claim", {})["passed"]


def test_actual_cot_pdf_fixture_has_independent_condition_numeric_relations():
    fixture = json.loads((FIXTURES / "cot-table2-page21.json").read_text(encoding="utf-8"))
    assert "540B\n17.9 56.9" in fixture["text"]
    assert fixture["control_reference"]["standard"] == "17.9"
    assert fixture["control_reference"]["chain-of-thought"] == "56.9"
    quote = fixture["text"].split("Table 3:", 1)[0].strip()
    result = {
        "answer": "For PaLM 540B on GSM8K, standard prompting scores 17.9%, while chain-of-thought prompting scores 56.9% [S1] (Table 2, page 21).",
        "generation_mode": "source-tables",
        "citations": [
            {
                "label": "S1",
                "document_id": fixture["sha256"],
                "chunk_id": "cot:p21",
                "page": 21,
                "quote": quote,
            }
        ],
        "evidence": [
            {
                "chunk": {
                    "chunk_id": "cot:p21",
                    "document_id": fixture["sha256"],
                    "page": 21,
                    "text": quote,
                }
            }
        ],
    }
    sources = {fixture["sha256"]: [""] * 20 + [fixture["text"]]}
    check = evaluate_answer(
        "For PaLM 540B on GSM8K, compare standard and chain-of-thought prompting results. Cite the table and page.",
        result,
        fixture,
        sources,
    )
    assert check["passed"]
    result["answer"] = result["answer"].replace(
        "standard prompting scores 17.9%", "standard prompting scores 56.9%"
    )
    check = evaluate_answer("Compare PaLM 540B standard and CoT on GSM8K", result, fixture, sources)
    assert check["citation_validity"]["passed"]
    assert not check["answer_quality"]["passed"]


def test_actual_other_paper_table_output_preserves_condition_relations_without_numbered_list():
    fixture = json.loads((FIXTURES / "cot-table2-page21.json").read_text(encoding="utf-8"))
    answer = "Table 2: Standard versus chain-of-thought prompting (page 21) [S1]\n\nModels and sizes in the table rows: PaLM (540B) [S1]\n\n| Model | Size | GSM8K / standard | GSM8K / CoT |\n| --- | --- | --- | --- |\n| PaLM | 540B | 17.9 | 56.9 [S1] |"
    checks = cot_quality(answer, fixture)
    assert checks["passed"]
    assert checks["table_numeric_relations"][0]["passed"]
    assert not cot_quality(answer.replace("17.9 | 56.9", "56.9 | 17.9"), fixture)["passed"]


def test_other_paper_models_results_query_uses_its_own_independent_reference():
    fixture = json.loads((FIXTURES / "cot-table2-page21.json").read_text(encoding="utf-8"))
    quote = fixture["text"].split("Table 3:", 1)[0].strip()
    result = {
        "answer": "For PaLM 540B on GSM8K, standard prompting scores 17.9%, while chain-of-thought prompting scores 56.9% [S1] (Table 2, page 21).",
        "generation_mode": "source-tables",
        "citations": [{"label": "S1", "document_id": fixture["sha256"],
                       "chunk_id": "cot:p21", "page": 21, "quote": quote}],
        "evidence": [{"chunk": {"chunk_id": "cot:p21", "document_id": fixture["sha256"],
                                  "page": 21, "text": quote}}],
    }
    check = evaluate_answer(
        "Which models were evaluated in Chain-of-Thought Prompting Elicits Reasoning, and what numerical results were reported?",
        result, fixture, {fixture["sha256"]: [""] * 20 + [fixture["text"]]},
    )
    assert check["answer_quality"]["kind"] == "other-paper-control"
    assert check["passed"]


def test_language_check_excludes_native_source_blockquotes():
    assert language_check(answer_for() + "\n> " + "A native English citation passage. " * 30, "ar")[
        "passed"
    ]
    assert not language_check(
        "The authors evaluate Arabic language models and report results.", "ar"
    )["passed"]
