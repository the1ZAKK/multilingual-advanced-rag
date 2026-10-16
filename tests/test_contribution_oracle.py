"""Independent PDF-grounded assertions, including correct citations with wrong prose."""

import copy
import hashlib
import json

import pymupdf
import pytest

from scripts.contribution_oracle import (
    SOURCE_FIXTURE,
    citation_caption_verification,
    evaluate_author_items,
    prose,
    verify_source_answer,
)

PARAPHRASES = [
    "They introduce three comprehensive benchmarks spanning STEM and code generation, "
    "constructed rigorously for quality through native Arabic content curation, synthetic "
    "generation, and translations checked by human reviewers.",
    "They extensively assess more than forty leading Arabic and multilingual language models "
    "on scientific and programming capabilities.",
    "They analyze correlations across tasks and test robustness, revealing how models behave "
    "and how different cognitive abilities relate.",
]
PROBLEM = "Arabic model evaluation lacks native scientifically grounded STEM and code benchmarks."


def answer_for(claims, *, problem=False, numbers=None):
    numbers = numbers or list(range(1, len(claims) + 1))
    lines = [f"Problem: {PROBLEM} [S3] (page 1)."] if problem else []
    for number, claim in zip(numbers, claims, strict=True):
        suffix = "[S1] [S2] (pages 1, 2)." if number == 1 else "[S2] (page 2)."
        lines.append(f"{number}. {claim.rstrip('.')} {suffix}")
    return "\n".join(lines)


@pytest.fixture
def pdf_result(tmp_path):
    source = json.loads(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    path = tmp_path / "3LM-gold-passage.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.insert_text((50, 40), PROBLEM, fontsize=8)
        page.insert_text((50, 100), source["page1"], fontsize=8)
        page = pdf.new_page(width=600, height=800)
        page.insert_text((50, 60), source["page2"], fontsize=8)
        pdf.save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    fragments = [
        ("S1", 1, source["page1"].strip()),
        ("S2", 2, source["page2"].split("Second,", 1)[0].strip()),
        ("S2", 2, "Second," + source["page2"].split("Second,", 1)[1].split("Third,", 1)[0].rstrip()),
        ("S2", 2, "Third," + source["page2"].split("Third,", 1)[1].rstrip()),
    ]
    citations = [
        {
            "label": label,
            "page": page,
            "quote": quote,
            "document_id": digest,
            "document_name": path.name,
            "chunk_id": f"{digest}:p{page}",
        }
        for label, page, quote in fragments
    ]
    claims = [
        prose(source["page1"].split("First,", 1)[1] + fragments[1][2]),
        prose(fragments[2][2]),
        prose(fragments[3][2]),
    ]
    result = {
        "answer": answer_for(claims),
        "citations": citations,
        "generation_mode": "source-contributions",
        "warning": None,
    }
    return path, result, claims


def test_full_source_rendered_contributions_have_individual_gold_support(pdf_result):
    path, result, _ = pdf_result
    checks = verify_source_answer(result, path, require_problem=False)
    assert checks["passed"]
    assert checks["gold_passage_matches_pdf_pages_1_2"]
    assert checks["full_author_statement_support_per_item"] == {"1": True, "2": True, "3": True}
    assert all(item["passed"] for item in checks["contribution_items"].values())


@pytest.mark.parametrize("mode", ["ollama", "source-contributions"])
def test_faithful_paraphrases_pass_without_matching_source_sentences(pdf_result, mode):
    path, result, _ = pdf_result
    result["answer"] = answer_for(PARAPHRASES)
    result["generation_mode"] = mode
    assert not any(prose(c["quote"]) in prose(result["answer"]) for c in result["citations"])
    checks = verify_source_answer(result, path, require_problem=False)
    assert checks["passed"]
    assert checks["numbering"] == [1, 2, 3]


def test_keyword_rich_benchmark_components_do_not_become_authored_contributions(pdf_result):
    path, result, _ = pdf_result
    claims = [
        "Native Arabic benchmark for STEM and code with rigorous content curation, synthetic "
        "generation, and translation with human verification",
        "Synthetic benchmark for evaluation of over 40 Arabic and multilingual LLMs in "
        "scientific and programming domains",
        "Translated code benchmark with analysis of cross-task correlations and robustness "
        "testing revealing model behavior and relationships between cognitive capabilities",
    ]
    result["answer"] = answer_for(claims)
    checks = verify_source_answer(result, path, require_problem=False)
    assert all(checks["exact_pdf_quotes"])
    assert all(checks["full_author_statement_support_per_item"].values())
    assert not checks["passed"]
    assert all(
        not item["assertions"]["author_contribution_not_benchmark_component"]
        for item in checks["contribution_items"].values()
    )


@pytest.mark.parametrize(
    ("item_index", "old", "new", "missing_assertion"),
    [
        (0, "native Arabic content curation, ", "", "native_arabic_curation"),
        (0, "synthetic generation, and ", "", "synthetic_generation"),
        (0, "translations checked by human reviewers", "translations", "translation_with_human_verification"),
        (1, "more than forty", "exactly forty", "more_than_forty_models"),
        (1, "Arabic and multilingual", "Arabic", "arabic_and_multilingual_models"),
        (1, "scientific and programming", "scientific", "scientific_and_programming_evaluation"),
        (2, " and test robustness", "", "robustness_testing"),
        (2, "correlations across tasks", "scores", "analysis_of_cross_task_correlations"),
        (2, "how models behave and ", "", "insights_into_model_behavior"),
        (2, " and how different cognitive abilities relate", "", "relationships_between_cognitive_capabilities"),
    ],
)
def test_each_essential_source_assertion_must_appear_in_its_item(
    pdf_result, item_index, old, new, missing_assertion
):
    path, result, _ = pdf_result
    claims = PARAPHRASES.copy()
    assert old in claims[item_index]
    claims[item_index] = claims[item_index].replace(old, new)
    result["answer"] = answer_for(claims)
    checks = verify_source_answer(result, path, require_problem=False)
    assert all(checks["exact_pdf_quotes"])
    assert not checks["passed"]
    assert missing_assertion in checks["contribution_items"][str(item_index + 1)][
        "missing_or_wrong_assertions"
    ]


@pytest.mark.parametrize("order", [(1, 0, 2), (0, 2, 1), (0, 1, 1)])
def test_swapped_or_duplicated_prose_fails_even_with_all_correct_source_quotes(pdf_result, order):
    path, result, _ = pdf_result
    result["answer"] = answer_for([PARAPHRASES[i] for i in order])
    checks = verify_source_answer(result, path, require_problem=False)
    assert all(checks["exact_pdf_quotes"])
    assert checks["exactly_three_contributions"]
    assert not checks["passed"]


@pytest.mark.parametrize("numbers", [[1, 2, 3, 4], [1, 2, 3, 3], [1, 1, 3]])
def test_fourth_or_duplicate_numbered_item_is_rejected(pdf_result, numbers):
    path, result, _ = pdf_result
    claims = PARAPHRASES + [PARAPHRASES[0]] if len(numbers) == 4 else PARAPHRASES
    result["answer"] = answer_for(claims, numbers=numbers)
    checks = verify_source_answer(result, path, require_problem=False)
    assert not checks["exactly_three_contributions"]
    assert not checks["passed"]


def test_negated_keyword_complete_evaluation_is_not_an_authored_contribution(pdf_result):
    path, result, _ = pdf_result
    claims = PARAPHRASES.copy()
    claims[1] = claims[1].replace("They extensively assess", "They do not extensively assess")
    result["answer"] = answer_for(claims)
    checks = verify_source_answer(result, path, require_problem=False)
    assert not checks["contribution_items"]["2"]["passed"]
    assert not checks["passed"]


def test_problem_is_separate_and_contributions_remain_one_to_three(pdf_result):
    path, result, _ = pdf_result
    result["citations"].append(
        {**result["citations"][0], "label": "S3", "quote": PROBLEM}
    )
    result["answer"] = answer_for(PARAPHRASES, problem=True)
    checks = verify_source_answer(result, path, require_problem=True)
    assert checks["passed"] and checks["problem_evaluation_gap"]
    result["answer"] = answer_for([PROBLEM, *PARAPHRASES])
    assert not verify_source_answer(result, path, require_problem=True)["passed"]


def test_page_break_continuation_and_exact_source_citations_are_required(pdf_result):
    path, result, _ = pdf_result
    truncated = copy.deepcopy(result)
    del truncated["citations"][1]
    checks = verify_source_answer(truncated, path, require_problem=False)
    assert not checks["full_author_statement_support_per_item"]["1"]
    assert not checks["passed"]
    altered = copy.deepcopy(result)
    altered["citations"][0]["quote"] = altered["citations"][0]["quote"].replace("\n", " ")
    assert not verify_source_answer(altered, path, require_problem=False)["passed"]


def test_citation_captions_deduplicate_labels_without_discarding_distinct_quotes(pdf_result):
    _, result, _ = pdf_result
    captions = [
        f"[{label}] {result['citations'][0]['document_name']}, page {page}"
        for label, page in (("S1", 1), ("S2", 2))
    ]
    assert len(result["citations"]) == 4
    assert citation_caption_verification(captions, result)["passed"]
    assert not citation_caption_verification([*captions, captions[1]], result)["passed"]


def test_whole_answer_keywords_cannot_substitute_for_item_semantics():
    checks = evaluate_author_items(
        "1. Benchmarks for STEM and code [S1]\n"
        "2. 40 Arabic multilingual evaluation [S2]\n"
        "3. Cross-task correlations robustness model behavior cognitive relationships [S2]"
    )
    assert checks["exactly_three_contributions"]
    assert not checks["passed"]
