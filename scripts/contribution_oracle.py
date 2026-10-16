"""Evaluation-only 3LM assertions annotated against PDF pages 1–2.

This conservative regression oracle is not a general semantic-entailment model.
Production must derive answers from sources rather than these paper-specific rules.
"""

import hashlib
import json
import re
from pathlib import Path

import pymupdf

SOURCE_FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/3lm-contributions.json"


def prose(text):
    """Normalize display prose only; exact citation checks use original PDF text."""
    text = re.sub(r"(?<=\w)-\s*\n(?=\w)", "", text)
    text = re.sub(r"\[S\d+\]|\(pages?\s+[\d, ]+\)", " ", text)
    return " ".join(text.casefold().split())


def matches(pattern, text):
    return bool(re.search(pattern, text, re.IGNORECASE))


def numbered_items(answer):
    markers = list(re.finditer(r"(?m)^\s*(\d+)[.)]\s+", answer))
    items = [
        {
            "number": int(marker.group(1)),
            "text": answer[marker.end() : markers[i + 1].start() if i + 1 < len(markers) else None]
            .strip(),
        }
        for i, marker in enumerate(markers)
    ]
    prefix = answer[: markers[0].start()].strip() if markers else answer.strip()
    return prefix, items


def evaluate_author_items(answer):
    """Check meaning and completeness in each numbered item, not global word coverage.

    The independent annotations require each contribution's action/object and its
    construction/analysis qualifiers. Unrecognized paraphrases need manual review.
    """
    prefix, items = numbered_items(answer)
    exact_numbering = [item["number"] for item in items] == [1, 2, 3]
    annotations = {}
    for position, item in enumerate(items[:3], 1):
        text = prose(item["text"])
        component_claim = matches(
            r"^(?:(?:we|they|the authors)\s+(?:present|introduce|offer|provide|release)\s+)?"
            r"(?:(?:the|an?|3lm's)\s+)?(?:native|synthetic|translated|coding|code-generation)"
            r"\b.{0,60}\b(?:benchmark|dataset|component)\b",
            text,
        )
        shared = {
            "author_contribution_not_benchmark_component": not component_claim,
            "no_negated_or_nested_numbered_claim": not matches(
                r"\b(?:not|never|without|no)\b|^\d+[.)]\s", text
            ),
        }
        if position == 1:
            checks = {
                "complete_three_benchmark_suite": matches(
                    r"\b(?:present\w*|introduc\w*|releas\w*|provid\w*|creat\w*|"
                    r"construct\w*|build\w*|built|offer\w*)\b.{0,100}\b(?:three|3)\b"
                    r".{0,65}\bbenchmarks?\b|\b(?:three|3)\b.{0,65}\bbenchmarks?\b"
                    r".{0,60}\b(?:constructed|built|created|introduced|presented)\b",
                    text,
                ),
                "stem_and_code_scope": matches(r"\b(?:stem|science|scientific)\b", text)
                and matches(r"\b(?:code|programming)\b", text),
                "quality_construction": matches(
                    r"\b(?:rigorous|careful|quality|authentic\w*|validated|verified)\b", text
                ),
                "native_arabic_curation": matches(
                    r"\bnative\s+arabic\b.{0,65}\b(?:curat\w*|collect\w*)\b|"
                    r"\b(?:curat\w*|collect\w*)\b.{0,65}\bnative\s+arabic\b",
                    text,
                ),
                "synthetic_generation": matches(
                    r"\bsynthetic\b.{0,40}\bgenerat\w*\b|"
                    r"\bgenerat\w*\b.{0,40}\bsynthetic\b",
                    text,
                ),
                "translation_with_human_verification": matches(
                    r"\btranslat\w*\b.{0,70}\bhuman\b.{0,35}"
                    r"\b(?:verif\w*|review\w*|check\w*|validat\w*)\b|"
                    r"\bhuman(?:-|\s)+(?:verif\w*|review\w*|check\w*|validat\w*)"
                    r".{0,45}\btranslat\w*\b|\bhumans?\b.{0,30}"
                    r"\b(?:verif\w*|review\w*|check\w*|validat\w*)\b.{0,45}\btranslat\w*\b",
                    text,
                ),
            }
        elif position == 2:
            checks = {
                "evaluation_of_models": matches(
                    r"\b(?:evaluat\w*|assess\w*|compar\w*|test\w*|benchmark\w*)\b"
                    r".{0,125}\b(?:llms?|(?:language\s+)?models?)\b|"
                    r"\b(?:llms?|(?:language\s+)?models?)\b.{0,125}"
                    r"\b(?:evaluat\w*|assess\w*|compar\w*|test\w*)\b",
                    text,
                ),
                "more_than_forty_models": matches(
                    r"\b(?:over|more than|greater than|exceed\w*)\s+(?:40|forty)\b|"
                    r"\b(?:40|forty)(?:\s*\+|(?:-|\s)+plus\b)",
                    text,
                ),
                "arabic_and_multilingual_models": matches(
                    r"\b(?:arabic\b.{0,55}\bmultilingual|multilingual\b.{0,55}\barabic)"
                    r"\b.{0,50}\b(?:llms?|(?:language\s+)?models?)\b",
                    text,
                ),
                "scientific_and_programming_evaluation": matches(
                    r"\b(?:scientific|science|stem)\b", text
                )
                and matches(r"\b(?:programming|code)\b", text),
            }
        else:
            checks = {
                "analysis_of_cross_task_correlations": matches(
                    r"\b(?:analys[ie]\w*|analy[sz]\w*|investigat\w*|examin\w*)\b"
                    r".{0,140}\bcorrelations?\b",
                    text,
                )
                and matches(
                    r"\b(?:cross-task|inter-task)\b.{0,30}\bcorrelations?\b|"
                    r"\bcorrelations?\b.{0,40}\b(?:across|between|among)\b"
                    r".{0,25}\btasks\b",
                    text,
                ),
                "robustness_testing": matches(
                    r"\brobustness\b.{0,30}\b(?:test\w*|analys[ie]\w*|check\w*)\b|"
                    r"\b(?:test\w*|analys[ie]\w*|check\w*)\b.{0,100}\brobustness\b",
                    text,
                ),
                "insights_into_model_behavior": matches(
                    r"\bmodels?\s+behaviou?r\b|\b(?:how|the way)\b.{0,20}"
                    r"\bmodels?\s+behav\w*\b",
                    text,
                ),
                "relationships_between_cognitive_capabilities": matches(
                    r"\b(?:relationship\w*|relation\w*|interplay|links?)\b.{0,85}"
                    r"\bcognitive\s+(?:capabilities|abilities)\b|"
                    r"\bcognitive\s+(?:capabilities|abilities)\b.{0,65}"
                    r"\b(?:relat\w*|interact\w*|link\w*)\b",
                    text,
                ),
            }
        checks = {**shared, **checks}
        annotations[str(position)] = {
            "number": item["number"],
            "text": item["text"],
            "assertions": checks,
            "missing_or_wrong_assertions": [name for name, valid in checks.items() if not valid],
            "passed": all(checks.values()),
        }
    return {
        "problem_text": prefix,
        "numbering": [item["number"] for item in items],
        "exactly_three_contributions": exact_numbering,
        "contribution_items": annotations,
        "passed": exact_numbering and all(item["passed"] for item in annotations.values()),
    }


def citation_caption_verification(captions, result):
    """Display one caption per source while retaining distinct verified source quotes."""
    expected = list(
        dict.fromkeys(
            f"[{citation['label']}] {citation['document_name']}, page {citation['page']}"
            for citation in result["citations"]
        )
    )
    return {
        "displayed": captions,
        "expected_unique_sources": expected,
        "passed": captions == expected and len(captions) == len(set(captions)),
    }


def verify_source_answer(result, path, *, require_problem):
    """Check the PDF gold passage, immutable citations, and full meaning of each item."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pymupdf.open(path) as pdf:
        pages = [page.get_text("text", sort=False) for page in pdf]
    fixture = json.loads(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    source_matches = len(pages) >= 2 and all(
        fixture[f"page{i}"].strip() in pages[i - 1] for i in (1, 2)
    )
    citation_checks = [
        citation["document_id"] == digest
        and isinstance(citation["page"], int)
        and 1 <= citation["page"] <= len(pages)
        and citation["quote"] in pages[citation["page"] - 1]
        and f"[{citation['label']}]" in result["answer"]
        for citation in result["citations"]
    ]
    semantic = evaluate_author_items(result["answer"])
    gold_fragments = {
        "1": [
            (1, fixture["page1"].split("First,", 1)[1].strip()),
            (2, fixture["page2"].split("Second,", 1)[0].strip()),
        ],
        "2": [(2, fixture["page2"].split("Second,", 1)[1].split("Third,", 1)[0].strip())],
        "3": [(2, fixture["page2"].split("Third,", 1)[1].strip())],
    }
    item_sources = {}
    for number, gold in gold_fragments.items():
        item = semantic["contribution_items"].get(number, {})
        labels = set(re.findall(r"\[(S\d+)\]", item.get("text", "")))
        item_sources[number] = bool(labels) and all(
            any(
                citation["label"] in labels
                and citation["document_id"] == digest
                and citation["page"] == page
                and prose(fragment) in prose(citation["quote"])
                for citation in result["citations"]
            )
            for page, fragment in gold
        )
    problem = prose(semantic["problem_text"])
    problem_gap = not require_problem or (
        bool(problem)
        and matches(r"\b(?:gap|lack\w*|limited|shortage|scarcity|need\w*|absence|insufficient)\b", problem)
        and matches(r"\barabic\b", problem)
        and matches(r"\b(?:evaluat\w*|benchmarks?)\b", problem)
        and matches(r"\b(?:stem|scien\w*)\b", problem)
        and matches(r"\b(?:code|programming)\b", problem)
        and bool(re.search(r"\[S\d+\]", semantic["problem_text"]))
    )
    mode = result["generation_mode"]
    passed = (
        source_matches
        and bool(citation_checks)
        and all(citation_checks)
        and semantic["passed"]
        and all(item_sources.values())
        and problem_gap
        and mode in {"ollama", "source-contributions"}
    )
    return {
        "pdf_sha256": digest,
        "gold_passage_matches_pdf_pages_1_2": source_matches,
        "exact_pdf_quotes": citation_checks,
        "generation_mode": mode,
        "contribution_items": semantic["contribution_items"],
        "numbering": semantic["numbering"],
        "exactly_three_contributions": semantic["exactly_three_contributions"],
        "full_author_statement_support_per_item": item_sources,
        "problem_evaluation_gap": problem_gap,
        "passed": passed,
        "note": "Manually annotated per-item PDF assertions with conservative paraphrase patterns; "
        "this is a regression oracle, not a general entailment model. Review unfamiliar prose manually.",
    }
