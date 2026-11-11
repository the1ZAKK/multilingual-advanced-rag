"""Development regressions with independent values and adversarial relationships."""

import json

import httpx
import pytest

from rag.generation.numerical import (
    atoms,
    strategy_bindings,
    validate_comparison_slot,
    validate_count_slot,
    validate_numeric_support,
    validate_quantity_comparison,
    validate_score_slot,
)
from rag.generation.ollama import GenerationUnavailable, semantic_items
from rag.generation.questions import question_parts
from rag.models import Chunk, Hit


def evidence(text, *, section="Abstract", page=1):
    return Hit(
        chunk=Chunk(
            chunk_id=f"source:p{page}:0",
            document_id="source",
            document_name="Independent.pdf",
            page=page,
            text=text,
            normalized=text,
            start=0,
            end=len(text),
            section=section,
        ),
        score=1,
    )


def test_exact_development_questions_have_four_required_slots():
    from pathlib import Path

    suite = json.loads(
        Path("data/evaluation/additional-paper-review.json").read_text(encoding="utf-8")
    )
    for case in suite["cases"]:
        parts = question_parts(case["question"])
        assert len(parts) == 4, (case["id"], parts)
    assert len(question_parts("How many sensors, sites, and patients were studied?")) == 3
    assert len(question_parts("كم عدد المستشعرات والمواقع والمرضى؟")) == 3
    assert len(question_parts("How do sensors and sites interact?")) == 1
    assert (
        len(question_parts("How many sensors, sites, patients, visits, and devices were used?"))
        == 5
    )


@pytest.mark.parametrize(
    "claim", ["9 models from 23 providers.", "تقييم 9 نماذج من 23 مقدمي خدمات."]
)
def test_swapped_counts_fail_even_when_all_numbers_are_in_one_real_quote(claim):
    with pytest.raises(ValueError, match="swapped counted entity"):
        validate_numeric_support(claim, ["We evaluated 23 models from 9 providers."])


def test_distinct_tasks_do_not_become_individual_evaluations():
    with pytest.raises(ValueError, match="swapped counted entity"):
        validate_numeric_support(
            "The evaluation comprises 180 individual test cases.",
            ["We used 180 tasks for 45,360 individual test cases."],
        )


@pytest.mark.parametrize(
    "claim",
    [
        "Few-shot CoT achieves 31.25% versus zero-shot at 87.5%, a 2.8x ratio.",
        "تبلغ دقة سلسلة التفكير بأمثلة قليلة 31.25% مقابل 87.5% دون أمثلة، بنسبة 2.8 ضعف.",
    ],
)
def test_swapped_condition_to_accuracy_mapping_is_rejected(claim):
    with pytest.raises(ValueError, match="swapped condition"):
        validate_numeric_support(
            claim, ["Few-shot CoT achieves 2.8x the zero-shot accuracy (87.5% vs 31.25%)."]
        )


def test_model_strategy_combinations_are_not_model_counts():
    from rag.generation.numerical import count_bindings

    assert count_bindings("281 model-strategy combinations") == set()
    assert count_bindings("210 distinct number-reading tasks") == {("task", "210")}


def test_inline_abstract_stops_before_index_terms_and_roman_sections():
    from rag.ingestion.structure import chunk_spans, section_marks

    text = (
        "Title\nAbstract—We evaluated 23 models on 48 tasks.\n"
        "Index Terms—model evaluation\nI. INTRODUCTION\n"
        "Related experiments use 900 questions.\nII. RELATED WORK\nOther findings.\n"
    )
    spans = chunk_spans(text, 900, 150, section_marks(text, None))
    abstract = "".join(text[start:end] for start, end, section in spans if section == "Abstract")
    assert "23 models" in abstract and "48 tasks" in abstract
    assert "Index Terms" not in abstract and "900" not in abstract
    assert abstract in text


def test_faithful_comparison_may_reverse_mention_order():
    validate_numeric_support(
        "Zero-shot achieves 31.25% versus few-shot CoT at 87.5%, a 2.8x ratio.",
        ["Few-shot CoT achieves 2.8x the zero-shot accuracy (87.5% vs 31.25%)."],
    )


@pytest.mark.parametrize(
    "claim",
    [
        "Few-shot CoT achieves 91% accuracy, while zero-shot achieves 35%, "
        "resulting in 2.6x higher accuracy for few-shot CoT.",
        "Few-shot CoT (few-shot chain-of-thought) achieves 91% versus zero-shot 35%, "
        "reported as 2.6x.",
        "تبلغ دقة سلسلة التفكير بأمثلة قليلة 91% مقابل 35% دون أمثلة، "
        "بنسبة 2.6x لسلسلة التفكير بأمثلة قليلة.",
    ],
)
def test_repeated_condition_names_do_not_erase_a_faithful_accuracy_mapping(claim):
    source = "Few-shot CoT achieves 2.6x the zero-shot accuracy (91% vs 35%)."
    assert strategy_bindings(claim) == {("few-cot", "91"), ("zero", "35")}
    validate_numeric_support(claim, [source])
    validate_comparison_slot("Compare few-shot CoT with zero-shot accuracy.", claim, [source])
    with pytest.raises(ValueError, match="swapped condition"):
        validate_numeric_support(claim.replace("91%", "35%").replace("35%,", "91%,"), [source])


@pytest.mark.parametrize(
    "claim",
    [
        "Few-shot CoT achieves 91% and few-shot CoT achieves 35%, compared to zero-shot.",
        "Few-shot CoT achieves 91% versus zero-shot 35%, while zero-shot CoT is another condition.",
        "Few-shot CoT achieves 91%, versus zero-shot 35% or 30%.",
    ],
)
def test_repeated_conditions_cannot_invent_a_missing_or_ambiguous_mapping(claim):
    assert not strategy_bindings(claim)
    with pytest.raises(ValueError, match="condition-to-accuracy"):
        validate_comparison_slot(
            "Compare few-shot CoT with zero-shot accuracy.",
            claim,
            ["Few-shot CoT achieves 2.6x the zero-shot accuracy (91% vs 35%)."],
        )


@pytest.mark.parametrize(
    "claim",
    [
        "Few-shot CoT and zero-shot accuracy have values of 91% and 35%, respectively, with a 2.6x ratio.",
        "تبلغ دقتا سلسلة التفكير بأمثلة قليلة ودون أمثلة 91% و35% على التوالي، بنسبة 2.6x.",
    ],
)
def test_respectively_explicitly_binds_the_ordered_condition_pair(claim):
    source = "Few-shot CoT achieves 2.6x the zero-shot accuracy (91% vs 35%)."
    assert strategy_bindings(claim) == {("few-cot", "91"), ("zero", "35")}
    validate_numeric_support(claim, [source])
    validate_comparison_slot("Compare few-shot CoT and zero-shot accuracy.", claim, [source])
    swapped = claim.replace("91%", "TMP").replace("35%", "91%").replace("TMP", "35%")
    with pytest.raises(ValueError, match="swapped condition"):
        validate_numeric_support(swapped, [source])


def test_respectively_cannot_fill_an_unnamed_condition_or_an_explicitly_negated_mapping():
    assert not strategy_bindings("Few-shot CoT has values of 91% and 35%, respectively.")
    assert not strategy_bindings("Few-shot CoT and zero-shot have 91% and 35%, not respectively.")


def test_every_numeric_citation_must_contribute_not_only_the_union():
    good = "Few-shot CoT achieves 2.8x the zero-shot accuracy (87.5% vs 31.25%)."
    extra = "Few-shot methods overall achieve 2.4x (75% vs 31%)."
    claim = "Few-shot CoT achieves 87.5% versus zero-shot 31.25%, reported as 2.8x."
    validate_numeric_support(claim, [good])
    with pytest.raises(ValueError, match="Irrelevant numerical citation"):
        validate_numeric_support(claim, [good, extra])
    assert atoms("2.8x and ٢٫٤ ضعف") == {"2.8", "2.4"}


def test_multiplier_suffixes_preserve_decimal_values_through_arabic_translation():
    from rag.generation.ollama import numeric_atoms

    assert numeric_atoms("2.8x and 14,575 questions") == numeric_atoms("2.8 ضعف و١٤٬٥٧٥ سؤالا")
    assert numeric_atoms("2.8x") != numeric_atoms("2.7x")
    with pytest.raises(ValueError, match="unsupported number"):
        semantic_items(
            {
                "statements": [
                    {
                        "text": "The accuracy is 2x higher.",
                        "citations": [{"ref_id": "S1:Q1"}],
                    }
                ]
            },
            [evidence("The accuracy is 2.8x higher.")],
            {},
            "What is the reported ratio?",
            "en",
        )


def test_roundtrip_unit_spelling_is_equivalent_but_values_and_units_cannot_disappear():
    from rag.generation.ollama import validate_roundtrip_content

    source = "Accuracy is 87.5%, a 2.8x increase."
    for reverse in (
        "Accuracy is 87.5 percent, a 2.8-fold increase.",
        "Accuracy is 87.5 per cent, a 2.8 times increase.",
    ):
        validate_roundtrip_content(source, reverse)
    for reverse in (
        "Accuracy is 87.5 percent, a 2.7-fold increase.",
        "Accuracy is 87.5, a 2.8-fold increase.",
        "Accuracy is 87.5 percent, a 2.8 increase.",
    ):
        with pytest.raises(ValueError, match="round-trip"):
            validate_roundtrip_content(source, reverse)


@pytest.mark.parametrize("answer", ["The models were evaluated.", "The study has 48 tasks."])
def test_omitted_requested_count_cannot_be_replaced_by_supported_background(answer):
    with pytest.raises(ValueError, match="missing requested model count"):
        validate_count_slot("How many models?", answer, ["48 tasks and 23 models were evaluated."])


def test_comparison_cannot_omit_source_reported_ratio():
    premise = "Few-\nshot CoT achieves 2.8x the zero-shot accuracy (87.5% vs 31.25%)."
    with pytest.raises(ValueError, match="missing source-reported ratio"):
        validate_comparison_slot(
            "Compare the reported prompting accuracies.",
            "Few-shot CoT achieves 87.5% versus zero-shot 31.25%.",
            [premise],
        )


@pytest.mark.parametrize(
    "question", ["What score did the best model achieve?", "ما نتيجة أفضل نموذج؟"]
)
def test_a_supported_but_nonnumeric_score_answer_is_still_incomplete(question):
    with pytest.raises(ValueError, match="missing requested score"):
        validate_score_slot(
            question, "The top model performed well.", ["The top model scored 61.4%."]
        )
    validate_score_slot(question, "The top model scored 61.4%.", ["The top model scored 61.4%."])


def test_deictic_comparison_cannot_substitute_another_supported_quantity():
    source = ["We evaluated 29 models on 180 tasks, comprising 45,360 individual test cases."]
    question = (
        "How many tasks and individual test cases were evaluated? Distinguish the two quantities."
    )
    with pytest.raises(ValueError, match="missing requested evaluation count"):
        validate_quantity_comparison(
            question,
            "Distinguish the two quantities.",
            "The two quantities are 180 tasks and 29 models.",
            source,
        )
    validate_quantity_comparison(
        question,
        "Distinguish the two quantities.",
        "The two quantities are 180 tasks and 45,360 individual test cases.",
        source,
    )


def test_arabic_coordinated_count_questions_keep_independent_required_slots():
    parts = question_parts("كم عدد المستشعرات، وكم عدد حالات الاختبار الفردية؟ ميّز بين الكميتين.")
    assert parts == [
        "كم عدد المستشعرات",
        "كم عدد حالات الاختبار الفردية؟",
        "ميّز بين الكميتين.",
    ]
    assert len(question_parts("كيف تعمل المستشعرات والكاميرات معًا؟")) == 1


def test_ordinary_identifier_protection_keeps_reverse_input_independent(service, monkeypatch):
    original_client = httpx.Client
    calls = []
    original = "The SensorSet questions use NLP."

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        result = (
            "تستخدم أسئلة __TERM_1__ منهج __TERM_2__."
            if len(calls) == 1
            else "The __TERM_1__ questions use __TERM_2__."
        )
        return httpx.Response(200, json={"message": {"content": json.dumps({"text": result})}})

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original_client(transport=httpx.MockTransport(respond))
    )
    arabic = service.generator.translate_ordinary_text(original, "ar")
    reverse = service.generator.translate_ordinary_text(arabic, "en")
    assert arabic == "تستخدم أسئلة SensorSet منهج NLP."
    assert reverse == original
    assert json.loads(calls[0]["messages"][1]["content"]) == (
        "The __TERM_1__ questions use __TERM_2__."
    )
    reverse_input = json.loads(calls[1]["messages"][1]["content"])
    assert reverse_input == "تستخدم أسئلة __TERM_1__ منهج __TERM_2__."
    assert original not in json.dumps(calls[1])


@pytest.mark.parametrize(
    "text,reason",
    [
        ("تحتوي البيانات على 23 سؤالا.", "must occur once"),
        ("يحتوي __TERM_1__ و__TERM_1__ على 23 سؤالا.", "must occur once"),
        ("يحتوي __TERM_9__ على 23 سؤالا.", "Unknown protected translation token"),
    ],
)
def test_ordinary_translation_rejects_dropped_duplicated_or_unknown_identifiers(
    service, monkeypatch, text, reason
):
    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"message": {"content": json.dumps({"text": text})}}
                )
            )
        ),
    )
    with pytest.raises(ValueError, match=reason):
        service.generator.translate_ordinary_text("SensorSet contains 23 questions.", "ar")


@pytest.mark.parametrize("change", [None, "number", "meaning"])
def test_protected_identifiers_do_not_bypass_number_or_meaning_checks(service, monkeypatch, change):
    hits = [evidence("SensorSet contains 23 questions.")]
    original_client = httpx.Client

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        if isinstance(user, str):
            if (
                "Translate the supplied verified answer sentence into Arabic"
                in body["messages"][0]["content"]
            ):
                result = {"text": f"يحتوي __TERM_1__ على {99 if change == 'number' else 23} سؤالا."}
            else:
                result = {
                    "text": f"__TERM_1__ contains 23 {'models' if change == 'meaning' else 'questions'}."
                }
        elif user.get("task") == "semantic-grounding-audit":
            result = {
                "verdicts": [
                    dict(
                        item_id=user["items"][0]["item_id"],
                        supported=True,
                        complete=True,
                        direct=True,
                        reason="Positive model verdict cannot bypass deterministic checks.",
                    )
                ]
            }
        else:
            result = {
                "statements": [
                    {"text": "SensorSet contains 23 questions.", "citations": [{"ref_id": "S1:Q1"}]}
                ]
            }
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original_client(transport=httpx.MockTransport(respond))
    )
    if change:
        with pytest.raises(
            GenerationUnavailable, match="source-grounded number|round-trip content"
        ):
            service.generator.generate("كم عدد الأسئلة؟", "ar", hits)
    else:
        result = service.generator.generate("كم عدد الأسئلة؟", "ar", hits)
        assert result.mode == "translated-answer" and "SensorSet" in result.answer
        assert result.citations[0].quote == hits[0].chunk.text


def test_ordinary_translation_refuses_reserved_tokens_in_input(service):
    with pytest.raises(ValueError, match="reserved scientific translation token"):
        service.generator.translate_ordinary_text("__TERM_1__ contains 23 questions.", "ar")


def test_explicit_abstract_selection_excludes_later_conflicting_aggregates(service):
    abstract = evidence("Abstract\nThe study evaluated 23 models.")
    later = evidence("The table compares 9 models and scores.", section="Results", page=3)
    service.index.chunks = [abstract.chunk, later.chunk]
    for question in ("According to the abstract, how many models?", "وفقا للملخص، كم عدد النماذج؟"):
        selected = service.select_evidence([later], question, {"source"})
        assert [h.chunk.chunk_id for h in selected] == [abstract.chunk.chunk_id]


def test_omission_is_retried_then_disclosed_even_with_positive_semantic_judge(
    service, monkeypatch, tmp_path
):
    hits = [evidence("We evaluated 23 models on 48 tasks.")]
    client = httpx.Client
    calls = []
    monkeypatch.setenv("RAG_GENERATION_TRACE_DIR", str(tmp_path))

    def respond(request):
        data = json.loads(request.content)
        user = json.loads(data["messages"][-1]["content"])
        calls.append(user)
        if user.get("task") == "semantic-grounding-audit":
            result = {
                "verdicts": [
                    dict(
                        item_id=user["items"][0]["item_id"],
                        supported=True,
                        complete=True,
                        direct=True,
                        reason="The task count is sourced.",
                    )
                ]
            }
        else:
            key = next(iter(user["question_parts"]))
            result = {key: {"text": "There are 48 tasks.", "citations": [{"ref_id": "S1:Q1"}]}}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(respond))
    )
    with pytest.raises(GenerationUnavailable, match="missing requested model count"):
        service.generator.generate("How many tasks and models were evaluated?", "en", hits)
    assert len([c for c in calls if "validation_error" in c]) == 2
    trace = [
        json.loads(line) for line in (tmp_path / "generation-wire.jsonl").read_text().splitlines()
    ]
    assert len([r for r in trace if r["stage"] == "generation-raw"]) == 4
    assert len([r for r in trace if r["stage"] == "validation-rejection"]) == 2


def test_untranslated_english_is_not_an_arabic_generated_answer():
    hits = [evidence("The study evaluated 23 models.")]
    with pytest.raises(ValueError, match="Output language mismatch"):
        semantic_items(
            {
                "statements": [
                    {
                        "text": "The study evaluated 23 models.",
                        "citations": [{"ref_id": "S1:Q1"}],
                    }
                ]
            },
            hits,
            {},
            "كم عدد النماذج؟",
            "ar",
        )


def test_arabic_question_pivot_explicitly_requests_english_before_guarded_translation(
    service, monkeypatch
):
    hits = [evidence("24 models were evaluated.")]
    client = httpx.Client
    seen = []

    def respond(request):
        wire = json.loads(request.content)
        user = json.loads(wire["messages"][-1]["content"])
        if user.get("task") == "semantic-grounding-audit":
            result = {
                "verdicts": [
                    dict(
                        item_id=user["items"][0]["item_id"],
                        supported=True,
                        complete=True,
                        direct=True,
                        reason="The exact count is supplied.",
                    )
                ]
            }
        else:
            seen.append(wire)
            assert user["language"] == "en"
            assert "required text language is ENGLISH" in wire["messages"][0]["content"]
            assert (
                "even when question or original_question is Arabic"
                in wire["messages"][0]["content"]
            )
            result = {
                "statements": [
                    {"text": "24 models were evaluated.", "citations": [{"ref_id": "S1:Q1"}]}
                ]
            }
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(respond))
    )
    monkeypatch.setattr(
        service.generator,
        "translate_ordinary_text",
        lambda text, language: (
            "تم تقييم 24 نموذجًا." if language == "ar" else "24 models were evaluated."
        ),
    )
    result = service.generator.generate("كم عدد النماذج؟", "ar", hits)
    assert len(seen) == 1 and result.mode == "translated-answer"
    assert "تم تقييم 24" in result.answer
    assert result.citations[0].quote == hits[0].chunk.text


def test_citation_repair_receives_failed_answer_and_relevant_original_refs(service, monkeypatch):
    hits = [
        evidence("The study is a benchmark for Arabic NLP."),
        evidence("The study uses Modern Standard Arabic.", page=2),
    ]
    client = httpx.Client
    generated = 0

    def respond(request):
        nonlocal generated
        user = json.loads(json.loads(request.content)["messages"][-1]["content"])
        if user.get("task") == "semantic-grounding-audit":
            supported = "Modern Standard Arabic" in " ".join(user["items"][0]["premises"])
            result = {
                "verdicts": [
                    dict(
                        item_id=user["items"][0]["item_id"],
                        supported=supported,
                        complete=supported,
                        direct=supported,
                        reason="Require the actual language passage.",
                    )
                ]
            }
        else:
            generated += 1
            if generated == 2:
                assert user["previous_answer"]["statements"][0]["citations"] == [
                    {"ref_id": "S1:Q1"}
                ]
                hints = user["repair_quote_candidates"][0]["candidate_sources"]
                assert hints[0]["label"] == "S2"
                assert {s["label"] for s in user["evidence"]} == {"S1", "S2"}
            result = {
                "statements": [
                    {
                        "text": "The study uses Modern Standard Arabic.",
                        "citations": [{"ref_id": "S1:Q1" if generated == 1 else "S2:Q1"}],
                    }
                ]
            }
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(respond))
    )
    result = service.generator.generate("In what language is the study written?", "en", hits)
    assert generated == 2
    assert result.citations[0].page == 2
    assert result.citations[0].quote == hits[1].chunk.text
