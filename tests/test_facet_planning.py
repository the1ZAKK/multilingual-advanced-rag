import json
import re
from copy import deepcopy

import httpx
import pytest

from rag.generation.facets import (
    comparison_text_pattern,
    count_text_pattern,
    plan_facet,
    requested_counts,
    required_count_refs,
)
from rag.generation.numerical import validate_comparison_slot, validate_numeric_support
from rag.generation.ollama import GenerationUnavailable
from rag.models import Chunk, Hit


def source(*quotes):
    return [
        dict(
            label="S1",
            page=1,
            document="Independent.pdf",
            quotes=[dict(ref_id=f"S1:Q{i}", quote=q) for i, q in enumerate(quotes, 1)],
        )
    ]


@pytest.mark.parametrize("quantity", ["23", "31"])
@pytest.mark.parametrize(
    "question", ["How many models were evaluated?", "كم عدد النماذج التي تم تقييمها؟"]
)
def test_count_contract_uses_source_values_and_the_requested_predicate(quantity, question):
    sources = source(
        f"We evaluated {quantity} models from 9 providers.",
        "Only 4 models consistently generate structured output.",
    )
    original = deepcopy(sources)
    contract, selected = plan_facet(sources, question)
    assert contract["requested_entities"] == ["model"]
    assert {item["quantity"] for item in contract["source_candidates"]} == {quantity}
    assert [q["ref_id"] for s in selected for q in s["quotes"]] == ["S1:Q1"]
    assert sources == original
    assert re.match(count_text_pattern(contract), f"{quantity} evaluated models.")
    assert not re.match(count_text_pattern(contract), "4 models.")


def test_generation_count_does_not_select_evaluation_total():
    contract, selected = plan_facet(
        source("We evaluated 23 models.", "Only 4 models consistently generate structured output."),
        "How many models consistently generate structured output across the test cases?",
    )
    assert {item["quantity"] for item in contract["source_candidates"]} == {"4"}
    assert selected[0]["quotes"][0]["ref_id"] == "S1:Q2"


def test_facet_roles_do_not_come_from_the_document_name_or_an_empty_pattern():
    assert requested_counts("According to ModelStudy, how many questions?") == {"question"}
    contract, _ = plan_facet(source("23.5 tasks are used."), "How many tasks?")
    assert re.match(count_text_pattern(contract), "23.5 tasks")
    assert not re.match(count_text_pattern(contract), " tasks")
    assert not re.match(count_text_pattern(contract), "23.55 tasks")


def test_count_schema_pattern_has_both_required_grammar_anchors():
    contract, _ = plan_facet(source("We evaluated 23 models."), "How many models?")
    pattern = count_text_pattern(contract)
    assert pattern.startswith("^") and pattern.endswith("$")
    assert r"\s" not in pattern  # Unsupported by the real provider's grammar converter.
    assert re.fullmatch(pattern, "23 evaluated models.")
    assert not re.fullmatch(pattern, "We evaluated 23 models.")
    assert not re.fullmatch(pattern, "23 ")


@pytest.mark.parametrize("question", ["How many models?", "كم عدد النماذج؟"])
def test_count_schema_keeps_its_intermediate_entity_in_english(question):
    contract, _ = plan_facet(source("23 models were evaluated."), question)
    pattern = count_text_pattern(contract)
    assert re.fullmatch(pattern, "23 evaluated models")
    assert re.fullmatch(pattern, "23 LLMs.")
    assert not re.fullmatch(pattern, "23 نماذج")


def test_count_schema_pairs_each_source_quantity_with_its_own_entity():
    contract, _ = plan_facet(
        source("We evaluated 23 models from 9 providers."), "How many models and providers?"
    )
    pattern = count_text_pattern(contract)
    assert re.fullmatch(pattern, "23 models") and re.fullmatch(pattern, "9 providers")
    assert not re.fullmatch(pattern, "23 providers")
    assert not re.fullmatch(pattern, "9 models")


def test_quantity_distinction_schema_keeps_both_source_counts_and_their_entities():
    contract, _ = plan_facet(
        source("We used 180 distinct tasks for 45,360 individual test cases."),
        "Distinguish the two quantities.",
        "How many distinct tasks and individual test cases are evaluated?",
    )
    pattern = count_text_pattern(contract)
    assert re.fullmatch(pattern, "180 distinct tasks and 45,360 individual test cases")
    assert re.fullmatch(pattern, "45360 individual evaluations; 180 tasks.")
    for bad in (
        "180 distinct tasks",
        "45360 individual test cases",
        "45360 tasks and 180 individual test cases",
        "180 tasks and 77 individual test cases",
        "180 مهام و45360 تقييمات",
    ):
        assert not re.fullmatch(pattern, bad)


def test_two_quantity_grammar_cannot_grow_an_unbounded_adjective_or_pre_count_clause():
    contract, _ = plan_facet(
        source("We used 180 distinct tasks for 45,360 individual test cases."),
        "Distinguish the two quantities.",
        "How many distinct tasks and individual test cases are evaluated?",
    )
    pattern = count_text_pattern(contract)
    assert re.fullmatch(pattern, "180 distinct tasks; 45,360 individual test cases.")
    assert not re.fullmatch(pattern, "180 " + "distinct" * 200 + " tasks; 45360 test cases")
    assert not re.fullmatch(pattern, "180 tasks are evaluated; 45360 individual test cases")
    assert not re.fullmatch(pattern, "180 tasks; 45360 individual test cases were evaluated")
    assert not re.fullmatch(pattern, "180 tasks; 45360 individual test cases were assessed")
    assert "[A-Za-z-]*" not in pattern


def test_quantity_distinction_uses_qualified_noun_phrases_without_an_added_evaluation_clause():
    contract, _ = plan_facet(
        source("We evaluated 67 tasks on 2,040 individual test cases."),
        "Distinguish the two quantities.",
        "How many tasks and individual test cases were evaluated?",
    )
    assert "noun phrases" in contract["answer_style"]
    assert "Retain source qualifications" in contract["answer_style"]
    assert re.fullmatch(count_text_pattern(contract), "67 tasks and 2,040 individual test cases.")


def test_comparison_value_grammar_uses_only_the_native_condition_mapping_and_reported_ratio():
    contract, _ = plan_facet(
        source(
            "Few-shot CoT achieves 2.6x higher accuracy than zero-shot approaches (91% vs 35%)."
        ),
        "Compare few-shot CoT accuracy with zero-shot accuracy.",
    )
    pattern = comparison_text_pattern(contract)
    good = "few-shot chain-of-thought accuracy: 91% versus zero-shot accuracy: 35% (2.6x)"
    assert re.fullmatch(pattern, good)
    assert r"\s" not in pattern and r"\b" not in pattern
    for wrong in [
        good.replace("91%", "35%", 1),
        good.replace("2.6x", "2.2x"),
        good.replace("few-shot chain-of-thought", "few-shot"),
        good.replace(" (2.6x)", ""),
        good + " up to 2.6x",
    ]:
        assert not re.fullmatch(pattern, wrong)
    contract["source_comparisons"].append(
        {**contract["source_comparisons"][0], "multiplier": "2.7"}
    )
    assert comparison_text_pattern(contract) is None


def test_a_source_without_an_unambiguous_reported_comparison_gets_no_value_grammar():
    contract, _ = plan_facet(
        source("Few-shot CoT versus zero-shot accuracy is 91% versus 35%."),
        "Compare few-shot CoT accuracy with zero-shot accuracy.",
    )
    assert comparison_text_pattern(contract) is None


def test_an_unrelated_cost_multiplier_cannot_become_the_accuracy_multiplier():
    contract, _ = plan_facet(
        source(
            "Few-shot CoT has higher accuracy than zero-shot approaches (91% vs 35%), but costs 2.6x more."
        ),
        "Compare few-shot CoT accuracy with zero-shot accuracy.",
    )
    assert comparison_text_pattern(contract) is None


def test_the_reported_accuracy_multiplier_remains_distinct_from_a_cost_multiplier():
    contract, _ = plan_facet(
        source(
            "Few-shot CoT has 2.2x higher accuracy than zero-shot approaches (33% vs 15%), but costs 2.6x more."
        ),
        "Compare few-shot CoT accuracy with zero-shot accuracy.",
    )
    assert contract["source_comparisons"][0]["multiplier"] == "2.2"
    assert "2\\.2" in comparison_text_pattern(contract)


def test_distinct_quantities_bind_their_own_unambiguous_sources_after_overlap_deduplication():
    sources = source("We used 67 distinct tasks.", "We evaluated 2,040 individual test cases.")
    sources += [
        {
            **sources[0],
            "label": "S2",
            "quotes": [
                {**quote, "ref_id": quote["ref_id"].replace("S1", "S2")}
                for quote in sources[0]["quotes"]
            ],
        }
    ]
    contract, scoped = plan_facet(
        sources,
        "Distinguish the two quantities.",
        "How many distinct tasks and individual test cases were evaluated?",
    )
    assert required_count_refs(contract) == ["S1:Q1", "S1:Q2"]
    assert {q["ref_id"] for s in scoped for q in s["quotes"]} == {"S1:Q1", "S1:Q2"}
    contract["source_candidates"].append(dict(entity="task", quantity="68", ref_id="S1:Q3"))
    assert required_count_refs(contract) is None


@pytest.mark.parametrize(
    "question", ["What language is the dataset written in?", "ما لغة الأسئلة؟"]
)
def test_attribute_plans_request_qualified_values_without_an_existence_claim(question):
    contract, _ = plan_facet(source("The data comprises questions in French."), question)
    assert contract["kind"] == "attribute"
    assert "noun phrase" in contract["answer_style"]
    assert "source qualifications" in contract["answer_style"]
    assert "existence" in contract["answer_style"]


def test_the_requested_condition_pair_excludes_a_different_real_comparison():
    good = "Few-shot CoT achieves 2.6x the zero-shot accuracy (91% vs 35%)."
    other = "Few-shot methods achieve 2.2x the zero-shot accuracy (80% vs 36%)."
    contract, selected = plan_facet(
        source(other, good), "Compare few-shot CoT accuracy with zero-shot accuracy."
    )
    assert contract["kind"] == "condition-comparison"
    assert selected[0]["quotes"] == [dict(ref_id="S1:Q2", quote=good)]
    with pytest.raises(ValueError, match="requested condition comparison"):
        validate_comparison_slot(
            "Compare few-shot CoT accuracy with zero-shot accuracy.", other, [other]
        )


def test_attribute_plans_select_the_data_attribute_without_unrelated_model_background():
    sources = source(
        "Models trained on French texts are evaluated in Arabic despite limited datasets.",
        "The data comprises 900 multiple-choice questions in French.",
    )
    contract, selected = plan_facet(
        sources, "In what language and question format is the dataset written?"
    )
    assert contract["kind"] == "attribute"
    assert [q["ref_id"] for s in selected for q in s["quotes"]] == ["S1:Q2"]
    assert plan_facet(sources, "Was this model trained on the French language?")[0] is None


def test_a_known_value_cannot_be_relabelled_as_an_uncited_entity():
    with pytest.raises(ValueError, match="swapped counted entity"):
        validate_numeric_support("23 questions were evaluated.", ["We evaluated 23 models."])


def test_omitted_counts_and_wrong_scope_fail_before_a_positive_semantic_judge(service, monkeypatch):
    text = "We evaluated 23 models. Only 4 models consistently generate structured output."
    hits = [
        Hit(
            chunk=Chunk(
                chunk_id="source:p1:0",
                document_id="source",
                document_name="Source.pdf",
                page=1,
                text=text,
                normalized=text,
                start=0,
                end=len(text),
            ),
            score=1,
        )
    ]
    original_client = httpx.Client
    seen = []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        seen.append(user)
        assert user.get("task") != "semantic-grounding-audit", "An invalid count reached the judge"
        assert user["answer_contract"]["source_candidates"][0]["quantity"] == "23"
        assert body["format"]["properties"]["statements"]["items"]["properties"]["text"]["pattern"]
        result = {"statements": [{"text": "4 models.", "citations": [{"ref_id": "S1:Q2"}]}]}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(respond))
    )
    with pytest.raises(GenerationUnavailable, match="missing requested model count"):
        service.generator.generate("How many models were evaluated?", "en", hits)
    assert len(seen) == 2


def test_facet_candidates_are_not_automatic_acceptance(service, monkeypatch):
    text = "We evaluated 23 models."
    hits = [
        Hit(
            chunk=Chunk(
                chunk_id="source:p1:0",
                document_id="source",
                document_name="Source.pdf",
                page=1,
                text=text,
                normalized=text,
                start=0,
                end=len(text),
            ),
            score=1,
        )
    ]
    original_client = httpx.Client
    audits = []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        if user.get("task") == "semantic-grounding-audit":
            audits.append(user)
            result = {
                "verdicts": [
                    dict(
                        item_id=user["items"][0]["item_id"],
                        supported=False,
                        complete=False,
                        direct=False,
                        reason="A negative audit must remain a rejection.",
                    )
                ]
            }
        else:
            result = {"statements": [{"text": "23 models.", "citations": [{"ref_id": "S1:Q1"}]}]}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(respond))
    )
    with pytest.raises(GenerationUnavailable, match="Semantic grounding failed"):
        service.generator.generate("How many models were evaluated?", "en", hits)
    assert len(audits) == 2
