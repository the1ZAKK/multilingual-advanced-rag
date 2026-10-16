import json
import re

import httpx
import pytest

from rag.generation.ollama import (
    REVERSE_TRANSLATION_SYSTEM,
    GenerationUnavailable,
    authored_contributions,
    direct_support_candidates,
    expand_scientific_terms,
    protect_scientific_terms,
    reject_prompt_echo,
    reverse_scientific_terms,
    semantic_items,
    validate_answer,
    validate_semantic_verdicts,
    validate_translation_atoms,
)
from rag.models import Chunk, Hit


def evidence(text, *, section=None):
    return [Hit(chunk=Chunk(
        chunk_id="other-paper:p1:0", document_id="other-paper", document_name="Other paper.pdf",
        page=1, text=text, normalized=text, start=0, end=len(text), section=section,
    ), score=1)]


def mock_model(monkeypatch, generated, audit=None, *, source_units=None):
    requests = []
    originals = {name.casefold(): unit["text"] for name, unit in (
        authored_contributions(source_units).items() if source_units else []
    )}
    translation_index = 0
    spans = {
        "a low-cost sensor for freshwater.": "مستشعر منخفض التكلفة للمياه العذبة.",
        "sensor classifiers on": "مصنفات للمستشعرات على",
        "river datasets.": "مجموعات بيانات من الأنهار.",
        "calibration drift": "انجراف المعايرة",
        "under changing temperatures.": "تحت درجات حرارة متغيرة.",
    }

    def handle(request):
        nonlocal translation_index
        body = json.loads(request.content)
        data = json.loads(body["messages"][1]["content"])
        system = body["messages"][0]["content"]
        if isinstance(data, str):
            if "Translate ONLY the supplied source span" in system:
                data = {"task": "relation-translation", "source_span": data,
                        "unit_id": ("first", "second", "third")[translation_index % 3]}
            else:
                data = {"task": "source-translation", "author_statement": data,
                        "unit_id": "reverse-" + ("first", "second", "third")[translation_index % 3],
                        "protected_terms": []}
        requests.append((body, data))
        if data.get("task") == "semantic-grounding-audit":
            result = audit(data) if audit else {"verdicts": [
                {"item_id": item["item_id"], "supported": True, "complete": True,
                 "direct": True, "reason": "The premise directly states the same claim."}
                for item in data["items"]
            ]}
        elif data.get("task") == "source-translation":
            if data["unit_id"].startswith("reverse-"):
                result = {"text": originals[data["unit_id"].removeprefix("reverse-")]}
                translation_index += 1
            else:
                result = generated(data) if callable(generated) else {
                    "text": generated[data["unit_id"]]["text"]
                }
        elif data.get("task") == "relation-translation":
            if callable(generated):
                result = generated(data)
            else:
                unit = generated[data["unit_id"]]["text"]
                result = {"text": unit if unit.startswith(("تقدم الدراسة", "First", "Second", "Third"))
                          else spans[data["source_span"]]}
        else:
            result = generated(data) if callable(generated) else generated
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    real_client = httpx.Client
    monkeypatch.setattr(
        "rag.generation.ollama.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle)),
    )
    return requests


def statement(text, ref="S1:Q1"):
    return {"text": text, "citations": [{"ref_id": ref}]}


def test_valid_source_quote_does_not_establish_claim_entailment(service, monkeypatch, caplog):
    hits = evidence("HumanBench is a benchmark for evaluating language models.")
    payload = {"statements": [statement("HumanBench is a language model.")]}
    # The provenance check correctly succeeds; a distinct semantic check fails.
    assert validate_answer(payload, hits)[1][0].quote in hits[0].chunk.text

    def audit(data):
        return {"verdicts": [{"item_id": data["items"][0]["item_id"], "supported": False,
                              "complete": True, "direct": False,
                              "reason": "The premise identifies a benchmark, not a model."}]}

    calls = mock_model(monkeypatch, payload, audit)
    with pytest.raises(GenerationUnavailable, match="Semantic grounding failed"):
        service.generator.generate("What is HumanBench?", "en", hits)
    assert len(calls) == 4  # generation/audit, then retained-context generation/audit
    assert "ollama_retry_validation_failed" in caplog.text


def test_supported_generic_claim_is_not_forced_into_numbered_list(service, monkeypatch):
    hits = evidence("The experiment evaluates a sensor classifier on WaterBench.")
    calls = mock_model(monkeypatch, {"statements": [
        statement("The evaluated model is a sensor classifier, and WaterBench is the benchmark.")
    ]})
    result = service.generator.generate("What model and benchmark are evaluated?", "en", hits)
    assert not result.answer.startswith("1.")
    assert result.mode == "ollama"
    assert calls[1][1]["items"][0]["premises"] == [hits[0].chunk.text]


def test_positive_introduction_requires_direct_object_evidence(service, monkeypatch):
    hits = evidence("We introduce an Arabic sensor classifier for water quality.")
    assert direct_support_candidates("Arabic sensor classifier", hits) == ["S1:Q1"]
    calls = mock_model(monkeypatch, {"statements": [
        statement("The work introduces an Arabic sensor classifier.")
    ]})
    result = service.generator.generate("Does this work introduce an Arabic sensor classifier?",
                                        "en", hits)
    assert result.mode == "ollama"
    assert calls[1][1]["items"][0]["require_direct"]


@pytest.mark.parametrize("language", ["ar", "en"])
def test_unsupported_introduction_is_evidence_local_refusal(service, monkeypatch, language):
    hits = evidence("We introduce a benchmark suite for evaluating language models.")
    calls = mock_model(monkeypatch, {})
    result = service.generator.generate("Does this study introduce a new water quality sensor?",
                                        language, hits)
    assert result.mode == "insufficient-evidence" and not calls
    if language == "en":
        assert "retrieved evidence is insufficient to establish" in result.answer
        assert "not proof of absence throughout the paper" in result.answer
        assert "does not introduce" not in result.answer
    else:
        assert "الأدلة المسترجعة" in result.answer and "ليس دليلاً" in result.answer
    assert result.citations[0].quote == hits[0].chunk.text


def test_appendix_instruction_and_negation_are_not_direct_introduction_support():
    assert not direct_support_candidates("sensor classifier", evidence(
        "We introduce a sensor classifier.", section="A.1 Prompt examples"
    ))
    assert not direct_support_candidates("sensor classifier", evidence(
        "We do not introduce a sensor classifier."
    ))


def test_positive_claim_fails_if_semantic_audit_finds_only_indirect_support(service, monkeypatch):
    hits = evidence("Researchers introduce a sensor classifier in related work.")

    def audit(data):
        return {"verdicts": [{"item_id": data["items"][0]["item_id"], "supported": True,
                              "complete": True, "direct": False,
                              "reason": "The introduction is attributed to other researchers."}]}

    mock_model(monkeypatch, {"statements": [statement("This paper introduces a sensor classifier.")]},
               audit)
    with pytest.raises(GenerationUnavailable, match="direct=False"):
        service.generator.generate("Does the paper introduce a sensor classifier?", "en", hits)


def test_unrelated_number_cannot_be_filled_from_other_context(service, monkeypatch):
    hits = evidence("The sensor classifier achieves 82.4 percent accuracy in freshwater.")
    payload = {"statements": [statement("The sensor classifier achieves 95 percent accuracy.")]}
    calls = mock_model(monkeypatch, payload)
    with pytest.raises(GenerationUnavailable, match="unsupported number"):
        service.generator.generate("What numerical result was reported?", "en", hits)
    assert len(calls) == 2  # reject before either semantic audit request


@pytest.fixture
def authored_other_paper():
    return evidence(
        "Our contributions are threefold: First, we introduce a low-cost sensor for freshwater. "
        "Second, we evaluate eight sensor classifiers on two river datasets. "
        "Third, we analyze calibration drift and robustness under changing temperatures.",
        section="Contributions",
    )


def translated_slots(hits):
    translations = {
        "First": "أولاً، نقدم مستشعراً منخفض التكلفة للمياه العذبة.",
        "Second": "ثانياً، نقيّم ثمانية مصنفات للمستشعرات على مجموعتي بيانات من الأنهار.",
        "Third": "ثالثاً، نحلل انجراف المعايرة والمتانة عند تغير درجات الحرارة.",
    }
    return {
        ordinal.casefold(): {"text": translations[ordinal],
                             "citations": [{"ref_id": ref} for ref in item["refs"]]}
        for ordinal, item in authored_contributions(hits).items()
    }


def test_arabic_author_translation_keeps_all_source_units_and_fixed_refs(
    service, monkeypatch, authored_other_paper
):
    hits = authored_other_paper
    calls = mock_model(monkeypatch, translated_slots(hits), source_units=hits)
    result = service.generator.generate("ما المساهمات الثلاث التي يذكرها المؤلفون؟", "ar", hits)
    assert result.mode == "translated-contributions" and "نقيم ثمانية" in result.answer
    translations = [call for call in calls if call[1]["task"] != "semantic-grounding-audit"]
    reverses = [call for call in translations if call[1]["task"] == "source-translation"]
    assert len(reverses) == 3
    for call in reverses:
        if call[1]["unit_id"].startswith("reverse-"):
            assert "First, we" not in call[1]["author_statement"]
            assert not call[1]["protected_terms"]
    items = [call[1]["items"][0] for call in calls if call[1]["task"] == "semantic-grounding-audit"]
    assert all(item["require_complete"] for item in items)
    assert items[1]["full_author_statement"].startswith("Second, we evaluate eight")
    assert items[1]["independent_reverse_translation"].startswith("Second, we evaluate eight")
    assert "الصفحة 1" in result.answer


@pytest.mark.parametrize("slot", ["first", "second", "third"])
def test_incomplete_or_wrong_translated_contribution_rejected_semantically(
    service, monkeypatch, authored_other_paper, slot
):
    payload = translated_slots(authored_other_paper)
    payload[slot]["text"] = "تقدم الدراسة مجموعة بيانات جديدة للمياه."
    bad_id = f"item-{('first', 'second', 'third').index(slot) + 1}"

    def audit(data):
        return {"verdicts": [
            {"item_id": item["item_id"], "supported": item["item_id"] != bad_id,
             "complete": item["item_id"] != bad_id, "direct": True,
             "reason": "The translated item replaces or omits the author's stated action."}
            for item in data["items"]
        ]}

    mock_model(monkeypatch, payload, audit, source_units=authored_other_paper)
    with pytest.raises(GenerationUnavailable,
                       match="supported=False, complete=False|Unfaithful|Protected translation token"):
        service.generator.generate("اذكر إسهامات البحث الثلاث.", "ar", authored_other_paper)


def test_english_source_const_cannot_override_arabic_output_requirement(
    service, monkeypatch, authored_other_paper
):
    payload = {
        ordinal.casefold(): {"text": item["text"],
                             "citations": [{"ref_id": ref} for ref in item["refs"]]}
        for ordinal, item in authored_contributions(authored_other_paper).items()
    }
    calls = mock_model(monkeypatch, payload, source_units=authored_other_paper)
    with pytest.raises(GenerationUnavailable, match="Output language mismatch"):
        service.generator.generate("ما مساهمات الدراسة؟", "ar", authored_other_paper)
    assert len(calls) == 2


def test_translation_cannot_change_source_bound_citation(service, monkeypatch, authored_other_paper):
    def invent_reference(data):
        return {"text": "نقدم مستشعراً منخفض التكلفة للمياه العذبة.",
                "citations": [{"ref_id": "S1:Q99"}]}

    mock_model(monkeypatch, invent_reference, source_units=authored_other_paper)
    with pytest.raises(GenerationUnavailable, match="Invalid translated relation span"):
        service.generator.generate("ما المساهمات الثلاث؟", "ar", authored_other_paper)


def test_semantic_audit_requires_every_claim_and_cannot_accept_truthy_strings():
    items = [{"item_id": "item-1", "require_direct": False}]
    for verdict in [
        {"verdicts": []},
        {"verdicts": [{"item_id": "item-1", "supported": "true", "complete": True,
                       "direct": True, "reason": "present"}]},
        {"verdicts": [{"item_id": "item-2", "supported": True, "complete": True,
                       "direct": True, "reason": "present"}]},
    ]:
        with pytest.raises(ValueError):
            validate_semantic_verdicts(verdict, items)


def test_generic_audit_uses_only_individually_cited_premises():
    hits = evidence("WaterBench is a dataset. The sensor classifier achieves 82.4 percent accuracy.")
    payload = {"statements": [statement("WaterBench is a dataset.")]}
    items = semantic_items(payload, hits, {}, "Explain WaterBench.", "en")
    assert items[0]["premises"] == ["WaterBench is a dataset."]


def test_native_arabic_author_list_preserves_three_distinct_complete_units(service, monkeypatch):
    hits = evidence(
        "مساهماتنا ثلاث: أوّلاً، نقدم مستشعراً منخفض التكلفة للمياه العذبة. "
        "ثانيًا، نقيّم ثمانية مصنفات على مجموعتي بيانات من الأنهار. "
        "ثالثاً، نحلل انجراف المعايرة والمتانة عند تغير درجات الحرارة.",
        section="Contributions",
    )
    authored = authored_contributions(hits)
    assert list(authored) == ["First", "Second", "Third"]
    assert "المعايرة والمتانة" in authored["Third"]["text"]
    payload = {ordinal.casefold(): {"text": item["text"],
                                   "citations": [{"ref_id": ref} for ref in item["refs"]]}
               for ordinal, item in authored.items()}
    calls = mock_model(monkeypatch, payload)
    result = service.generator.generate("ما الإسهامات الثلاث لهذه الدراسة؟", "ar", hits)
    assert result.mode == "source-contributions" and len(calls) == 1
    assert all(str(number) + ". " in result.answer for number in (1, 2, 3))


@pytest.mark.parametrize(
    ("source", "bad_translation", "error"),
    [
        ("We present comprehensive benchmarks spanning code generation.",
         "نقدم معايير مكثفة لتصنيف الكود.", "comprehensive|code generation"),
        ("We evaluate over 40 multilingual models.",
         "نقيم حوالي 40 نموذجا متعدد اللغات.", "numeric relation"),
        ("We evaluate over 12 multilingual models.",
         "نقيم 12 نموذجا متعدد اللغات.", "numeric relation"),
        ("We analyze cross-task correlations and robustness testing.",
         "نحلل التوصيات بين المهام والاختبارات المتينة.", "correlations|robustness"),
        ("We curate native content using synthetic generation and human verification.",
         "نرتب محتوى مصنعاً باستخدام إنتاج مصنف ومراجعة آلية.", "native|synthetic|verification"),
        ("We analyze the relationship between different cognitive capabilities.",
         "نحلل علاقة مختلفة من المهارات العقلية.", "capabilities"),
        ("We conduct the most comprehensive assessment.",
         "نجري التقييم الأكثر معرفة.", "comprehensive"),
    ],
)
def test_observed_and_other_paper_translation_meaning_changes_fail_source_activated_guards(
    source, bad_translation, error
):
    with pytest.raises(ValueError, match=error):
        validate_translation_atoms(source, bad_translation)


@pytest.mark.parametrize(
    ("source", "translation"),
    [
        ("We present three comprehensive benchmarks spanning code generation.",
         "نقدم ثلاثة معايير تقييم شاملة لتوليد الشفرة البرمجية."),
        ("We evaluate over 12 multilingual models.",
         "نقيّم أكثر من ١٢ نموذجاً متعدد اللغات."),
        ("We evaluate over 40 multilingual models.",
         "نقيم ما يزيد على أربعين نموذجا متعدد اللغات."),
        ("We analyze cross-task correlations and robustness testing.",
         "نحلل الترابط عبر المهام واختبارات المتانة."),
        ("We use native content curation, synthetic generation, and translation with human verification.",
         "نستخدم تنظيم المحتوى الأصيل والتوليد الاصطناعي والترجمة مع مراجعة بشرية."),
        ("We analyze the relationship between different cognitive capabilities.",
         "نحلل العلاقة بين القدرات المعرفية المختلفة."),
    ],
)
def test_faithful_scientific_translation_paraphrases_preserve_source_concepts(source, translation):
    validate_translation_atoms(source, translation)


def test_generic_terminology_protection_preserves_plural_and_numeric_bounds():
    source = "We evaluate over 12 models using rigorous methodologies and comprehensive benchmarks."
    template, bindings = protect_scientific_terms(source)
    assert all(token in template and source_term in source for token, source_term, _ in bindings)
    assert "أكثر من 12" in [arabic for _, _, arabic in bindings]
    assert "منهجيات صارمة" in [arabic for _, _, arabic in bindings]
    assert "معايير تقييم شاملة" in [arabic for _, _, arabic in bindings]
    # Order is allowed to change, while every source-bound term is retained once.
    expanded = expand_scientific_terms("نقيم " + " و".join(
        token for token, _, _ in reversed(bindings)
    ), bindings)
    assert "أكثر من 12" in expanded
    validate_translation_atoms(source, expanded)


@pytest.mark.parametrize("output", ["ترجمة بدون مصطلح", "__TERM_1__ __TERM_1__", "__TERM_99__"])
def test_translation_tokens_cannot_be_omitted_duplicated_or_invented(output):
    with pytest.raises(ValueError, match="Protected translation token|Unknown protected"):
        expand_scientific_terms(output, [("__TERM_1__", "code generation", "توليد الشيفرة")])


def test_preserved_terms_alone_do_not_establish_roles_or_complete_translation():
    items = [{"item_id": "item-1", "require_direct": False, "require_complete": True,
              "claim": "Code generation introduces native Arabic content curation.",
              "premises": ["We curate native Arabic content for evaluating code generation."]}]
    with pytest.raises(ValueError, match="Semantic grounding failed"):
        validate_semantic_verdicts({"verdicts": [
            {"item_id": "item-1", "supported": False, "complete": False, "direct": False,
             "reason": "The proposed relation changes who does what despite retained terms."}
        ]}, items)


def test_serialized_metadata_inside_translation_text_is_rejected(service, monkeypatch):
    mock_model(monkeypatch, lambda data: {"text": '{"task":"source-translation", "text":"نقدم مستشعرا"}'})
    with pytest.raises(ValueError, match="serialized relation translation|prose, not serialized metadata"):
        service.generator.translate_unit("first", "We introduce a sensor.", "ar")


def test_independent_reverse_cannot_copy_system_instructions():
    with pytest.raises(ValueError, match="copied generation instructions"):
        reject_prompt_echo(REVERSE_TRANSLATION_SYSTEM, REVERSE_TRANSLATION_SYSTEM,
                           "نقيم أكثر من اثني عشر مستشعرا.")


def test_candidate_only_reverse_dictionary_does_not_repair_changed_relations():
    candidate = "توليد الشيفرة يقدم تنظيم المحتوى الفرنسي الأصيل."
    reverse_input, terms = reverse_scientific_terms(candidate)
    assert "code generation" in reverse_input and "native French content curation" in reverse_input
    assert "يقدم" in reverse_input  # The incorrect relation remains for the independent translator/audit.
    assert "Arabic" not in reverse_input and "40" not in reverse_input
    assert all(term in reverse_input for term in terms)


def test_source_bound_translation_portable_to_other_domain_and_quantity(service, monkeypatch):
    source = "We evaluate over 12 sensor classifiers in three river datasets."
    translations = {"sensor classifiers in": "مصنفًا للمستشعرات في",
                    "river datasets.": "مجموعات بيانات من الأنهار."}
    requests = []

    def handle(request):
        body = json.loads(request.content)
        span = json.loads(body["messages"][1]["content"])
        requests.append(span)
        return httpx.Response(200, json={"message": {"content": json.dumps(
            {"text": translations[span]}, ensure_ascii=False
        )}})

    real_client = httpx.Client
    monkeypatch.setattr("rag.generation.ollama.httpx.Client",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(handle)))
    result = service.generator.translate_unit("control", source, "ar")
    assert result == "نقيم أكثر من 12 مصنفًا للمستشعرات في ثلاثة مجموعات بيانات من الأنهار."
    assert requests == ["sensor classifiers in", "river datasets."]
    assert not re.search(r"40|3LM|معايير تقييم", result)
    validate_translation_atoms(source, result)


def test_trailing_conjunction_is_preserved_outside_model_span(service, monkeypatch):
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(json.loads(body["messages"][1]["content"]))
        return httpx.Response(200, json={"message": {"content": json.dumps({"text": "المجالات"})}})

    real_client = httpx.Client
    monkeypatch.setattr("rag.generation.ollama.httpx.Client",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(handle)))
    assert service.generator.translate_span("control", "domains and") == "المجالات و"
    assert calls == ["domains"]
