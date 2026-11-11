import json

import httpx
import pytest

from rag.generation.ollama import GenerationUnavailable
from rag.models import Chunk, Hit


def hits(text):
    return [
        Hit(
            chunk=Chunk(
                chunk_id="source:p1:0",
                document_id="source",
                document_name="Independent.pdf",
                page=1,
                text=text,
                normalized=text,
                start=0,
                end=len(text),
            ),
            score=1,
        )
    ]


def install(monkeypatch, respond):
    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(respond))
    )


def positive(user):
    return {
        "verdicts": [
            dict(
                item_id=user["items"][0]["item_id"],
                supported=True,
                complete=True,
                direct=True,
                reason="Positive judge fixture.",
            )
        ]
    }


@pytest.mark.parametrize(
    "quantity,entity,arabic", [(17, "providers", "مزودين"), (1, "provider", "مزود")]
)
def test_provider_terminology_is_source_activated_and_reverse_input_remains_independent(
    service, monkeypatch, quantity, entity, arabic
):
    claim = f"{quantity} {entity}"
    reverse_inputs = []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        if isinstance(user, str):
            if "into Arabic" in body["messages"][0]["content"]:
                assert user == f"{quantity} __TERM_1__"
                result = {"text": user}
            else:
                reverse_inputs.append(user)
                assert len(body["messages"]) == 2
                assert user == claim
                assert "previous_translation" not in json.dumps(body)
                result = {"text": user}
        elif user.get("task") == "semantic-grounding-audit":
            result = positive(user)
        else:
            result = {"statements": [{"text": claim, "citations": [{"ref_id": "S1:Q1"}]}]}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    answer = service.generator.generate(
        "كم عدد مقدمي الخدمات؟", "ar", hits(f"We evaluated {quantity} {entity}.")
    )
    assert answer.mode == "translated-answer"
    assert f"{quantity} {arabic}" in answer.answer
    assert reverse_inputs == [claim]


@pytest.mark.parametrize("output", ["18 __TERM_1__", "17 نماذج", "17 __TERM_1__ __TERM_1__"])
def test_provider_binding_cannot_hide_changed_numbers_entities_or_repeated_terms(
    service, monkeypatch, output
):
    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        if isinstance(user, str):
            if "into Arabic" in body["messages"][0]["content"]:
                result = {"text": output}
            else:
                pytest.fail("Invalid translation reached independent reverse translation")
        elif user.get("task") == "semantic-grounding-audit":
            result = positive(user)
        else:
            result = {"statements": [{"text": "17 providers", "citations": [{"ref_id": "S1:Q1"}]}]}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    with pytest.raises(GenerationUnavailable, match="number|translation token"):
        service.generator.generate(
            "كم عدد مقدمي الخدمات؟", "ar", hits("We evaluated 17 providers.")
        )


def test_attribute_prompt_fix_does_not_accept_an_added_existence_assertion():
    from rag.generation.ollama import validate_roundtrip_content

    with pytest.raises(ValueError, match="round-trip content"):
        validate_roundtrip_content(
            "The questions are in Modern Standard Arabic (MSA).",
            "Questions exist in Modern Standard Arabic (MSA).",
        )


@pytest.mark.parametrize(
    "source,arabic",
    [
        ("few-shot chain-of-thought prompting", "التلقين باستخدام سلسلة التفكير بأمثلة قليلة"),
        ("zero-shot approaches", "أساليب دون أمثلة"),
        ("few-shot accuracy", "دقة بأمثلة قليلة"),
        ("accuracies", "قيم الدقة"),
    ],
)
def test_scientific_compounds_preserve_the_condition_and_noun_as_one_source_span(source, arabic):
    from rag.generation.ollama import expand_scientific_terms, protect_answer_identifiers

    template, bindings = protect_answer_identifiers(source, translate_terms=True)
    assert template == "__TERM_1__"
    assert bindings == [("__TERM_1__", source, arabic)]
    assert expand_scientific_terms(template, bindings) == arabic


def test_compound_guidance_uses_only_selected_spans_and_reverse_has_no_forward_context(
    service, monkeypatch
):
    seen = []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        seen.append(body)
        if "into Arabic" in body["messages"][0]["content"]:
            guidance = json.loads(body["messages"][1]["content"])
            assert guidance["required_technical_terms"] == [
                {"source": "Few-shot prompting", "arabic": "التلقين بأمثلة قليلة"},
                {"source": "accuracy", "arabic": "دقة"},
                {"source": "zero-shot approaches", "arabic": "أساليب دون أمثلة"},
                {"source": "accuracies", "arabic": "قيم الدقة"},
            ]
            assert (
                user
                == "__TERM_1__ improves __TERM_2__ over __TERM_3__, with __TERM_4__ of 93% vs 31%."
            )
            result = {
                "text": "يحسن __TERM_1__ __TERM_2__ مقارنة بـ__TERM_3__، مع __TERM_4__ 93% مقابل 31%."
            }
        else:
            assert len(body["messages"]) == 2
            assert "few-shot prompting" in user and "zero-shot approaches" in user
            assert "accuracies" in user and "models" not in user
            assert "previous_translation" not in json.dumps(body)
            assert "validation_error" not in json.dumps(body)
            result = {
                "text": "Few-shot prompting improves accuracy over zero-shot approaches, with accuracies of 93% vs 31%."
            }
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    arabic = service.generator.translate_ordinary_text(
        "Few-shot prompting improves accuracy over zero-shot approaches, with accuracies of 93% vs 31%.",
        "ar",
    )
    assert "أساليب دون أمثلة" in arabic and "قيم الدقة" in arabic
    service.generator.translate_ordinary_text(arabic, "en")


@pytest.mark.parametrize(
    "source,wrong",
    [
        ("zero-shot approaches", "نماذج دون أمثلة"),
        ("accuracies", "دقائق"),
        ("few-shot accuracy", "دقة دون أمثلة"),
    ],
)
def test_wrong_noun_or_condition_cannot_replace_a_protected_compound(
    service, monkeypatch, source, wrong
):
    install(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"message": {"content": json.dumps({"text": wrong})}}
        ),
    )
    with pytest.raises(ValueError, match="Protected translation token"):
        service.generator.translate_ordinary_text(source, "ar")


def test_reverse_compounds_cannot_invent_an_approach_when_the_arabic_says_models(
    service, monkeypatch
):
    from rag.generation.ollama import validate_roundtrip_content

    def respond(request):
        body = json.loads(request.content)
        assert len(body["messages"]) == 2
        assert json.loads(body["messages"][-1]["content"]) == "نماذج zero-shot."
        return httpx.Response(
            200, json={"message": {"content": json.dumps({"text": "Zero-shot models."})}}
        )

    install(monkeypatch, respond)
    reverse = service.generator.translate_ordinary_text("نماذج دون أمثلة.", "en")
    with pytest.raises(ValueError, match="round-trip content"):
        validate_roundtrip_content("Zero-shot approaches.", reverse)


def test_translation_repairs_only_the_failed_stage_and_keeps_reverse_input_blind(
    service, monkeypatch, tmp_path
):
    source = "23 questions were evaluated."
    generations, forwards, reverses, audits = [], [], [], []
    monkeypatch.setenv("RAG_GENERATION_TRACE_DIR", str(tmp_path))

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        if isinstance(user, str):
            if "into Arabic" in body["messages"][0]["content"]:
                forwards.append(body)
                if len(forwards) == 2:
                    repair = json.loads(body["messages"][1]["content"])["repair"]
                    assert "source-grounded number" in repair["validation_error"]
                    assert "99" in repair["previous_translation"]
                result = {"text": f"تم تقييم {99 if len(forwards) == 1 else 23} سؤالا."}
            else:
                reverses.append(body)
                assert len(body["messages"]) == 2
                assert source not in json.dumps(body)
                assert "previous_translation" not in json.dumps(body)
                assert "validation_error" not in json.dumps(body)
                result = {"text": source}
        elif user.get("task") == "semantic-grounding-audit":
            audits.append(user)
            result = positive(user)
        else:
            generations.append(user)
            result = {"statements": [{"text": source, "citations": [{"ref_id": "S1:Q1"}]}]}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    answer = service.generator.generate("كم عدد الأسئلة؟", "ar", hits(source))
    assert answer.mode == "translated-answer" and "23" in answer.answer
    assert len(generations) == 1 and len(forwards) == 2 and len(reverses) == 1
    assert len(audits) == 2
    assert audits[-1]["items"][0]["require_complete"] is True
    assert audits[-1]["items"][0]["full_author_statement"] == source
    assert audits[0]["items"][0]["requested_facet"] == "كم عدد الأسئلة؟"
    assert audits[-1]["items"][0]["requested_facet"] == "كم عدد الأسئلة؟"
    assert all(a["items"][0]["question_context"] == "كم عدد الأسئلة؟" for a in audits)
    events = [
        json.loads(line) for line in (tmp_path / "generation-wire.jsonl").read_text().splitlines()
    ]
    assert any(event["request"].get("phase") == "ordinary-translation" for event in events)


@pytest.mark.parametrize(
    "arabic", ["الأسئلة باللغة العربية المعاصرة.", "الأسئلة باللغة الإنجليزية."]
)
def test_reverse_translator_cannot_launder_a_missing_standard_language_qualification(
    service, monkeypatch, arabic
):
    source = "The questions are in Modern Standard Arabic."
    reverses = []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        if isinstance(user, str):
            if "into Arabic" in body["messages"][0]["content"]:
                result = {"text": arabic}
            else:
                reverses.append(body)
                result = {"text": source}
        elif user.get("task") == "semantic-grounding-audit":
            result = positive(user)
        else:
            result = {"statements": [{"text": source, "citations": [{"ref_id": "S1:Q1"}]}]}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    with pytest.raises(GenerationUnavailable, match="missing meaning|Protected translation token"):
        service.generator.generate("ما لغة الأسئلة؟", "ar", hits(source))
    assert reverses == []


def test_native_technical_phrase_can_be_bound_without_leaking_forward_source(service, monkeypatch):
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        result = {"text": "Questions in Modern Standard Arabic."}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    translated = service.generator.translate_ordinary_text(
        "أسئلة باللغة العربية الفصحى المعاصرة.", "en"
    )
    assert "Modern Standard Arabic" in translated
    assert json.loads(seen[0]["messages"][-1]["content"]) == "أسئلة باللغة Modern Standard Arabic."
    assert len(seen[0]["messages"]) == 2


def test_model_http_error_is_retained_without_inventing_an_answer(service, monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_GENERATION_TRACE_DIR", str(tmp_path))
    install(monkeypatch, lambda request: httpx.Response(500, json={"error": "schema rejected"}))
    with pytest.raises(GenerationUnavailable, match="HTTPStatusError"):
        service.generator.generate("How many questions?", "en", hits("23 questions."))
    records = [
        json.loads(line) for line in (tmp_path / "generation-wire.jsonl").read_text().splitlines()
    ]
    assert records[0]["stage"] == "generation-http-error"
    assert records[0]["result"]["status_code"] == 500
    assert "schema rejected" in records[0]["result"]["body"]


@pytest.mark.parametrize("english,arabic", [("Arabic", "العربية"), ("French", "الفرنسية")])
def test_language_centric_qualification_uses_only_the_actual_arabic(
    service, monkeypatch, english, arabic
):
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps({"text": f"The {english}-centric model."})}},
        )

    install(monkeypatch, respond)
    service.generator.translate_ordinary_text(f"النموذج المتمحور حول {arabic}.", "en")
    assert json.loads(seen[0]["messages"][-1]["content"]) == f"النموذج {english}-centric."
    assert len(seen[0]["messages"]) == 2


@pytest.mark.parametrize("arabic", ["النموذج العربي.", "النموذج المركزي باللغة العربية."])
def test_language_mentions_or_central_models_cannot_replace_language_centric(
    service, monkeypatch, arabic
):
    from rag.generation.ollama import validate_translation_atoms

    with pytest.raises(ValueError, match="missing meaning"):
        validate_translation_atoms("The Arabic-centric model.", arabic)


def test_language_centric_qualification_cannot_switch_languages():
    from rag.generation.ollama import validate_translation_atoms

    validate_translation_atoms("The French-centric model.", "النموذج المتمحور حول الفرنسية.")
    with pytest.raises(ValueError, match="missing meaning"):
        validate_translation_atoms("The French-centric model.", "النموذج المتمحور حول العربية.")


def test_ordinary_translation_binds_full_scientific_terms_and_retains_identifiers(
    service, monkeypatch
):
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        assert json.loads(body["messages"][-1]["content"]) == (
            "__TERM_1__ questions in __TERM_2__ (__TERM_3__)."
        )
        result = {"text": "أسئلة __TERM_1__ باللغة __TERM_2__ (__TERM_3__)."}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    translated = service.generator.translate_ordinary_text(
        "Multiple-choice questions in Modern Standard Arabic (MSA).", "ar"
    )
    assert translated == "أسئلة اختيار من متعدد باللغة العربية الفصحى المعاصرة (MSA)."
    assert "Modern Standard Arabic" in json.dumps(seen[0])  # Forward-only terminology guidance.


@pytest.mark.parametrize("text", ["أسئلة من متعدد.", "__TERM_1__ __TERM_1__"])
def test_ordinary_translation_cannot_drop_or_duplicate_a_protected_scientific_term(
    service, monkeypatch, text
):
    install(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"message": {"content": json.dumps({"text": text})}}
        ),
    )
    with pytest.raises(ValueError, match="Protected translation token"):
        service.generator.translate_ordinary_text("Multiple-choice questions.", "ar")


def test_source_audits_bind_each_count_to_its_own_request_without_prefilling_verdicts(
    service, monkeypatch
):
    question = "How many models and providers were evaluated?"
    requests = {}
    audits = []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        if user.get("task") == "semantic-grounding-audit":
            item = user["items"][0]
            audits.append(item)
            assert item["requested_facet"] == requests[item["claim"]]
            assert item["question_context"] == question
            assert "question_part" not in item  # Experimental extra verdict remains optional.
            assert "supported" not in item and "complete" not in item
            result = positive(user)
        else:
            role = user["answer_contract"]["requested_entities"][0]
            text = "23 models" if role == "model" else "9 providers"
            requests[text] = user["question"]
            key = next(iter(user["question_parts"]))
            result = {key: {"text": text, "citations": [{"ref_id": "S1:Q1"}]}}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    answer = service.generator.generate(
        question, "en", hits("We evaluated 23 models from 9 providers.")
    )
    assert answer.mode == "ollama" and len(audits) == 2
    assert len(set(requests.values())) == 2


def test_question_context_cannot_override_a_negative_source_judge(service, monkeypatch):
    question = "How many models were evaluated?"
    rejected = []

    def respond(request):
        user = json.loads(json.loads(request.content)["messages"][-1]["content"])
        if user.get("task") == "semantic-grounding-audit":
            rejected.append(user)
            assert user["items"][0]["requested_facet"] == question
            result = positive(user)
            result["verdicts"][0].update(supported=False, complete=False, direct=False)
        else:
            result = {"statements": [{"text": "23 models", "citations": [{"ref_id": "S1:Q1"}]}]}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    with pytest.raises(GenerationUnavailable, match="Semantic grounding"):
        service.generator.generate(question, "en", hits("We evaluated 23 models."))
    assert len(rejected) == 2


@pytest.mark.parametrize("question", ["ما صيغة الأسئلة؟", "كيف صيغت الأسئلة؟"])
def test_english_grounding_hint_never_certifies_untranslated_internal_prose(
    service, monkeypatch, question
):
    seen = []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        assert user.get("task") != "semantic-grounding-audit"
        seen.append(body)
        schema = body["format"]["properties"]["statements"]["items"]["properties"]["text"]
        assert schema["pattern"] == r"^[A-Za-z0-9].*$"
        result = {
            "statements": [{"text": "الأسئلة اختيار من متعدد.", "citations": [{"ref_id": "S1:Q1"}]}]
        }
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    with pytest.raises(GenerationUnavailable, match="expected English"):
        service.generator.generate(question, "ar", hits("The data uses multiple-choice questions."))
    assert len(seen) == 2


def test_protected_prompt_terms_cannot_swap_their_accuracies(service, monkeypatch):
    source = "Few-shot CoT achieves 91% versus zero-shot 35%, a 2.6x ratio."
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        assert json.loads(body["messages"][-1]["content"]) == (
            "__TERM_1__ achieves 91% __TERM_2__ __TERM_3__ 35%, a 2.6x ratio."
        )
        result = {"text": "تحقق __TERM_3__ دقة 91% __TERM_2__ __TERM_1__ بدقة 35%، بنسبة 2.6 ضعف."}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    translated = service.generator.translate_ordinary_text(source, "ar")
    with pytest.raises(ValueError, match="swapped condition"):
        service.generator.audit_answer_translation(
            {"text": source, "citations": [{"ref_id": "S1:Q1"}]},
            translated,
            hits(source),
            "Compare few-shot CoT and zero-shot accuracies.",
            1,
        )
    assert len(seen) == 1  # Refuse before reverse translation or a permissive judge.


def test_reverse_prompt_terminology_keeps_cot_distinct_from_its_base_strategy(service, monkeypatch):
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append(json.loads(body["messages"][-1]["content"]))
        result = {"text": "Few-shot __TERM_1__ and zero-shot __TERM_2__."}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    install(monkeypatch, respond)
    service.generator.translate_ordinary_text(
        "سلسلة التفكير بأمثلة قليلة (CoT) وسلسلة التفكير دون أمثلة (CoT).", "en"
    )
    assert "chain-of-thought" not in seen[0]
    assert "__TERM_1__" in seen[0] and "__TERM_2__" in seen[0]
    assert "سلسلة التفكير" not in seen[0]


def test_ordinary_term_may_already_be_faithfully_expanded_without_changing_name_rules():
    from rag.generation.ollama import expand_scientific_terms, protect_answer_identifiers

    _, bindings = protect_answer_identifiers("Multiple-choice questions.", translate_terms=True)
    text = "أسئلة اختيار من متعدد."
    assert expand_scientific_terms(text, bindings, allow_literal_terms=True) == text
    with pytest.raises(ValueError, match="Protected translation token"):
        expand_scientific_terms(text, bindings)  # Existing strict protocol remains the default.
    _, bindings = protect_answer_identifiers(
        "Multiple-choice questions for QWEN2.", translate_terms=True
    )
    with pytest.raises(ValueError, match="Protected translation token"):
        expand_scientific_terms(
            "أسئلة اختيار من متعدد لـQWEN2.", bindings, allow_literal_terms=True
        )


@pytest.mark.parametrize(
    "source,text",
    [
        ("Multiple-choice questions.", "اختيار من متعدد واختيار من متعدد"),
        ("Multiple-choice and multiple-choice.", "اختيار من متعدد"),
        ("Few-shot CoT and few-shot.", "سلسلة التفكير بأمثلة قليلة (CoT)"),
    ],
)
def test_literal_terms_cannot_duplicate_or_reuse_one_phrase_for_two_source_terms(source, text):
    from rag.generation.ollama import expand_scientific_terms, protect_answer_identifiers

    _, bindings = protect_answer_identifiers(source, translate_terms=True)
    with pytest.raises(ValueError, match="Protected translation token"):
        expand_scientific_terms(text, bindings, allow_literal_terms=True)
