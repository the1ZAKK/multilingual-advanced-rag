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
    with pytest.raises(GenerationUnavailable, match="missing meaning"):
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
