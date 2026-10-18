import json

import httpx
import pytest

from rag.generation.questions import flatten_parts, question_parts
from rag.models import ChatRequest


@pytest.mark.parametrize(
    "question,count",
    [
        ("What gap is addressed, and what sources are used?", 2),
        ("How is content curated, and what mechanism is used?", 2),
        ("ما المشكلة، وما المصادر المستخدمة؟", 2),
        ("Which models and datasets are evaluated?", 1),
        ("How do X and Y work?", 1),
        ("Does it assess retrieval? Distinguish current scope from future work.", 2),
        ("هل يقيس الاسترجاع؟ ميّز بين النطاق الحالي والعمل المستقبلي.", 2),
    ],
)
def test_only_explicit_interrogatives_split(question, count):
    assert len(question_parts(question)) == count


def test_missing_or_extra_part_rejected():
    for value in ({"part_1": {}}, {"part_1": {}, "part_2": {}, "part_3": {}}):
        with pytest.raises(ValueError, match="every supplied question part"):
            flatten_parts(value, ["first", "second"])


def test_multipart_slots_reach_existing_citation_and_semantic_validation(
    service, pdf_bytes, monkeypatch
):
    service.ingest(pdf_bytes(), "source.pdf")
    real_client = httpx.Client
    calls = []

    def handle(request):
        data = json.loads(request.content)
        user = json.loads(data["messages"][-1]["content"])
        calls.append(user)
        if user.get("task") == "semantic-grounding-audit":
            result = {
                "verdicts": [
                    dict(
                        item_id=i["item_id"],
                        supported=True,
                        complete=True,
                        direct=True,
                        reason="Contract fixture.",
                    )
                    for i in user["items"]
                ]
            }
        else:
            assert set(data["format"]["properties"]) == set(user["question_parts"])
            assert len(user["question_parts"]) == 1
            assert user["question"] in user["question_parts"].values()
            ref = user["evidence"][0]["quotes"][0]["ref_id"]
            result = {
                key: dict(
                    text="Hybrid retrieval combines vector and keyword search.",
                    citations=[dict(ref_id=ref)],
                )
                for key in user["question_parts"]
            }
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle))
    )
    result = service.chat(
        ChatRequest(query="What combines search methods, and how does retrieval work?")
    )
    assert result.generation_mode == "ollama"
    assert len([c for c in calls if c.get("task") == "semantic-grounding-audit"]) == 2
    assert result.citations


def test_parts_over_budget_never_silently_dropped(service, pdf_bytes):
    service.ingest(pdf_bytes(), "source.pdf")
    service.generator.settings = service.settings.model_copy(update={"max_answer_statements": 1})
    result = service.chat(ChatRequest(query="What is retrieval, and how does it work?"))
    assert result.generation_mode == "extractive"
    assert "Question parts exceed" in result.warning


def test_part_focus_keeps_immutable_quotes_and_does_not_mutate_input():
    from copy import deepcopy

    from rag.generation.questions import focus_sources

    sources = [
        dict(label="S1", page=1, quotes=[dict(ref_id="S1:Q1", quote="Oriole is a benchmark.")]),
        dict(
            label="S2",
            page=2,
            quotes=[dict(ref_id="S2:Q1", quote="Oriole content is reviewed by experts.")],
        ),
    ]
    original = deepcopy(sources)
    result = focus_sources(sources, "How is Oriole content curated?", limit=1)
    assert result[0]["label"] == "S2"
    assert result[0]["quotes"][0] == original[1]["quotes"][0]
    assert sources == original
    assert focus_sources(sources, "كيف تنظم البيانات؟") == sources
    assert focus_sources(sources, "كيف ينظم Oriole المحتوى؟") == sources


def test_plain_question_does_not_receive_multipart_schema_directions():
    from rag.generation.ollama import BASIC_SYSTEM, generation_instructions

    assert generation_instructions({}, "en") == BASIC_SYSTEM
    assert "question_parts" not in generation_instructions({}, "en")
    assert "question_parts" in generation_instructions({}, "en", multipart=True)
