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


@pytest.mark.parametrize("cover_question", [True, False])
def test_multipart_slots_reach_existing_citation_and_semantic_validation(
    service, pdf_bytes, monkeypatch, cover_question
):
    service.ingest(pdf_bytes(), "source.pdf")
    service.generator.settings = service.settings.model_copy(
        update={"question_coverage_audit": True}
    )
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
                        answers_question=cover_question,
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
    audits = [c for c in calls if c.get("task") == "semantic-grounding-audit"]
    assert all(i["question_part"] and i["question_answer"] for c in audits for i in c["items"])
    if cover_question:
        assert result.generation_mode == "ollama"
        assert len(audits) == 2
    else:
        # A source-supported but irrelevant slot is rejected on both attempts.
        assert result.generation_mode == "extractive"
        assert "does not cover question part" in result.warning
        assert len(audits) == 2
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


@pytest.mark.parametrize("answer_flag", [None, "true", False])
def test_question_coverage_requires_explicit_true_boolean(answer_flag):
    from rag.generation.ollama import validate_semantic_verdicts

    item = dict(item_id="item-1", question_part="What sources are used?", require_direct=False)
    verdict = dict(
        item_id="item-1",
        supported=True,
        complete=True,
        direct=True,
        reason="A figure caption does not identify the source materials.",
    )
    if answer_flag is not None:
        verdict["answers_question"] = answer_flag
    with pytest.raises(ValueError):
        validate_semantic_verdicts({"verdicts": [verdict]}, [item])


def test_split_claims_keep_whole_slot_for_coverage_but_own_premises(monkeypatch):
    from rag.config import Settings
    from rag.generation.ollama import OllamaGenerator
    from rag.models import Chunk, Hit

    text = "The study uses legal documents. It uses synthetic questions. It tests model knowledge."
    chunk = Chunk(
        chunk_id="doc:p1:0",
        document_id="doc",
        document_name="Source.pdf",
        page=1,
        text=text,
        normalized=text,
        start=0,
        end=len(text),
    )
    generator = OllamaGenerator(Settings())
    seen = []
    monkeypatch.setattr(generator, "audit_semantic_item", lambda item: seen.append(item))
    payload = {
        "statements": [
            {"text": sentence, "citations": [{"ref_id": f"S1:Q{i}"}]}
            for i, sentence in enumerate(
                [
                    "The study uses legal documents.",
                    "It uses synthetic questions.",
                    "It tests model knowledge.",
                ],
                1,
            )
        ]
    }
    generator.verify_semantics(
        payload,
        [Hit(chunk=chunk, score=1)],
        {},
        "What sources are used, and what is tested?",
        "en",
        question_parts=["What sources are used?", "What sources are used?", "What is tested?"],
    )
    assert len(seen) == 3
    assert seen[0]["question_answer"] == seen[1]["question_answer"]
    assert "synthetic questions" in seen[0]["question_answer"]
    assert seen[2]["question_answer"] == "It tests model knowledge."
    assert seen[0]["premises"] == ["The study uses legal documents."]
