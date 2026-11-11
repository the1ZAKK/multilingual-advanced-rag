import json

import httpx
import pytest

from rag.generation.ollama import GenerationUnavailable
from rag.models import Chunk, Hit


@pytest.mark.parametrize("proposal", ["complete", "negative_complete", "missing_count"])
def test_each_claim_keeps_its_full_question_and_own_facet_without_skipping_other_slots(
    service, monkeypatch, proposal
):
    text = (
        "We evaluated 23 models. "
        "Few-shot CoT achieves 2.6x higher accuracy than zero-shot approaches (91% vs 35%)."
    )
    question = (
        "How many models were evaluated? Compare few-shot CoT accuracy with zero-shot accuracy."
    )
    hits = [
        Hit(
            chunk=Chunk(
                chunk_id="independent:p1:0",
                document_id="independent",
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
    audits = []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        if user.get("task") == "semantic-grounding-audit":
            item = user["items"][0]
            audits.append(item)
            assert item["question_context"] == question
            assert "requested_facet identifies the" in body["messages"][0]["content"]
            assert "not evidence" in body["messages"][0]["content"]
            if "23" in item["claim"]:
                assert item["requested_facet"] == "How many models were evaluated?"
            else:
                assert item["requested_facet"] == question.split("? ")[1]
            result = {
                "verdicts": [
                    dict(
                        item_id=item["item_id"],
                        supported=True,
                        complete=not (proposal == "negative_complete" and "91%" in item["claim"]),
                        direct=True,
                        reason="The independent judge explicitly decides this claim's completeness.",
                    )
                ]
            }
        else:
            key = next(iter(body["format"]["properties"]))
            if key == "part_1":
                value = "models" if proposal == "missing_count" else "23 models"
                ref = "S1:Q1"
            else:
                assert key == "part_2"
                value = (
                    "few-shot chain-of-thought accuracy: 91% versus zero-shot accuracy: 35% (2.6x)"
                )
                ref = "S1:Q2"
            result = {key: dict(text=value, citations=[dict(ref_id=ref)])}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(respond))
    )
    if proposal == "complete":
        result = service.generator.generate(question, "en", hits)
        assert result.mode == "ollama" and "23 models" in result.answer and len(audits) == 2
    else:
        with pytest.raises(GenerationUnavailable):
            service.generator.generate(question, "en", hits)
        assert bool(audits) == (proposal == "negative_complete")
