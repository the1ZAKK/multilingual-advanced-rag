import json

import httpx
import pytest

from rag.generation.ollama import GenerationUnavailable
from rag.models import Chunk, Hit


@pytest.mark.parametrize(
    "proposal",
    ["complete", "missing_ref", "omitted_count", "swapped", "negative_judge", "invalid_slot"],
)
def test_two_quantity_decoding_keeps_full_validation_and_does_not_repair_citations(
    service, monkeypatch, proposal
):
    source = "We used 67 distinct tasks. Our evaluation comprises 2,040 individual test cases."
    evidence = [
        Hit(
            chunk=Chunk(
                chunk_id="independent:p1:0",
                document_id="independent",
                document_name="Independent.pdf",
                page=1,
                text=source,
                normalized=source,
                start=0,
                end=len(source),
            ),
            score=1,
        )
    ]
    audits, distinctions = [], []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        if user.get("task") == "semantic-grounding-audit":
            audits.append(user)
            result = {
                "verdicts": [
                    dict(
                        item_id=user["items"][0]["item_id"],
                        reason="Independent judge fixture.",
                        supported=proposal != "negative_judge",
                        complete=True,
                        direct=True,
                    )
                ]
            }
        else:
            part = next(iter(user["question_parts"]))
            contract = user["answer_contract"]
            if contract["kind"] == "quantity-distinction":
                refs = [{"ref_id": "S1:Q1"}, {"ref_id": "S1:Q2"}]
                assert body["format"]["properties"][part]["properties"]["citations"] == {
                    "const": refs
                }
                distinctions.append(user)
                text = "67 distinct tasks; 2,040 individual test cases"
                if proposal == "missing_ref":
                    refs = refs[:1]
                elif proposal == "omitted_count":
                    text = "67 distinct tasks"
                elif proposal == "swapped":
                    text = "2040 distinct tasks; 67 individual test cases"
            else:
                role = contract["requested_entities"][0]
                text, ref = (
                    ("67 distinct tasks", "S1:Q1")
                    if role == "task"
                    else ("2,040 individual test cases", "S1:Q2")
                )
                refs = [{"ref_id": ref}]
            result = {part: dict(text=text, citations=refs)}
            if contract["kind"] == "quantity-distinction" and proposal == "invalid_slot":
                result = {part: None}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(respond))
    )
    question = "How many distinct tasks are used, and how many individual test cases are evaluated? Distinguish the two quantities."
    if proposal == "complete":
        answer = service.generator.generate(question, "en", evidence)
        assert answer.mode == "ollama" and "67" in answer.answer and "2,040" in answer.answer
        assert len(distinctions) == 1 and audits
    else:
        with pytest.raises(GenerationUnavailable):
            service.generator.generate(question, "en", evidence)
        assert len(distinctions) == 2
        assert bool(audits) == (proposal == "negative_judge")
