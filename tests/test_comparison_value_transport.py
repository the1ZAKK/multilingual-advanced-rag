import json
import re

import httpx
import pytest

from rag.generation.ollama import GenerationUnavailable
from rag.models import Chunk, Hit


@pytest.mark.parametrize(
    "proposal",
    ["complete", "missing_ratio", "wrong_ratio", "swapped", "wrong_source", "negative_judge"],
)
def test_source_derived_comparison_values_cannot_bypass_full_validation(
    service, monkeypatch, proposal
):
    text = (
        "Few-shot CoT achieves 2.6x higher accuracy than zero-shot approaches (91% vs 35%). "
        "Few-shot methods achieve 2.2x higher accuracy than zero-shot approaches (80% vs 36%)."
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
    good = "few-shot chain-of-thought accuracy: 91% versus zero-shot accuracy: 35% (2.6x)"
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
                        supported=proposal != "negative_judge",
                        complete=True,
                        direct=True,
                        reason="Independent audit fixture; decoding does not certify support.",
                    )
                ]
            }
        else:
            pattern = body["format"]["properties"]["statements"]["items"]["properties"]["text"][
                "pattern"
            ]
            assert re.fullmatch(pattern, good)
            value = good
            ref = "S1:Q1"
            if proposal == "missing_ratio":
                value = value.replace(" (2.6x)", "")
            elif proposal == "wrong_ratio":
                value = value.replace("2.6x", "2.2x")
            elif proposal == "swapped":
                value = value.replace("91%", "TEMP").replace("35%", "91%").replace("TEMP", "35%")
            elif proposal == "wrong_source":
                ref = "S1:Q2"
            result = {"statements": [dict(text=value, citations=[dict(ref_id=ref)])]}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(respond))
    )
    question = "Compare few-shot CoT accuracy with zero-shot accuracy."
    if proposal == "complete":
        result = service.generator.generate(question, "en", hits)
        assert result.mode == "ollama" and "91%" in result.answer and audits
    else:
        with pytest.raises(GenerationUnavailable):
            service.generator.generate(question, "en", hits)
        assert bool(audits) == (proposal == "negative_judge")
