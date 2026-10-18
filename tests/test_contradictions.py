import pytest

from rag.generation.contradictions import reject_polarity_conflicts


def test_explicit_same_subject_conflict_is_rejected():
    statements = [
        {
            "text": "The initial Oriole benchmark does not assess retrieval of facts and laws from external knowledge bases."
        },
        {
            "text": "The initial Oriole benchmark assesses retrieval of facts and laws from external knowledge bases, with future work focusing on more cases."
        },
    ]
    with pytest.raises(ValueError, match="Conflicting positive and negative"):
        reject_polarity_conflicts(statements)


@pytest.mark.parametrize(
    "positive",
    [
        "Future tasks will assess retrieval of facts and laws from external knowledge bases.",
        "A different benchmark assesses retrieval of facts and laws from external knowledge bases.",
        "The initial Oriole benchmark assesses arithmetic accuracy.",
    ],
)
def test_future_work_distinct_subjects_and_objects_not_conflated(positive):
    reject_polarity_conflicts(
        [
            {
                "text": "The initial Oriole benchmark does not assess retrieval of facts and laws from external knowledge bases."
            },
            {"text": positive},
        ]
    )


def test_conflict_reaches_existing_retry_before_model_judge(service, pdf_bytes, monkeypatch):
    import json

    import httpx

    from rag.models import ChatRequest

    service.ingest(
        pdf_bytes(
            "The initial Oriole benchmark does not assess retrieval of facts and laws from external knowledge bases."
        ),
        "scope.pdf",
    )
    real_client = httpx.Client
    generated = []

    def handle(request):
        user = json.loads(json.loads(request.content)["messages"][-1]["content"])
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
            generated.append(user)
            ref = user["evidence"][0]["quotes"][0]["ref_id"]
            sentences = [
                "The initial Oriole benchmark does not assess retrieval of facts and laws from external knowledge bases."
            ]
            if len(generated) == 1:
                sentences.append(
                    "The initial Oriole benchmark assesses retrieval of facts and laws from external knowledge bases."
                )
            result = {"statements": [dict(text=t, citations=[dict(ref_id=ref)]) for t in sentences]}
        return httpx.Response(200, json={"message": {"content": json.dumps(result)}})

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle))
    )
    result = service.chat(ChatRequest(query="Describe the initial Oriole retrieval scope."))
    assert len(generated) == 2
    assert "Conflicting positive and negative" in generated[1]["validation_error"]
    assert result.generation_mode == "ollama"
    assert "does not assess" in result.answer
