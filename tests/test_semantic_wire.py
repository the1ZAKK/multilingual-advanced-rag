import json
from copy import deepcopy

import httpx
import pytest

from rag.generation.ollama import unpack_semantic_verdicts, validate_semantic_verdicts


def item(**extra):
    return dict(item_id="item-1", require_complete=True, require_direct=True, **extra)


def wire():
    return {
        "verdicts": [
            dict(
                item_id="item-1",
                reason="The quoted premise explicitly supports this claim.",
                verdict=dict(supported=True, complete=True, direct=True),
            )
        ]
    }


def test_wire_adapter_keeps_exact_boolean_values_and_does_not_mutate_evidence():
    data = wire()
    original = deepcopy(data)
    flat = unpack_semantic_verdicts(data, [item()])
    validate_semantic_verdicts(flat, [item()])
    assert data == original
    assert flat["verdicts"][0] == {
        "item_id": "item-1",
        "reason": original["verdicts"][0]["reason"],
        "supported": True,
        "complete": True,
        "direct": True,
    }
    assert unpack_semantic_verdicts(flat, [item()]) == flat


@pytest.mark.parametrize(
    "flag,value",
    [
        ("supported", False),
        ("complete", False),
        ("direct", False),
        ("supported", "true"),
        ("complete", 1),
        ("direct", None),
    ],
)
def test_positive_reason_cannot_override_false_or_nonboolean_flags(flag, value):
    data = wire()
    data["verdicts"][0]["verdict"][flag] = value
    with pytest.raises(ValueError, match="Semantic|semantic"):
        validate_semantic_verdicts(unpack_semantic_verdicts(data, [item()]), [item()])


def test_nested_wire_cannot_omit_required_flags_or_smuggle_overrides():
    for change in ("missing", "override", "mixed"):
        data = wire()
        verdict = data["verdicts"][0]
        if change == "missing":
            del verdict["verdict"]["complete"]
        elif change == "override":
            verdict["verdict"]["item_id"] = "item-1"
        else:
            verdict["complete"] = True
        with pytest.raises(ValueError, match="semantic verdict"):
            unpack_semantic_verdicts(data, [item()])


def test_optional_question_coverage_remains_a_required_independent_flag_when_enabled():
    requested = item(question_part="What methods?")
    data = wire()
    with pytest.raises(ValueError, match="flags"):
        unpack_semantic_verdicts(data, [requested])
    data["verdicts"][0]["verdict"]["answers_question"] = False
    with pytest.raises(ValueError, match="does not cover question part"):
        validate_semantic_verdicts(unpack_semantic_verdicts(data, [requested]), [requested])


@pytest.mark.parametrize(
    "rejected_flag", [None, "supported", "complete", "direct", "answers_question"]
)
def test_live_audit_wire_explains_before_deciding_and_preserves_negative_flags(
    service, monkeypatch, tmp_path, rejected_flag
):
    requested = item(
        claim="17 observations were collected.",
        premises=["We collected 17 observations."],
        question_context="كم عدد الملاحظات؟",
        requested_facet="كم عدد الملاحظات؟",
    )
    if rejected_flag == "answers_question":
        requested["question_part"] = requested["requested_facet"]
        requested["question_answer"] = requested["claim"]
    response_data = wire()
    if rejected_flag:
        response_data["verdicts"][0]["verdict"][rejected_flag] = False
    monkeypatch.setenv("RAG_GENERATION_TRACE_DIR", str(tmp_path))
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        schema = body["format"]["properties"]["verdicts"]["items"]
        # The pinned provider sorts keys. The explanation must be emitted
        # before the nested verdict under either insertion or sorted order.
        assert list(schema["properties"]) == ["item_id", "reason", "verdict"]
        assert sorted(schema["properties"]) == ["item_id", "reason", "verdict"]
        expected = {"supported", "complete", "direct"}
        if rejected_flag == "answers_question":
            expected.add("answers_question")
        assert set(schema["properties"]["verdict"]["required"]) == expected
        assert schema["additionalProperties"] is False
        assert schema["properties"]["verdict"]["additionalProperties"] is False
        supplied = json.loads(body["messages"][1]["content"])
        assert supplied == {"task": "semantic-grounding-audit", "items": [requested]}
        return httpx.Response(200, json={"message": {"content": json.dumps(response_data)}})

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(respond))
    )
    if rejected_flag:
        with pytest.raises(
            ValueError, match="Semantic grounding failed|does not cover question part"
        ):
            service.generator.audit_semantic_item(requested)
    else:
        service.generator.audit_semantic_item(requested)
    assert len(calls) == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "generation-wire.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    raw = next(event for event in events if event["stage"] == "semantic-audit-wire-response")
    assert json.loads(raw["result"]) == response_data
