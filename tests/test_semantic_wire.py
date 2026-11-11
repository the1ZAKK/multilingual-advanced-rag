from copy import deepcopy

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
