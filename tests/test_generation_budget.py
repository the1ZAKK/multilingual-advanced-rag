import pytest

from rag.generation import ollama
from rag.generation.ollama import GenerationUnavailable, OllamaGenerator, bounded_generation


def test_generation_budget_covers_multiple_calls_and_resets_after_failure(settings, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(ollama, "monotonic", lambda: clock[0])
    generator = OllamaGenerator(
        settings.model_copy(update={"llm_timeout": 40, "llm_total_timeout": 50})
    )
    timeouts = []

    @bounded_generation
    def chain(self):
        timeouts.append(self.request_timeout())
        clock[0] += 35
        timeouts.append(self.request_timeout())
        clock[0] += 16
        self.request_timeout()

    with pytest.raises(GenerationUnavailable, match="budget exhausted"):
        chain(generator)
    assert timeouts == [40, 15]
    assert generator.request_timeout() == 40


def test_generated_sentences_keep_refs_and_are_audited_individually():
    from rag.generation.ollama import split_generated_statements

    statements = [
        {
            "text": "The corpus has school questions. It covers multiple subjects.",
            "citations": [{"ref_id": "S1:Q1"}],
        }
    ]
    parts = split_generated_statements(statements)
    assert [p["text"] for p in parts] == [
        "The corpus has school questions.",
        "It covers multiple subjects.",
    ]
    assert all(p["citations"] == statements[0]["citations"] for p in parts)
    assert len(statements) == 1
