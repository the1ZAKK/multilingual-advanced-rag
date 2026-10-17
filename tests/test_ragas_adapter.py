import pytest

from rag.evaluation.ragas_adapter import score_with_ragas


def test_optional_ragas_local_judge_wiring(settings, monkeypatch):
    pytest.importorskip("ragas")
    import langchain_huggingface
    import langchain_ollama
    import ragas.embeddings
    import ragas.llms
    import ragas.metrics

    seen = []

    class Metric:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            self.initialized = False

        def init(self, run_config):
            assert run_config.max_retries == 1
            self.initialized = True

        async def single_turn_ascore(self, sample):
            assert self.initialized
            assert sample.user_input == "question"
            seen.append(sample.reference)
            return 0.5  # Test double only; never exported as an experiment result.

    monkeypatch.setattr(ragas.metrics, "Faithfulness", Metric)
    monkeypatch.setattr(ragas.metrics, "AnswerCorrectness", Metric)
    monkeypatch.setattr(ragas.llms, "LangchainLLMWrapper", lambda model: model)
    monkeypatch.setattr(ragas.embeddings, "LangchainEmbeddingsWrapper", lambda model: model)
    monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda **kwargs: kwargs)
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", lambda **kwargs: kwargs)
    result = score_with_ragas(
        [
            {
                "user_input": "question",
                "response": "answer",
                "retrieved_contexts": ["evidence"],
                "reference": "reference",
            }
        ],
        settings,
    )
    assert result["judge"] == settings.ollama_model
    assert set(result["rows"][0]) == {"faithfulness", "answer_correctness"}
    assert seen[0]["llm"]["base_url"] == settings.ollama_url
