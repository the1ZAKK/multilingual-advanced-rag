"""Optional local judge evaluation. Explicit invocation only; never downloads on API startup."""

import asyncio
import math

from rag.config import Settings


def score_with_ragas(samples: list[dict], settings: Settings) -> dict:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
    from ragas import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerCorrectness, Faithfulness
    from ragas.run_config import RunConfig

    llm = LangchainLLMWrapper(
        ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_url,
            temperature=0,
            format="json",
            reasoning=False,
            num_ctx=settings.llm_context_tokens,
            num_predict=1200,
            client_kwargs={"timeout": settings.llm_timeout},
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.embedding_model, model_kwargs={"device": settings.device}
        )
    )
    faithful = Faithfulness(llm=llm)
    correctness = AnswerCorrectness(llm=llm, embeddings=embeddings)
    run_config = RunConfig(timeout=settings.llm_timeout, max_retries=1)
    faithful.init(run_config)
    correctness.init(run_config)

    async def run():
        rows = []
        for item in samples:
            sample = SingleTurnSample(**item)
            row = {"faithfulness": await faithful.single_turn_ascore(sample)}
            if sample.reference:
                row["answer_correctness"] = await correctness.single_turn_ascore(sample)
            if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in row.values()):
                raise ValueError(
                    "Local Ragas judge returned a non-finite or out-of-range score; no replacement result was fabricated"
                )
            rows.append(row)
        return {
            "rows": rows,
            "judge": settings.ollama_model,
            "warning": "Local LLM judge scores are model-dependent estimates; inspect Arabic judge reliability manually.",
        }

    return asyncio.run(run())
