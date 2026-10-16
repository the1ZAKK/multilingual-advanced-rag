"""Opt-in local diagnosis: preserve actual Ollama wire responses and validator errors."""

import json
import logging
from pathlib import Path

import httpx
from diagnose_3lm import QUESTIONS

from rag.config import Settings
from rag.generation.ollama import GenerationUnavailable, OllamaGenerator
from rag.models import ChatResponse


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="trace")
    args = parser.parse_args()
    directory = Path("reports/3lm-diagnosis")
    logging.basicConfig(
        level=logging.INFO,
        filename=directory / f"{args.phase}-generation.log",
        encoding="utf-8",
        format="%(levelname)s %(name)s %(message)s",
    )
    query = QUESTIONS[1]
    real_client = httpx.Client
    with real_client(timeout=180, trust_env=False) as client:
        result = client.post(
            "http://localhost:8000/chat",
            json={
                "query": query,
                "use_llm": False,
            },
        )
        result.raise_for_status()
        evidence = ChatResponse.model_validate(result.json()).evidence
    records = []

    class RecordingClient(real_client):
        def post(self, url, **kwargs):
            response = super().post(url, **kwargs)
            records.append({"request": kwargs.get("json"), "response": response.json()})
            (directory / f"{args.phase}-generation-wire.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return response

    httpx.Client = RecordingClient
    try:
        answer, _ = OllamaGenerator(Settings()).generate(query, "en", evidence)
        print(answer.encode("ascii", "backslashreplace").decode())
    except GenerationUnavailable as exc:
        print(str(exc), repr(exc.__cause__))
    finally:
        httpx.Client = real_client


if __name__ == "__main__":
    main()
