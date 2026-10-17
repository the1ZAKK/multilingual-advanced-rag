"""Stable public response labels; a label describes the method, not correctness."""

from typing import Literal

ResponseMode = Literal[
    "ollama",
    "translated-answer",
    "source-contributions",
    "translated-contributions",
    "source-tables",
    "insufficient-evidence",
    "extractive",
]

MODE_DESCRIPTIONS = {
    "ollama": "Qwen-generated prose; source citations and claim support checked separately.",
    "translated-answer": "Source-grounded English claims translated into Arabic, with unchanged source references, numeric checks and an independent reverse-translation audit.",
    "source-contributions": "Complete author contribution statements rendered from the source.",
    "translated-contributions": "Author contribution statements translated with source bindings and meaning checks.",
    "source-tables": "Source table rows and columns rendered without Qwen; the answer states the selection scope.",
    "insufficient-evidence": "Retrieved evidence is insufficient for the requested claim; this does not establish paper-wide absence.",
    "extractive": "Source excerpts; a complete generated answer was not produced. See any failure warning.",
}
