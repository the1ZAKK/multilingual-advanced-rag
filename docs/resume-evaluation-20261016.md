# Local resumption and generalization evaluation — 2026-10-16

The earlier repeated 3LM checks do not establish general Arabic/English paper-answer quality. This session exposed substantial failures on additional papers. No push, deployment, paid API, package installation, upload or reindex was performed.

## Runtime and preservation

Python 3.11.9 and the existing virtual environment were intact. Restricted sandbox access initially made the interpreter appear missing; the existing environment worked with the required local access. Nothing was reinstalled; `.env` was not edited. Ollama was already serving port 11434 and was reused, including the existing interactive Qwen process. Installed models: `qwen3:1.7b`, `qwen3:4b`, `qwen2.5:3b-instruct`. The configured `qwen3:1.7b` was verified through health and real generation.

API and Streamlit were started on loopback ports 8000 and 8501. A separate evaluation backend on 8001 reuses `reports/architecture-generalization/baseline-index`, containing AceGPT and ArabicNumBench. Only workers identified by their project command lines and process ancestry were stopped for code reloads.

Read-only fingerprints of every live SQLite table and both backups exactly match `reports/grounding-language/preservation-architecture-before.json`: 3 documents, 142 chunk/vector records, metadata and retained PDF bytes. New audit: `reports/grounding-language/preservation-resume-20261016.json`. No models, backups or indexed documents were deleted. The workspace has no `.git` directory, so Git status/diff cannot establish a clean checkout. Existing local fixes were retained and repaired in place.

## Repairs and checks

Initial results: **3 failed, 305 passed, 1 skipped**; Ruff found import ordering and an unused variable. Repairs address contradictory contribution counts raising `IndexError`, Arabic declaration text consuming the first contribution, and missing-item diagnostics.

Dashboard testing exposed an untested API call passing `query=` to a table renderer accepting only evidence and language. The call now respects that interface. This restores service availability; it does **not** implement focused table selection. A numerical-chat service regression exercises the boundary. Extractive fallback now uses exact native common substrings when reordered chunks cannot be quoted contiguously; independent paragraphs are retained. A regression covers reversed blocks.

The evaluation runner accepts UTF-8 BOM reference files and records request errors instead of aborting with `KeyError`. Initial partial records remain in `reports/resume-20261016/heldout-first-attempt.json`.

Final production-code verification: **310 passed, 1 skipped, 1 warning; Ruff clean**. The optional RAGAS test is skipped; live RAGAS judgments were not run. The warning is the existing FastAPI/Starlette HTTPX TestClient deprecation. These checks do not prove the answer-quality failures below are solved.

## Method and observed results

Actual Streamlit AppTest submissions call the local HTTP API and Qwen. This is dashboard execution, not pixel-based browser verification. Records separate language, native-PDF/indexed citation provenance, displayed-source grouping, displayed mode, generation route and warning. Exact quotes do not validate row/column bindings or semantic entailment. Vacuously valid citations on refusals are not evidence of correctness.

The October 10 frozen references have prior baseline exposure. Their 15 single-run controls are additional-paper generalization checks relative to 3LM, not pristine unseen-paper tests. Four exact questions were newly authored on October 16 from original PDF passages before submission: English and Arabic for each paper. Those questions are fresh; the papers are already exposed. No broad benchmark accuracy, fluent-Arabic or statistical reliability claim follows.

Original passages and rendered pages were inspected, including AceGPT page 4 and ArabicNumBench page 3 for fresh cases, and retained AceGPT page 5 to verify table geometry. New references, source text/renders and actual responses are under `reports/resume-20261016/`.

Independent review of the completed 15-control batch found:

| Cases | Correctness | Completeness | Citation provenance |
|---|---|---|---|
| AceGPT authored list, English and Arabic | Cautious evidence-local refusal; no invented list | Fail: page 2 has an explicit three-item list | Pass both |
| AceGPT ACVA and Table 8 questions | Fail: wrong table caption/conditions or unrelated tables; score contaminates one model identity | Fail: requested rows/comparisons omitted | Pass both |
| AceGPT adaptation and introduced chat model | Faithful excerpts contain the actual pipeline/introduction and baseline distinction | Required content present, but verbose fallback rather than synthesized explanation | Pass both |
| AceGPT Arabic summary | Faithful English excerpts contain the problem and pipeline | Content present; requested Arabic explanation fails | Pass |
| AceGPT localized reward model | Faithful excerpts describe localized preference training/PPO | Partial: missing the chosen/rejected output definition and linear-head explanation | Pass |
| AceGPT unsupported PRM | Correctly declines inference of step supervision | Partial: omits actual response-preference reward-model distinction | Pass |
| ArabicNumBench problem/method | Faithful excerpts describe the gap and high-level counts | Fail: detailed prompting protocol and metric definitions missing | Pass |
| ArabicNumBench Arabic metrics and numerical results | Faithful excerpts; no reliable requested numerical comparison | Fail: metric formulas/example or eight model-condition values missing; Arabic metrics also fails language | Pass both |
| ArabicNumBench supported benchmark introduction | Direct author statement is present in source excerpts | Core support present, despite clutter and generation failure | Pass |
| ArabicNumBench author list/findings | Correctly avoids inventing contributions | Fail: does not deliver complete three-item Key Findings continuation | Pass |
| ArabicNumBench unsupported PRM | Correct evidence-local insufficiency | Covers requested refusal/related-work distinction at a general level | Pass |

All 15 controls passed native citation provenance and displayed-source grouping. Language heuristic passed 13/15; failures were Arabic outputs falling back to English excerpts. Routes were 8 `extractive`, 5 `insufficient-evidence`, and 2 `source-tables`. These method counts are not correctness scores. Case-level judgments must retain the distinction between cautious refusal, faithful source excerpts, wrong relational claims and complete answers.

Extractive output is not automatically a semantic failure: review credits requested information actually present in it. Conversely, it does not receive completeness credit merely for mentioning the right topic. Content coverage, requested-language delivery and synthesis usability are separate dimensions. No single binary aggregate accuracy is claimed.

The ACVA output labels Table 5 values with the neighboring Table 6 caption, includes Table 18 few-shot results, and omits AceGPT-13B-chat. Independently inspected page 5 Table 5 gives AceGPT **75.02%/74.62%** versus GPT-3.5 Turbo **75.57%/79.03%** on All/Clean, zero-shot F1. Table 8 output instead returns unrelated ACVA results. These are semantic failures despite exact quotes.

The positive AceGPT-chat control exposes judge inconsistency: `supported=True, complete=False, direct=True` with a reason explicitly confirming the SFT/RLAIF introduction relation. The system falls back. It must not be described as absent source support. Other generation failures include numeric checks, JSON decoding, uncited sentences and citation-set mismatches; precise warnings/logs are preserved.

## Labels and next priorities

The four new exact questions completed against the final tested code:

| New question | Content correctness and completeness | Language | Native citations |
|---|---|---|---|
| AceGPT position bias (English) | Faithful excerpts cover answer-order switching, consistent preferences, 12K pairs, 800 examples and 0.84 correlation; fragmented and cluttered | Pass | Pass |
| AceGPT chosen/rejected preference outputs (Arabic) | Missing binary head and chosen/rejected definitions; retrieved left-column training text misses right-column explanation | Fail | Pass |
| ArabicNumBench limitations (English) | Complete original limitations passage: 210 pilot cases, granular analysis, everyday scope, scientific/legal/dialectal extensions, limited prompting space | Pass | Pass |
| ArabicNumBench API/temperature/demonstrations (Arabic) | Incorrect numerical-table routing; OpenRouter, temperature 0.0 and three demonstrations absent from excerpts | Fail | Pass |

All four returned `extractive` with explicit failure warnings; all four displayed the mode and passed source grouping. Two contain the requested factual content, despite generation failure. Across both batches: **19/19 native citation checks, 15/19 language heuristic checks**, with **12 extractive, 5 insufficient-evidence, 2 source-tables**. These are validation/method counts, not aggregate semantic accuracy. New questions used automatic language selection once; explicit-language repeats were not performed.

Saved full answers: `heldout-dashboard.json` and `fresh-dashboard.json`. The case-level manual source review is `independent-quality-review.json`, with answer hashes and separate correctness/coverage/language/provenance judgments. The 15-control batch ran before the final paragraph-preservation refinement; the four new questions and regression suite ran with that final refinement. No production code changed after the 310-test run.

Final live read-only hybrid search returned a hit using the cached Sentence Transformer (first-search latency about 17.5 seconds, including lazy loading). API health and Streamlit `/_stcore/health` both succeeded after the final restart. The temporary evaluation backend is stopped after the audit; requested services remain on 8000, 8501 and 11434.

`ollama` means generated prose; `source-contributions` full source author statements; `translated-contributions` translated author statements; `source-tables` deterministic rendering of extracted relationships; `insufficient-evidence` a retrieved-support limitation; `extractive` labeled source excerpts with any failure warning. The dashboard displays the mode and its explanation. **No mode is a correctness verdict.**

Remaining priorities: generic inline-list declaration/section recovery, caption-to-table geometry, clean entity extraction, requested table/model/condition selection, full findings continuations and reliable semantic judging. Preserve the 3LM controls and newly exposed failures as separate targets. Never put paper-specific gold answers into production. Broader relevance annotations, repeated runs, fluent Arabic review, live RAGAS and pristine unseen-paper tests remain open.
