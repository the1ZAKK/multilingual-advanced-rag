# Implementation and validation tracker

## Current cloud continuation — October 2026

- [x] Scoped retrieval, table binding/selection, bounded generation and evaluation audit fixes.
- [x] Python 3.11/3.12 suites, live learned models and real browser screenshots.
- [x] Portable source package and backup-first updater with conflict refusal.
- [ ] Reliable ordinary Arabic generation, pristine unseen-paper/human evaluation and Windows execution.

See [current validation](docs/validation-20261016.md). Sections below are historical;
their old model-access blockers and counts do not describe the current cloud run.


## Resumed local validation — 2026-10-16
- [x] Reuse existing Python and Ollama; start API/dashboard on 8000/8501 and verify configured Qwen.
- [x] Verify live corpus/vector/source and both backup fingerprints unchanged without reindexing.
- [x] Repair diagnosed contribution regressions, table-call server error, native-quote fallback and evaluator failures; 310 tests passed, 1 optional test skipped; Ruff clean.
- [x] Run 15 existing additional-paper controls and four new Arabic/English questions on AceGPT/ArabicNumBench; independently review content coverage and relational errors separately from 19/19 native citation checks. Preserve labeled source response modes.
- [ ] General Arabic/English answer correctness and completeness across additional papers: actual runs expose unresolved lists, table binding/selection, language and judge failures. See [resumption report](docs/resume-evaluation-20261016.md).
- [ ] Pristine unseen-paper benchmark and live RAGAS judgments. Older environment blockers below are historical; local Qwen and cached learned embeddings are now available.

A checked implementation item means working code exists and its relevant automated/offline
checks passed. Live model-dependent checks remain explicitly open; no phase is declared
fully validated while those checks are pending.

## Phase 1 — Environment and MVP
- [x] Inspect cloud Python/Git/RAM/GPU/tools (CPU-only, Python 3.12).
- [x] Create isolated local project; preserve sibling research documents.
- [x] Locked CPU dependencies, PDF validation, Arabic normalization, original page chunks.
- [x] FAISS indexing/search, persistence and cited extractive question-answer interface.
- [x] Sentence Transformer/Qwen3 embedding adapter and query instruction tests.
- [ ] Download and run real multilingual embeddings (cloud proxy denied model domains).

## Phase 2 — Advanced retrieval
- [x] Arabic-aware BM25, configurable case folding, positive smoothed IDF.
- [x] RRF, deduplication, component scores and configuration.
- [x] Optional lexical baseline and separate CrossEncoder adapter; mocked integration tests.
- [ ] Live cross-encoder latency/relevance validation (same model-download blocker).

## Phase 3 — Evaluation
- [x] Annotation schema; rejection of unknown passage IDs; graded relevance metrics.
- [x] Four-pipeline synthetic demo; CSV/JSON exports; independent CLI reproduction.
- [x] Query/reranking/response latency and clearly labeled lexical answer proxies.
- [x] Optional local Ragas integration; compatible dependency versions and mocked wiring test.
- [ ] Real Arabic/English/cross-lingual manually annotated benchmark.
- [ ] Live Qwen/Ragas faithfulness and correctness judgments (Ollama/Qwen absent).

## Phase 4 — Engineering
- [x] Seven required FastAPI endpoints, validation, per-file errors and structured logs.
- [x] Streamlit documents/chat/comparison/evaluation, escaped bidi text, interaction test.
- [x] SQLite transactions, duplicate checks, deletion, checksum-validated FAISS recovery.
- [x] Automated unit/integration/end-to-end tests and live local functional probe.
- [x] Docker build and non-root container API functional checks.
- [x] Compose validation and CPU dependency lock; optional trusted build CA support.
- [ ] Validate learned-model generation inside Docker on a machine with model access.

## Phase 5 — Portfolio
- [x] English README, Chinese description, architecture diagram, Windows instructions.
- [x] CI definition for Windows/Linux and Python 3.11/3.12 (local workflow file only).
- [x] MIT application license, security, evaluation and troubleshooting documentation.
- [x] Honest engineering report, CV statements and Arabic PRM research directions.
- [ ] Capture real dashboard screenshots (capture checklist provided).
- [ ] Run CI on GitHub and validate on the user's actual Windows laptop.
- [x] User explicitly approved delivery to a new GitHub branch, preserving main.
- [ ] Public deployment or other publication requires separate explicit approval.
