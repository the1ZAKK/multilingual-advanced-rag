> Historical report. See [current validation](validation-20261016.md) for current checks and limitations.

# Engineering report

## What is implemented

An independent Python package with page-preserving PyMuPDF ingestion; SHA-256 document
identity; stable page/offset chunk IDs; Unicode Arabic normalization; configurable chunking,
limits and BM25 case folding; Sentence Transformer embeddings; normalized FAISS inner-product
search; rank-bm25 lexical search with positive smoothed IDF; deduplicated RRF; optional
CrossEncoder pair scoring or a labeled lexical baseline; evidence budgeting; local Qwen/Ollama
JSON generation; citation-label/exact-quote validation; and an explicit extractive fallback.

SQLite stores original text, normalized chunks, vectors and metadata in transactions.
FAISS caches use corpus fingerprints and checksums and are loaded or rebuilt on restart.
BM25 is deterministically rebuilt without pickle. Deletion rebuilds both indexes.

The backend implements all seven requested endpoints. The dashboard provides documents,
chat history, passage inspection, retrieval comparison and evaluation charts/exports.
The evaluation framework reports graded retrieval metrics and latency, with optional lexical
answer proxies and a separate Ragas/local judge runner. Model adapters do not share an API:
embeddings encode text; rerankers score query/passage pairs; generation uses Ollama HTTP.

## Validation evidence

Cloud machine: Python 3.12.14, Git 2.52.0, ~33 GiB RAM, no detected NVIDIA GPU, Ollama absent.
CPU PyTorch, Transformers, Sentence Transformers, FAISS, BM25, FastAPI, Streamlit and optional
Ragas/LangChain dependencies installed successfully using the committed lock.

- **40 Pytest tests passed**: PDF/Arabic/English processing, chunk offsets/overlap, genuine
  FAISS retrieval, BM25, RRF, configuration, reranking doubles, embedding instruction wiring,
  citations, Ollama HTTP contract, SQLite restart/deletion, failed ingestion, corrupted-cache
  recovery, FastAPI end-to-end, Streamlit chat/comparison and optional Ragas wiring.
- Ruff lint and formatting checks passed.
- Live Uvicorn: health, PDF upload, all four retrieval modes, cited extractive chat and deletion
  passed. Streamlit health passed; the dashboard interaction test exercises its actual UI code.
- The Docker image built with locked CPU model dependencies; a non-root container (UID 1000)
  passed upload, four-mode retrieval and citation checks. Compose configuration validated; container restart restored metadata and retrieval, and deletion passed.
  Cloud builds needed proxy hostname resolution and an optional BuildKit CA secret. No TLS
  verification bypass was used. The optional CA is mounted only during dependency commands.
- The synthetic two-question English demo ran across four pipelines (8 case/pipeline pairs).
  The standalone benchmark CLI reproduced its retrieval scores. These scores do not measure
  scientific, Arabic or cross-lingual performance; hash embeddings and lexical reranking were
  selected explicitly. Latency means are one-run observations, not performance guarantees.

## Unverified model-dependent capabilities

Real Hugging Face model downloads returned proxy HTTP 403. Network additions were saved
for environment review but a draft save does not change current connectivity. Model adapters
are tested with doubles, not measured model weights. Real semantic/cross-lingual retrieval
quality, real neural reranking quality, Qwen generation and Ragas judge quality remain unrun.
The selected optional Ragas/LangChain versions import and the adapter wiring passes; the
actual local judge needs Ollama and cached embeddings. No paid API is required or invoked.

Windows instructions and CI matrix are prepared but were not executed on Windows or GitHub.
Docker learned-model downloads/generation remain unrun. Real screenshots are pending.
The sibling research-document repository was left unchanged; none of its documents was
copied into this project. The user subsequently authorized GitHub delivery on the dedicated
`multilingual-advanced-rag` branch of `the1ZAKK/project`, preserving `main`. The application
uses its own commit history and project files at the branch root. No public deployment,
registry publication or paid cloud resource was performed.

## Engineering tradeoffs and limits

Exact FAISS flat search gives a transparent baseline. SQLite-backed vectors support recovery,
but all chunks/vectors live in process memory and mutation rebuilds are O(corpus size).
A process-local lock serializes model calls and corpus mutation; run one API worker.
There is no multi-process index coordination, OCR, auth, distributed retrieval, schema migration,
streaming generation, or conversational follow-up rewriting. Scanned pages are rejected or
warned, never fabricated. Arabic column layouts and broken font mappings need manual review.
Character chunking is page-bound and may split a sentence; token-aware section chunking is
future work. Model-specific context/token limits must be tuned for larger deployments.

Exact quotes and source labels validate provenance, not semantic entailment. Instruction
separation mitigates prompt injection without guaranteeing immunity. Evidence thresholds
are heuristics, and lexical answer metrics are not faithfulness judgments. The source corpus,
LLM judge and final claims require human review for research/legal decisions.

## CV / LinkedIn statements supported by this implementation

- Built a modular Arabic/English PDF research pipeline with FAISS, BM25 and Reciprocal Rank
  Fusion, preserving document hashes, page metadata and verifiable source excerpts.
- Engineered transactional SQLite persistence, deterministic index reconstruction, duplicate
  prevention, document deletion and checksum-checked FAISS cache recovery.
- Developed a FastAPI backend and bidirectional Streamlit dashboard for ingestion, cited chat,
  retrieval comparison and evaluation exports.
- Implemented Recall@K, Precision@K, MRR and graded NDCG evaluation with annotation validation
  and reproducible synthetic pipeline benchmarks.
- Validated the offline system with 40 automated tests, live API probes and a non-root Docker
  container; prepared locked CPU dependencies and Windows/Linux CI definitions.

Do not claim measured multilingual accuracy, reranker improvements, Qwen answer quality,
production-scale throughput, a scientific benchmark, or successful GitHub CI until those
experiments actually run. Adapters can be described as implemented integrations; distinguish
that from validated live model results.

## Arabic PRM research extensions

1. Annotate Arabic evidence-grounded reasoning steps with support/contradiction/irrelevance
   labels; train a PRM to score reasoning steps against retrieved passages rather than answer
   style. Avoid storing hidden model reasoning without an explicit research protocol.
2. Compare cross-encoder reranking against PRM-guided evidence selection, holding candidate
   sets fixed. Measure retrieval relevance, attribution precision, final answer correctness,
   latency and calibration on held-out Arabic/English/cross-lingual cases.
3. Construct adversarial Arabic document instructions and dialect/code-switch queries; measure
   refusal, citation entailment and injection resilience with human-reviewed examples.
4. Study normalization/diacritic/morphology ablations and token/section chunking on Arabic
   legal or scientific corpora, with inter-annotator agreement and document-level split controls.
5. Calibrate abstention using PRM/evidence support scores and evaluate selective accuracy
   versus coverage. Compare local Ragas judgments to Arabic expert labels and audit judge bias.
