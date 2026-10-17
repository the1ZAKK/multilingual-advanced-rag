# Evaluation methodology

For actual additional-paper Arabic/English runs, separate correctness/completeness and citation provenance: see [the October 16 resumption report](resume-evaluation-20261016.md). Its valid-citation outputs still expose substantial answer-quality failures; earlier 3LM success does not establish generalization.

Annotate retrieval relevance at chunk level; IDs contain the PDF SHA-256, page and character
offsets. Upload documents, query `/search`, inspect source passages, then record relevant
IDs. Annotation is invalidated by different PDF bytes, chunk settings or extraction versions.
The API rejects unknown IDs. Keep document hashes, parser version, chunk configuration,
model revisions, language, annotator decisions, and hardware with each experiment.

```json
{
  "cases": [{
    "id": "ar-q01",
    "question": "كيف يُقيّم الاستدلال في اللغة العربية؟",
    "language": "ar",
    "relevant": {"SHA256:p3:0-900": 2, "SHA256:p4:0-900": 1},
    "reference_answer": "Optional human-written grounded answer"
  }],
  "modes": ["dense", "bm25", "hybrid", "hybrid-rerank"],
  "generate_answers": false,
  "use_llm": false
}
```

Grades are positive finite values up to 10 (normally 1–3). Recall counts relevant IDs
regardless of grade. Precision@K uses denominator K, even with fewer retrieved passages.
MRR is reciprocal rank of the first relevant passage within the retrieved top 10. NDCG uses
`2^grade - 1` gain and logarithmic discount, with ideal ordering from all annotated relevant
passages. Duplicate IDs in rankings are removed. Metrics are macro-averaged across cases.
No-hit queries contribute zero. Unannotated chunks are treated as nonrelevant; incomplete
annotations therefore bias measured precision. Query latency includes embeddings, both
retrievers and requested reranking. Response latency is measured separately via a second
chat call, including retrieval and generation; this is not a retrieval-cache benchmark.

The first model load can dominate latency. Warm models, repeat query sets, report p50/p95,
measure cold starts separately and use a held-out test set for defensible comparisons.
The included scripts report one pass and means; confidence intervals and repeated timing
are future extensions. Do not use the two-question synthetic demo as portfolio accuracy
claims. It exercises the pipeline only and is not an Arabic or cross-lingual benchmark.

For a real evaluation, manually annotate Arabic, English, Arabic→English, and English→Arabic
questions. Keep independent train/development/test partitions; tune retrieval depth, chunk
size, overlap and RRF constants only on development data. Compare all four pipelines on the
same corpus, same annotations and same hardware. A lexical reranker comparison must be
labeled lexical; use actual CrossEncoder weights for neural reranking claims.

Answer metrics: `answer_token_f1` compares normalized answer tokens to the reference,
`evidence_token_support` measures answer vocabulary found in retrieved text. These do not
prove correctness, entailment or faithfulness and are especially limited for paraphrases
and Arabic morphology. Optional `scripts/ragas_evaluate.py` uses local Ollama through
LangChain wrappers for Ragas Faithfulness and AnswerCorrectness. Inputs follow
`ragas-example.json`: user_input, response, retrieved_contexts, optional reference. Record
judge/model versions and manually audit Arabic reliability. Never substitute mocked scores
for model judgments. Ragas invocation errors propagate; no fabricated replacement score.

CSV and JSON preserve per-case results and summaries. The dashboard plots relevance metrics
and latency; all exports originate in actual runs. `scripts/demo.py` generates fixture PDFs,
binds stable IDs to those documents and writes `reports/demo/dataset.json` and results.

BM25 uses rank-bm25 with positive smoothed IDF `log(1 + (N-df+0.5)/(df+0.5))`
to keep ubiquitous terms retrievable in very small corpora. Arabic normalization preserves
ta marbuta, removes diacritics/tatweel, and canonicalizes alef/ya variants. English case
folding can be disabled with `RAG_BM25_CASEFOLD=false`; the index rebuilds deterministically.

The committed `data/evaluation/demo-documents/` PDFs are synthetic English fixtures. Upload
both with default chunk settings and evaluate `data/evaluation/demo.json` to reproduce the
demo schema. `docs/demo-results.json` records an actual hash/lexical run on these fixtures.
Running `scripts/demo.py` twice does not add duplicates: generated PDF bytes are deterministic.

## Current audit exports

Use `include_answers=true` to retain predictions and verified citations for independent
review. Case-level document filters are validated before ranking and annotations outside
the selected documents are rejected. Run IDs, UTC timestamps, configuration and document
hashes support reproducibility. p50/p95 are observed query timings for that run, including
cold loads; no throughput guarantee or significance claim follows from a tiny dataset.

`citation_provenance_valid` checks identity and exact original/indexed substrings. It does
not measure entailment. A refusal without citations returns null, not a vacuous true.
Language checks use script heuristics; they do not establish fluent Arabic or translation
quality. Scored denominators accompany aggregated rates. CSV nested fields contain JSON.

The live question runner records source modes, fallback warnings and full answers, without
changing uploaded documents. Reference-based semantic grading and additional held-out
annotation still require independent review. Never turn a passing mode/quote check into a
semantic accuracy score. Source-table and source-list responses are not Qwen-generation
successes. See the current validation report for actual successful and failed runs.
