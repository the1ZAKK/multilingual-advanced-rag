> Superseded by the [18 October continuation report](validation-20261018.md). The results below describe the previous release.

# Current validation and delivery — October 2026

This supersedes historical setup counts. Work continued from uploaded
`project-source.zip`, imported unchanged as Git baseline `6581c08`. The archive
contains source, not the user's active Windows `.env`, database, model cache or
backups. Those laptop files were not accessed or modified here. No push,
deployment or paid service was used in this continuation.

## Delivered changes

- Document selection constrains dense and lexical retrieval before ranking and
  applies to chat/comparison. Unknown document IDs are rejected.
- Tall two-column PDF pages and Roman-numbered author lists are supported.
  Source-bound exact quote validation remains strict.
- Table caption binding and explicit table/model/condition selection avoid
  adjacent-table confusion. Row/column limits are labeled as non-exhaustive.
- Ordinary answers have statement/output limits and a shared generation, repair,
  translation and audit deadline. Sentence markers preserve source bindings.
  Semantic-judge requirement flags are no longer mistaken for verdicts.
- Evaluation exports include run/corpus/configuration metadata, p50/p95 timings,
  optional answers, separate language/provenance denominators and strict scope.
- Portable synthetic PDFs, real browser screenshots, Windows resume helper,
  Docker build helper and a hash-checked backup-first updater are included.

## Executed checks

| Check | Observed result |
|---|---|
| Python 3.12.14, optional extras installed | 335 passed, 4 skipped |
| Python 3.11.16, base/development extras | 334 passed, 5 skipped, 1 upstream deprecation warning |
| Ruff lint / formatting | Passed; 85 files already formatted |
| Real Chromium dashboard | Documents, extractive chat, comparison and evaluation exercised |
| Learned embeddings | Actual multilingual MiniLM loaded; normalized 384-dimensional English/Arabic vectors |
| Learned reranker | Actual multilingual cross-encoder loaded; small English relevance smoke passed |
| Real Qwen3 | English generation, conservative refusal and source-table paths exercised |

Four suite skips require laptop PDF/reference artifacts absent from this upload.
The extra base-environment skip is the optional Ragas dependency. Windows launcher
was reviewed but **not executed on Windows**. GitHub CI has not run for this update.

## Live findings, including failures

Official Qwen GGUF files were imported into a private local Ollama container with
publisher SHA-256 verification. Qwen3-1.7B Q8 and Qwen2.5-1.5B Q4 were tried; these
are not identical quantizations to the user's registry tag. Model weights and
academic PDFs are excluded from this source package.

Four papers were ingested into a new isolated cloud corpus: ArabicMMLU, AlGhafa,
AceGPT and ArabicNumBench (527 chunks). This did not migrate the laptop corpus.
Final Qwen3 development cases:

| Case | Outcome |
|---|---|
| ArabicMMLU English overview/counts | `ollama`; generated cited answer |
| Same question in Arabic | `extractive`; wrong-language output rejected |
| AlGhafa English overview | `ollama`; generated cited answer |
| Unsupported AlGhafa process-reward-model premise | `insufficient-evidence`; no invented positive claim |
| AceGPT Table 5 zero-shot comparison | `source-tables`; correct caption/selected model values |
| AceGPT Table 8 named base-model row | `source-tables`; correct table/six values |

The English ArabicMMLU answer reported 40 tasks and 14,575 questions. AlGhafa's
answer separated its evaluation benchmark and HandMade corpus. These spot checks
do not establish general accuracy. [Complete answers and warnings](validation-results/live-development-cases.json).
Historical failed development runs remain in cloud `reports/research-validation/`.

A four-query retrieval probe (English/Arabic; sparse primary-passage annotations)
used the actual embedder/cross-encoder across all four documents without scope:

| Pipeline | MRR | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| Dense | 0.1125 | 0.2500 | 0.2500 |
| BM25 | 0.3750 | 0.3750 | 0.5000 |
| Hybrid | 0.3170 | 0.1250 | 0.3750 |
| Hybrid + cross-encoder | 0.6528 | 0.6250 | 0.7500 |

Annotations are incomplete and cases exposed during development. Loading affects
latency. This is diagnostic evidence, not a pristine benchmark or general pipeline
ranking. Chat's overview context differs from standard retrieval evaluation.
[Machine-readable summary](validation-results/retrieval-development-summary.json).

## Remaining limitations

Ordinary Arabic prose is **not reliably solved**: the final Arabic overview
rejected an English response and returned source excerpts. Earlier Qwen2.5 runs
also failed quote/semantic checks. Fail-closed warnings must not be called
successful generated answers. Guarded authored-list translation does not establish
general Arabic generation quality.

Exact source matching checks provenance, not the truth or completeness of every
claim. The semantic judge can err. Human review is needed for important claims,
negation, table relationships and translations. PDF layouts vary and broad
numerical responses are intentionally limited. Independent unseen-paper and human
Arabic evaluation remain open. Historical 3LM scores are not cross-paper evidence.

Follow [upgrade instructions](upgrading.md). No automatic reindex occurs. Optional
migration backs up first and changes passage IDs; regenerate affected annotations.

## Release checks

The final source built successfully as a non-root Docker image. The previous
build attempt exhausted the cloud disk; redundant temporary model-import files
and old build cache were removed, and dependency ownership copying was reduced.
The final container functional probe exercised health, PDF upload, retrieval,
cited extractive chat and cleanup. The restarted source API passed the same probe,
and the running dashboard returned its health response.

A live optional Ragas smoke ran after fixing explicit metric initialization:
one synthetic English example returned faithfulness **1.0** and answer correctness
**0.75** using real Qwen3 and learned embeddings. This is a single judge-dependent
smoke result, not a project accuracy score. [Recorded result](validation-results/ragas-smoke.json).

The safe updater was exercised against a fresh extraction of the exact uploaded
ZIP: 70 source files changed, a second application changed zero files, three
private sentinels (`.env`, index and model cache) remained identical, the backup
ZIP passed integrity checks, and a later private README edit was refused before
any source writes. The portable archive has its own SHA-256 source inventory.

Cloud installation and restart instructions were saved as a configuration draft
for `/workspace/rag-current`. Saving does not publish or prove fresh-task
restoration. Review/save in environment settings and publish there if you want a
reusable snapshot. The active local-only source changes were not pushed to GitHub.
