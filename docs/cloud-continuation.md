# Continuing development in Codex and ChatGPT

The source repository is https://github.com/the1ZAKK/multilingual-advanced-rag.
Codex can edit, run, and test this checkout in the managed cloud environment.
ChatGPT can explain results and help plan the work. GitHub stores commits and
runs CI; a GitHub username identifies an account, not an independent reviewer.

The current acceptance target remains a local application. A managed Codex task
is a development machine, not a permanent public application host. Restarting a
task requires starting its API, dashboard, and local model service again using
the saved environment startup instructions. Persistent public hosting would
require a separate deployment and operating plan.

## Current candidate

`arabic-quality-candidate` has explicitly selected, checksum-pinned comparison
profiles for Qwen3-4B-Instruct-2507 (Unsloth community GGUF) and
Qwen2.5-7B-Instruct (Bartowski community GGUF). The original official base-model
profile remains available. These profiles leave the application's model default
unchanged. Every model report records repository, revision, file hash, size,
installed model, exact source commit, generation settings and runtime version.

The [2026-10-31 cycle](review-kit/quality-cycle-20261031.md) adds general count,
numerical-relation, citation and translation guards plus portable review export.
The [2026-11-11 continuation](review-kit/quality-cycle-20261111.md) preserves the
failed 4B recordings and tests an explicit 7B comparison. Eight exposed bilingual
development controls include the four original questions without changes. A
finished recording or successful CI does not establish answer-quality acceptance.
The CSV in every newly exported packet remains ungraded for source inspection.

The hosted recording workflow accepts a model profile and an explicit choice
of experimental question-coverage auditing. The new Instruct recording uses
coverage auditing disabled, matching the application's current default. Earlier
base-model runs used that audit enabled, so differences are configuration
comparisons, not a controlled claim that the model alone improved accuracy.

`data/evaluation/additional-paper-review.json` freezes four bilingual numerical
questions about ArabicMMLU and ArabicNumBench before candidate predictions.
Both source PDFs are written in English about Arabic benchmarks. Their identity
hashes and page-1 reference passages were checked against the uploaded PDFs.
These checks do not establish performance on native Arabic source prose.
The initial managed-cloud attempt to fetch their official arXiv copies was
blocked with HTTP 403; local matching originals remain usable. Hosted source
preflight must verify the public PDF hashes before any hosted model evaluation.

## Release gates

- Run software checks and retain the exact tested commit and configuration.
- Compare answers with source passages for correctness, completeness, numbers,
  language, and entailment. Exact quotations alone do not establish those properties.
- Give the ungraded packet and original PDFs to an actual independent person.
  Record that person's reviewer identity and completed judgments. AI review
  remains separate; no human reviewer has yet been appointed.
- Hosted Windows acceptance demonstrates compatibility on that runner. Actual
  laptop startup, upload, indexing, chat, and restart remain unverified.
- Publish a stable release only after the required review and target checks pass.
  A candidate branch, successful CI, or an answer in `ollama` mode is insufficient.

Do not overwrite user indexes, original PDFs, private settings, or earlier
review artifacts when continuing. Use a new isolated evaluation output directory
for every model attempt. Keep failed attempts for comparison.

## 2026-10-19 candidate checkpoint

- Candidate commit `1ad5f63`: 407 software tests passed, 4 skipped; Ruff clean.
- The additional-paper 1.7B baseline completed four attempts. The English
  ArabicMMLU response omitted the requested 35-model count; both Arabic cases
  incorrectly entered table routing and fell back to excerpts. The English
  ArabicNumBench response was rejected by a semantic judge that demanded
  methodological proof of a result explicitly reported in its premise.
  These are observed defects, not accepted answers.
- The routing correction keeps explicit abstract requests on prose evidence
  in both languages, while retaining explicit numbered-table requests. A
  learned-retrieval check verified that the ArabicMMLU Arabic request now
  retains the Abstract and the requested source numbers. That check does not
  establish generation completeness; the two-case real-model rerun is separate.
- [Six-case Instruct recording](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/39219159043)
  uses source commit `6bc903d`, before the routing correction.
- [Four additional-paper Instruct attempts](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/39219463133)
  use source commit `d66c62d`, also before that correction. Both hosted Windows
  jobs passed, but their model jobs were still running at this checkpoint.
- Local additional-paper baseline: `reports/additional-qwen17-001`;
  ungraded human packet: `reports/additional-qwen17-human-packet`;
  routing-only Arabic rerun: `reports/additional-qwen17-routing-fixed-001`.
  These ignored reports belong to this workspace and are not GitHub artifacts.
- Independent human reviews remain zero. No stable release was published.

## 11 November 2026 final checkpoint

Application source `3f7544c80247f90606026b866014f5e58202db86` passes 444 tests (5 skipped), Ruff,
all Windows/Linux Python jobs, Docker and hosted Windows learned acceptance.
Recording 43025032186 nevertheless returns eight explicit excerpt fallbacks;
complete answers and faithful Arabic acceptance remain open. The local workspace
became unavailable after source validation; hosted verification completed the
native-PDF, exact-answer/CSV and ZIP checks and preserved the full wire trace.

The [final report](quality-cycle-20261111.md) links the exact ungraded packet,
model/runtime settings, failed attempts and integrity proof. These eight exposed
controls are not unseen evaluation. The original d66c62d review is still wholly
AI-assisted and cannot grade later outputs. No independent human review or stable
release has passed. Application model defaults and user data remain unchanged.
