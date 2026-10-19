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

`arabic-quality-candidate` adds an explicitly selected, checksum-pinned
Qwen3-4B-Instruct-2507 test profile. Its GGUF is a community quantization
published by Unsloth, not an official Qwen GGUF. The original official base-model
profile remains available. Neither profile changes the application's default.
The model report records repository, revision, file hash, size, installed model,
and runtime version.

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
