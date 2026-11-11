# Quality cycle — 11 November 2026

Software checks pass, but complete-answer acceptance remains unmet. The latest
verified 7B recording returned **eight explicitly labeled excerpt fallbacks**.
All four Arabic requests still contain English source text. No independent
human review or stable release is claimed.

## Changes

- Reuse the abstract-routing correction; recognize inline abstracts and section
  boundaries without rewriting native quotation text.
- Require separate count/score facets and reject omitted quantities, swapped
  count/entity or prompting-condition/accuracy relationships, missing ratios,
  irrelevant numerical citations and untranslated prose.
- Preserve decimal multipliers and written numerical units through Arabic and
  reverse translation. Split Arabic “وكم” count clauses into separate parts.
- Protect source-derived identifiers with opaque translation tokens. Keep all
  existing number, language, reverse-content and semantic checks.
- Preserve failed proposals and export exact, ungraded, source-bound packets.
  Hosted verification remains available when the local workspace is unavailable.

## Recorded evidence

Application/generation source: `3f7544c80247f90606026b866014f5e58202db86`.
Later commits add packet verification and documentation only.

[Model run 43025032186](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/43025032186)
used `qwen25-instruct-hf:7b`, Bartowski's community Qwen2.5-7B-Instruct
Q4_K_M GGUF, verified SHA-256
`65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`.
Ollama 0.40.5 ran on a four-CPU Linux host. Import parameters were
8,192 context tokens and four threads. Evaluation used 6,000 source characters,
8,192 generation/audit context tokens, four statements, temperature zero,
300-second request and 600-second answer deadlines. Question-coverage auditing
stayed explicitly disabled. All other validators remained required.

[Exact settings, answers and refusals](https://github.com/the1ZAKK/multilingual-advanced-rag/blob/ci-evaluation-43025032186/docs/validation-results/completion-model.json) and the
[machine-readable cycle report](validation-results/quality-cycle-20261111.json)
record the full installed digest, revision, retrieval configuration and per-case
timings. Expected answer guides were never generation input.

- **444 passed, 5 skipped**, with Ruff lint, format and diff checks clean locally.
- [Windows/Linux Python 3.11/3.12 and Docker](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/43025032233)
  passed; hosted Windows learned-retrieval acceptance also passed.
- Eight exposed development controls ran: the four original questions unchanged,
  plus four bilingual controls. These are not unseen evaluation, and both source
  PDFs are written in English.
- Thirteen completed proposals, fifteen validation refusals and forward/reverse
  translation attempts are preserved. The full wire recording also retains all
  fifty raw generation responses, including partial proposals.

## Remaining defects

| Control | Latest observed failure |
| --- | --- |
| mmlu-size-en | Judge refuses a directly reported task count; repair omits the question count. |
| mmlu-size-ar | Evaluated-model slot repeats the task count; semantic refusal remains. |
| numbers-prompt-en | Repair substitutes model/task counts for the accuracy comparison and includes an irrelevant citation. |
| numbers-prompt-ar | Statement budget exceeded; no verified Arabic answer. |
| mmlu-format-en | Inconsistent semantic rejection of source counts/format. |
| mmlu-format-ar | Identifier spelling is preserved, but meaning/wording still fails reverse-content validation. |
| numbers-task-evaluations-en | Follow-up substitutes models for evaluations, then omits evaluations. |
| numbers-task-evaluations-ar | Clause separation works; the judge still refuses the directly reported task count. |

The guards prevent these proposals from being presented as complete generated
answers. They do not make generation or Arabic translation reliable. The strict
reverse-content guard can also refuse faithful paraphrases. Native citation matches and successful CI do not establish answer quality.

## Review packet

[Download the exact new ZIP](https://raw.githubusercontent.com/the1ZAKK/multilingual-advanced-rag/ci-evaluation-43025032186/docs/review-kit/RAG-PDF-Review-Kit-43025032186.zip).

The [hosted verification](https://github.com/the1ZAKK/multilingual-advanced-rag/blob/ci-review-verification-43031692051/docs/validation-results/review-packet-verification.json)
passed: both original PDF hashes, seven manifest file hashes, all twenty-eight
native citation/page entries, eight PDF links and exact answer/CSV bindings.
The UTF-8 CSV has eight `not_reviewed` rows; reviewer, reviewer_type and notes
are blank. ZIP SHA-256: `2f5cf6d41f9a9f7ff961df9053b5a1083fae817fa42b9b5d11e73896e5b25904`.

The [lossless raw trace](https://github.com/the1ZAKK/multilingual-advanced-rag/blob/ci-review-verification-43031692051/docs/validation-results/generation-wire.jsonl.gz)
and original compact report are stored on the same data-only verification
branch, beyond Actions artifact retention. Earlier failed recordings remain in
the [history](review-kit/quality-cycle-20261111.md) and their own result branches.

The workspace became unavailable after local tests and the source push. Remaining
packet checks and raw-trace preservation completed on GitHub. No uploaded PDFs,
existing user indexes, private settings, backups or submitted reviews were edited.

The original three partial / one fail AI-assisted review still applies only to
recording 39219463133 at d66c62d. This new CSV is ungraded. Independent human
review, native Arabic source-prose evaluation and actual laptop acceptance
remain pending. Application model defaults are unchanged; no stable release
was published.
