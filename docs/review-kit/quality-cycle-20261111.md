# Continued quality cycle — 2026-11-11

Starting source: `af2a8e24fd0d5679625c5a3858fcca806f00ab26` on
`arabic-quality-candidate`. The workspace and remote agree and contain no
unrelated edits. The resumed local baseline is 431 passed, 5 skipped.
Quality run 41188100292 passed the four Windows/Linux Python jobs and Docker.

## What the completed recording actually established

Recording 41188100282 used that exact source and the verified Unsloth community
Qwen3-4B-Instruct-2507 Q4_K_M profile. All eight development controls ran through
the real application. Seven returned explicitly labeled excerpt fallbacks.
The remaining generated answer correctly reported the two requested counts
initially, but its follow-up comparison substituted the number of models for
individual evaluations. Neither recording completion nor citation provenance
therefore establishes answer-quality acceptance.

All four Arabic requests still failed the English pivot's output-language
check, despite the explicit system instruction. The small semantic judge also
rejected elementary source statements with internally inconsistent reasons,
such as saying that a stated count did not explicitly establish that count.
These failures are retained, not treated as human verdicts or silently accepted.

The old ungraded packet is preserved on `ci-evaluation-41188100282` at
`docs/review-kit/RAG-PDF-Review-Kit-41188100282.zip`. Its source remains af2a8e2.
It is not the new comparison packet and is not evidence of passed human review.

## New repairs and comparison

- A deictic instruction such as “distinguish the two quantities” is now bound
  to the original requested count roles. Another true source count cannot
  silently replace one of those quantities.
- Add an explicit `qwen25` comparison profile using Bartowski's community
  Qwen2.5-7B-Instruct Q4_K_M GGUF. The application and installer defaults are
  unchanged. The model is downloaded only for an explicitly selected profile.
  Hosted model/workflow changes record this profile; ordinary application
  source pushes continue to run software CI without a multi-gigabyte model run.
- Keep every existing quote, native-page, count/entity, comparison, output
  language, translation, reverse-translation and semantic verdict check.
  No paper answers, reference guides or expected points are generation input.
- Retain the same eight previously exposed development controls, retrieval
  settings, 6,000 context characters, four statements, temperature zero and
  600-second per-answer deadline. The job-level allowance is 120 minutes for
  the larger CPU model; per-answer validation and deadlines are not relaxed.

The comparison model is pinned to repository
`bartowski/Qwen2.5-7B-Instruct-GGUF`, revision
`8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, file
`Qwen2.5-7B-Instruct-Q4_K_M.gguf`, 4,683,074,240 bytes, SHA-256
`65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`.
These pins were checked against the publisher's file metadata before the run;
the installer must independently verify the downloaded bytes before import.

The new recording must supply its exact commit, installed model/runtime,
settings, outputs, failed attempts and citation checks. Its review CSV remains
ungraded for source inspection. These are development controls, not unseen
evaluation; both original PDFs are English-language sources. Independent human
review and actual laptop acceptance remain outstanding. No stable release is
published or approved.

## Preserved earlier recordings

The separate [fresh source-selection check](../validation-results/quality-cycle-source-selection-20261111.json)
used parser v5 in a new isolated index and the original byte-verified PDFs.
All eight English/Arabic controls retained only the matching page-1 Abstract,
with every catalog quote present verbatim on its native page. The selected
ArabicMMLU abstract is 1,297 characters; ArabicNumBench is 1,623. The latter
retains the 71-model/10-provider/59,010-evaluation facts and CoT comparison,
excluding the later 2.7x aggregate. Its 281 model-strategy combinations remain
distinct from model counts in the numerical checks.
This source-selection check explicitly used hash embeddings and no LLM. It is
separate from the hosted learned-retrieval/model recording and cannot establish
answer-quality acceptance.

All rows below ran the same eight exposed controls with the pinned 4B Instruct
profile, 6,000 context characters, four statements, temperature zero, 8,192
context tokens, a 300-second request timeout and a 600-second answer deadline.
Experimental question-coverage auditing was disabled. Their full outputs,
runtime identity, settings, rejected proposals and translation attempts remain
in their own reports. Results are never applied to another commit.

| Workflow run | Source | Recorded modes and observed limitation | Exact report |
| --- | --- | --- | --- |
| [41185838438](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/41185838438) | 2a085bf | Seven excerpt fallbacks; one generated English numerical comparison. Publication failed with insufficient workflow-file permissions. | [Preserved original report](../validation-results/quality-cycle-first-41185838438.json) |
| [41186116651](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/41186116651) | 60c4f4c | Seven excerpt fallbacks; one generated quantity distinction substituted models for evaluations. | [Exact report](https://github.com/the1ZAKK/multilingual-advanced-rag/blob/ci-evaluation-41186116651/docs/validation-results/completion-model.json) |
| [41186460507](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/41186460507) | 0fcaa2b | Same modes; semantic-judge refusals and Arabic pivot failures remain. | [Exact report](https://github.com/the1ZAKK/multilingual-advanced-rag/blob/ci-evaluation-41186460507/docs/validation-results/completion-model.json) |
| [41186741681](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/41186741681) | 893675f | Same modes; English numerical comparison also exceeded its statement budget. | [Exact report](https://github.com/the1ZAKK/multilingual-advanced-rag/blob/ci-evaluation-41186741681/docs/validation-results/completion-model.json) |
| [41187192503](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/41187192503) | c3e4d97 | Same modes; strict unit correction did not resolve the model's refusals or Arabic pivot failures. | [Exact report](https://github.com/the1ZAKK/multilingual-advanced-rag/blob/ci-evaluation-41187192503/docs/validation-results/completion-model.json) |
| [41188100282](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/41188100282) | af2a8e2 | Seven excerpt fallbacks; the one generated quantity distinction is incomplete, as described above. | [Exact report](https://github.com/the1ZAKK/multilingual-advanced-rag/blob/ci-evaluation-41188100282/docs/validation-results/completion-model.json) |

These mode counts describe execution, not semantic grades. All four Arabic
requests fell back to English excerpts in these recordings. No review CSV was
graded. The first report is copied byte-for-byte from the preserved artifact
so its failed publication does not make its exact answers depend solely on
Actions artifact retention.

The [lossless wire traces and integrity manifest](../validation-results/quality-cycle-traces/README.md)
also retain partial raw responses and failed proposals beyond Actions artifact
retention. The recorded source/model/settings remain attached to each run.

## Completed 7B recording and observed repairs

[Recording 43014685739](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/43014685739)
used source `c5719b436b2b35f4c36f868a16a55ee6ee3b4475` and the pinned 7B profile
above. The downloaded GGUF and imported model were verified. Its installed
digest is `fa105931cc3d0c4ee41724c29197600ab3868248b8b4ddfd2961d958ac7ed6e5`;
Ollama is 0.40.5. All software CI jobs and hosted Windows acceptance passed.
The [exact report](https://github.com/the1ZAKK/multilingual-advanced-rag/blob/ci-evaluation-43014685739/docs/validation-results/completion-model.json)
preserves all eight returned answers, 12 complete generation proposals, 13
validation refusals, two forward translation attempts and exact settings.
The lossless wire trace additionally preserves partial raw responses.

**Answer-quality acceptance failed: all eight responses were excerpt fallbacks.**
Every returned citation matched its original PDF page, but all four Arabic
requests retained English source excerpts. Provenance is not semantic acceptance.

| Control | Observed final refusal |
| --- | --- |
| mmlu-size-en | 600-second answer deadline / ReadTimeout |
| mmlu-size-ar | 600-second answer deadline / ReadTimeout; an intermediate proposal also omitted the evaluated-model count |
| numbers-prompt-en | Irrelevant numerical citation after an over-budget first proposal; the repair substituted model/task counts for the requested accuracy comparison |
| numbers-prompt-ar | 600-second answer deadline / ReadTimeout |
| mmlu-format-en | Semantic judge contradicted its own recognition of the stated question count and MSA format |
| mmlu-format-ar | Forward translation changed a source-bound dataset identifier |
| numbers-task-evaluations-en | Follow-up substituted models for evaluations, then omitted the requested evaluation count |
| numbers-task-evaluations-ar | Required evaluation count omitted from a compound question slot |

The resulting [ungraded packet](https://raw.githubusercontent.com/the1ZAKK/multilingual-advanced-rag/ci-evaluation-43014685739/docs/review-kit/RAG-PDF-Review-Kit-43014685739.zip)
is bound to c5719b4, not the later repairs. Its SHA-256 is
`35de034108b0001ea6e5256557c3b29dbd6890a8e74ed5b64a57f921a26a9eec`.
All seven manifest file hashes, both original PDF hashes, eight question/answer
bindings, eight PDF links and every citation/page were independently rechecked
by the AI development agent. The UTF-8 CSV has eight not_reviewed rows and no
reviewer, reviewer_type or notes. It remains available as a failed attempt.

Two concrete software repairs follow these observations. Arabic coordinated
“وكم” clauses now create independent required count slots, matching English
“and how many”. Source-derived identifiers in ordinary forward and reverse
translation are now protected with opaque tokens and restored exactly. Missing,
duplicated or invented tokens fail; reverse translation still sees only its
Arabic input, and all number, language, round-trip content and semantic checks
remain required. No source-specific names or answers are hardcoded.

New regressions cover clause separation, identifier restoration, independent
reverse input, dropped/duplicated/invented tokens and number/meaning changes
despite a positive model verdict. The model report also records the exact import
parameters and execution host for the next CPU comparison. Application defaults,
the verified 7B weights, generation limits and validation requirements stay the
same. A fresh recording is required for these repairs; this earlier packet
cannot validate their new outputs.

## Final recording after the observed repairs

Recording [43025032186](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/43025032186) uses application
source `3f7544c80247f90606026b866014f5e58202db86`. All eight controls returned explicit excerpt
fallbacks; all four Arabic final outputs retain English source text. Identifier
spelling is preserved and Arabic count clauses now have separate slots, but
complete generated answers and faithful Arabic acceptance remain unmet.

Local validation is 444 passed, 5 skipped; all Windows/Linux Python jobs,
Docker and hosted Windows acceptance passed. The exact new ZIP and full trace
passed [hosted packet verification](https://github.com/the1ZAKK/multilingual-advanced-rag/blob/ci-review-verification-43031692051/docs/validation-results/review-packet-verification.json) after
the local workspace became unavailable. Twenty-eight native citation entries
match page 1; the eight-row UTF-8 CSV is ungraded. No human review is inferred.

See the [concise final report](../quality-cycle-20261111.md) and
[machine-readable report](../validation-results/quality-cycle-20261111.json)
for per-case defects, exact source/model/settings, immutable download and proof
links, and the distinction between software checks and answer acceptance.
