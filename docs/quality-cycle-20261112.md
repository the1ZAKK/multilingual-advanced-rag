# Recovered Arabic answer-quality cycle — 12 November 2026

The [fallback-repair continuation](quality-cycle-20261112-fallback-repairs.md)
records the subsequently recovered hosted final status/log, the evidence that its
exact model recording is unavailable, strict repairs and new source-labelled runs.
This document retains the earlier checkpoint and its original limitations.

The interrupted work was recovered rather than restarted. The semantic audit
format change was already saved at `c0b28b8`, and the active eight-question
recording at `d2e5b89` was reused. Recovery commit `01b868a` adds five HTTP-path
regressions and continuation documentation without changing production code.

## Software validation

| Check | Result |
| --- | --- |
| Recovered baseline | 504 passed, 5 skipped |
| Full suite after recovery regressions | 509 passed, 5 skipped |
| Ruff lint and formatting | Passed |
| Windows/Linux, Python 3.11/3.12, at `01b868a` | All four jobs passed |
| Docker at `01b868a` | Passed |
| Hosted Windows application acceptance at `d2e5b89` | Passed |

The five new tests send the real semantic-audit HTTP request through a mock
transport. They verify that the short evidence explanation precedes the nested
verdict, all boolean decisions remain required, a positive explanation cannot
override a negative decision, original question/evidence bindings remain intact,
and the exact raw response is preserved. Existing nonboolean, omitted-field,
numerical, citation and translation regressions remain unchanged.

The model recordings below use the real application with learned multilingual
MiniLM embeddings, hybrid retrieval and no reranker. Settings are 6000 evidence
characters, 8192 context tokens, at most four statements, temperature 0,
300-second request and 600-second answer deadlines, and experimental extra
question-coverage auditing disabled. The original application model default is
unchanged. Each report retains its exact model revision, GGUF hash, installed
digest, runtime, settings, outputs, failures and citation checks.

## Recovered recordings

| Recording | Source | Verified model | Recorded outcome |
| --- | --- | --- | --- |
| Local `1ba5474a-9211-4a68-866a-044bd0481640` | `c0b28b8` | Qwen3-4B-Instruct-2507 Q4_K_M | Four generated outputs and four explicit excerpt fallbacks |
| Hosted `43083457888` | `c0b28b8` | Qwen2.5-7B-Instruct Q4_K_M | One generated output and seven explicit excerpt fallbacks |
| Hosted `43086599560` | `d2e5b89` | Qwen2.5-7B-Instruct Q4_K_M | Reused; artifact upload failed and no completed model report is retrievable |

The local recording's source is the commit in `answers.json`, `c0b28b8`.
The later checkout head in its working checkpoint note is not substituted for
that identity. No local model/API/dashboard process survived recovery; the
hosted recording was already running. No duplicate model workflow was started.

The local 4B outputs include complete English and Arabic ArabicMMLU size/score
responses, an English ArabicNumBench count/comparison response, and an English
language/format response. The Arabic ArabicNumBench comparison, Arabic
language/format response, and both task-versus-evaluation distinction responses
remain explicit fallbacks. These observations are not independent human grades.

The trace preserves concrete unresolved failures: the reverse translator changed
`providers` to `suppliers`, and changed `The questions are in` to `Questions
exist in`. The conservative content check rejected those lexical changes.
Other attempts produced malformed JSON or failed numerical grounding. The
earlier 7B judge also rejected source-explicit facts for reasons contradicted by
its supplied premises. No validator was bypassed to turn these cases into passes.

## Review packets and provenance

The recovered local 4B outputs were exported into a new ungraded packet. Its
SHA-256 is
`2397ae051e07ff4a9f4487a38f27a61860af88070b6db21e86df10f55c0773b1`.
The existing verification workflow's Python checks were executed locally:
both PDF hashes, all eight frozen questions and exact recorded responses,
runtime/settings bindings, seven manifest file hashes, eight HTML PDF links,
native quotations/pages, UTF-8 BOM and untouched reviewer fields passed.
All 230 local wire events remain preserved losslessly.

The earlier hosted packet also passed those checks, including equality with its
already-published Git blob and all 29 native citations. Its packet SHA-256 is
`ebef2b88ecd413a78422b0aa972285da3cd0ba2ffe25b6db3d11ad48e765e72c`.
GitHub's downloaded artifact hash was verified before extraction.

The [new recovered 4B packet](review-kit/RAG-PDF-Review-Kit-recovered-local4b-c0b28b8.zip)
contains the exact saved local outputs at `c0b28b8`, not purported outputs at
`d2e5b89`. Its [verification record](validation-results/recovered-local4b-packet-20261112.json)
retains all source, answer, model, settings, quotation and CSV checks. Four
generated modes do not establish four source-review passes; the four fallback
outputs remain failed complete-answer attempts. The CSV remains entirely
ungraded. Only this requested packet, which contains the original public PDFs
and review materials, is published; it contains no raw logs or wire traces.

Successful generation, exact quotations and packet integrity do not establish
answer completeness, faithful Arabic or relevant citation entailment.

## Hosted recording dependency failure

At 04:37 China time on 12 November, the workflow changed from `in_progress` to
`queued`, still on attempt 1. The individual model-job step endpoint reports a
failed artifact upload and successful runtime cleanup and job completion, while
the evaluation step itself still says `in_progress` and export steps say
`pending`. This inconsistent state cannot establish either model completion or
its exact answer outcomes. The job-list and job-log endpoints repeatedly
returned `ReadTimeout`; only the Windows artifact is available. No result
branch or new hosted packet is available at the time of this report.

These are observed GitHub recording/publication failures, not answer-quality
verdicts. Neither partial model answers nor successful generation counts are
invented. The existing recording remains preserved under its original identity;
no duplicate or replacement model evaluation was started. The remaining step is
to recover that recording/artifact when GitHub exposes it, or establish its final
failure before considering a new isolated attempt. Do not grade `d2e5b89` from
the earlier local or hosted `c0b28b8` outputs.

## Auto-review and preservation

Auto-review rejected pushing local commit `1b00f7d` with the diagnostic archive
because it would publish model outputs, traces, logs and source-related data to
a public repository. The first rejection also described the remote as
unverified. Repository identity was checked against the connected GitHub account;
credential-signature scanning did not override the disclosure rejection.

The original archive and its byte hashes remain intact on the local
`recovery-local-evidence-20261112` branch and as the original two untracked
files. A separate tests/documentation-only commit was published to the exact
authorized repository and candidate branch. No archive or raw diagnostic data
was included in that recovery update. The ordinary Git CLI lacked credentials;
the permitted update used the connected GitHub tools and a checked branch-head
lease. No other repository, release or branch of user work was modified.

The original 3 partial / 1 fail AI-assisted judgments apply only to recording
39219463133 at `d66c62d`, before the abstract-routing correction. They do not
grade these later recordings. All eight questions are exposed development
controls, and both PDFs contain English source prose. Independent human source
review, native Arabic source-prose evaluation and actual laptop acceptance
remain pending. No stable release has been published.
