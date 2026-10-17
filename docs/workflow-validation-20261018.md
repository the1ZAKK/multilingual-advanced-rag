# Workflow validation — 18 October 2026

This continuation implements the [daily workflow](project-workflow.md), runs it on
three additional source papers, and fixes diagnosed extraction and UI defects.
It does not declare the whole project finished. The laptop filesystem/services
were not accessible here; Windows startup and local backups were not performed.
No active user index was migrated. All paper runs used newly created cloud indexes.

## Changes

- Added a source-checked runner that refuses existing run directories and verifies
  PDF SHA-256 identities and exact native-page reference quotes before writing output.
  Every question is scoped to the verified document ID. Gold expectations are not
  passed to the generator. Hash/offline and learned/Qwen profiles are explicit.
- Added CSV review export and summaries bound to exact runs, responses, questions,
  expectations and source identities. Human and AI-assisted judgments have separate
  denominators. Ungraded cases do not receive a vacuous correctness score.
- Added **Export chat for review** to the dashboard, including failed requests and
  blank review fields. Fixed chat failures disappearing after a Streamlit rerender.
- Fixed the standalone question runner silently choosing one of multiple documents
  with the same filename. Ambiguous names now require an explicit document ID.
- Implemented parser `columns-sentences-v4`: native order for single-column pages,
  indented abstract recognition, and whitespace-aware fallback/overlap boundaries.
  Two-column geometry and native RTL order remain supported. Original PDF bytes,
  exact quotation checks and claim-validation requirements remain unchanged.

Parser v4 affects new ingestion. Existing documents keep their current chunks until
an explicit, backed-up migration is requested. Old passage-level annotations need
regeneration after migration. See [upgrading](upgrading.md).

## Before-fix real-model run

Seven cases were frozen before execution in
[`workflow-review.json`](../data/evaluation/workflow-review.json). Original PDF
hashes and native-page references were verified. Papers are ArabLegalEval, AraTable
and AceGPT; original PDFs are not redistributed. This is a development control set,
not a pristine unseen-paper benchmark. Prior exposure to these research topics
prevents claiming an independent scientific evaluation.

The run used actual multilingual MiniLM embeddings, hybrid retrieval, no active
cross-encoder reranker, real local `qwen3-hf:1.7b` Q8, 6000 context characters,
three ordinary statements and the existing total deadline. This model differs
from the laptop registry tag's quantization. No new model was downloaded.

| Case | Mode | AI-assisted source review |
|---|---|---|
| ArabLegalEval problem/source material | `ollama` | Partial: correct gap, but omits Saudi documents and synthesized questions |
| ArabLegalEval retrieval scope/future work | `ollama` | Partial: correct current negative, but misses the specific future retrieval assessment |
| Same retrieval question in Arabic | `extractive` | Fail to deliver the requested Arabic answer; safe source fallback |
| AraTable three task types | `ollama` | Pass: all three tasks match page 2 |
| AraTable curation/evaluation mechanism | `extractive` | Fail: fragmented single-column source evidence led to semantic rejection |
| Unsupported AraTable process reward model | `insufficient-evidence` | Pass: no invented positive claim or paper-wide absence claim |
| AceGPT Table 5 zero-shot F1 comparison | `source-tables` | Pass: both models, both columns and all four values match page 5 |

The initial run had **3 pass, 2 partial and 2 fail** AI-assisted judgments.
Independent human reviews: **zero**. Exact provenance checks passing did not make
incomplete answers correct. AI-assisted labels are not objective ground truth.

## Extraction fix and targeted retest

AraTable's single-column abstract had been passed through layout sorting. That
introduced indentation/margin stamps absent from native PDF text, lost the Abstract
section label and broke continuous citation evidence into small fragments. Overlap
could also start inside an ordinary word. Synthetic regressions reproduce both
conditions without embedding a paper-specific production answer.

A new index with parser v4 preserved continuous native method statements. The same
curation/mechanism question then returned `ollama` with correct LLM-generation and
human-filtering information. It still omitted the asked self-deliberation mechanism:
**partial**, not pass. A short general coverage-prompt experiment produced the same
incomplete answer and was discarded. No citation checks were weakened.

Only the affected method case was rerun with the live model after this fix; do not
combine the earlier seven-case run and the targeted retest into a final-version
accuracy score. The complete seven-case offline wiring run also passed with parser
v4, but source excerpts do not demonstrate generated-answer correctness.

[Compact outputs, settings and explicit reviewer types](validation-results/workflow-20261018.json).
Full private run artifacts and review CSVs remain under locally ignored `reports/`.

## Verification

| Check | Result |
|---|---|
| Python 3.12.14 | 367 passed, 4 skipped |
| Python 3.11.16 base/development extras | 366 passed, 5 skipped; one upstream deprecation warning |
| Ruff lint and formatting | Passed |
| Source API restarted from this checkout | Functional upload/retrieval/cited-chat/cleanup probe passed |
| Real Chromium on the updated dashboard | Chat export downloaded and inspected; actual questions, source citations, request profile and ungraded fields verified |
| Docker | Built and passed the offline functional probe as UID 1000; parser v4 and review module imports verified |
| Streamlit AppTest | Normal chat/comparison and persistent failed-request regressions passed |
| Offline workflow | Seven source-hash/reference-checked cases; no Ollama calls; zero auto-assigned verdicts |

Four suite skips require laptop PDF/reference artifacts absent from this source
checkout. The extra Python 3.11 skip is optional Ragas. The new review tests cover
stale or changed reviews, source/hash/page mismatch, preserved private indexes,
explicit reviewer types, spreadsheet prefixes and failed turns. No GitHub CI or
actual Windows execution is claimed from these local runs.

The previous GitHub `main` was preserved in a local Git bundle before work. The
source update excludes `.env`, personal PDFs, indexes, model weights and backups.
Cloud install/restart instructions were saved as a draft; saving is not publication
or a verified fresh-task restore. Review/save and publish in environment settings
if a reusable cloud snapshot is desired.

## Remaining work

Ordinary Arabic research answers and complete multi-part answers remain unresolved.
The small semantic judge can approve weak answers; independent source review is
still necessary. Before claiming research readiness, repeat the frozen checks with
a demonstrably more capable local model, add genuinely independent paper controls,
and obtain human review. Verify the actual Windows installation and GitHub CI.
The new workflow makes those gaps reproducible and visible rather than reporting
software test counts as answer accuracy.
