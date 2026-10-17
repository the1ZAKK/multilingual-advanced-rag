# Continuation report — 18 October 2026

This follows the [previous release](validation-20261016.md). It improves Arabic
failure handling and repeatable source updates. It does **not** establish that
ordinary Arabic research answers are generally reliable or the project is 100%
validated. No new model was downloaded, no paid service used, and no push or
external deployment performed. Uploaded laptop settings, documents and indexes
were not available here and were not changed.

## Arabic generation and translation

Adding native Arabic instructions to the direct generator was insufficient.
The larger-context ArabicMMLU run timed out. A smaller-context run returned Arabic
but omitted the requested counts/source information and included unrelated facts.
It must not be counted as a successful answer simply because its mode was `ollama`.

For ordinary Arabic questions with English-only evidence, the new route:

1. Generates English claims and validates their exact source references,
   numerical support and per-claim semantic grounding.
2. Translates only those verified claims into Arabic; references are copied by
   code and never entrusted to the translator.
3. Independently reverse-translates each Arabic candidate. This translator sees
   only the Arabic candidate, not the English claim or the source passages.
4. Checks numeric values (including Arabic digits/grouping and attached
   conjunctions), identifiers and names, and rejects changed round-trip content.
5. Audits complete meaning against the original English claim and its actual
   premises, then validates Arabic prose/citations before `translated-answer`.

All calls share the existing total deadline. Failures return labeled source
excerpts. Native Arabic evidence and source-authored contribution lists retain
separate existing paths. `RAG_TRANSLATION_MODEL` optionally selects an installed
model for ordinary translations; otherwise it uses the configured Qwen model.
No models are downloaded by this setting. Runtime health and evaluation exports
record the translator and generation limits.

## What the live model checks showed

The actual Qwen3-1.7B Q8 model produced mistranslations, including school exams
becoming teachers and the Levant/Gulf becoming unrelated geographical regions.
Its semantic judge sometimes approved these errors. Independent checks must
therefore not rely solely on affirmative model verdicts.

Qwen2.5-1.5B Q4, already installed in this cloud, also changed research terminology
and geographical names during a direct translation comparison. It is **not**
recommended here as a demonstrated quality fix. These development cases are not
unseen benchmarks; the model quantizations differ from the laptop registry tag.

The final live synthetic controls are recorded in
[guarded Arabic checks](validation-results/guarded-arabic-checks.json):

| Control | Final result |
|---|---|
| Source says the dataset contains 64 questions | Correct Arabic count answer, `translated-answer`, source citation retained |
| Source says 64 sensor records; translation adds interconnected-device observations | Rejected by content guard; source excerpts required |

The numeric guard also closes an independent defect: `14,575` no longer supplies
separate values `14` and `575`. Equivalent `١٤٬٥٧٥` and a number attached to Arabic
`و` are recognized without modifying the source quote. A claim of `14` questions
is rejected against a source count of `14,575`.

## Limits and tradeoffs

The lexical round-trip veto is intentionally conservative. Apart from simple
plurals and data/dataset, it does not silently equate English content words;
faithful synonymous translations can therefore fail. It is still not a semantic
proof: matching words can conceal wrong roles or order, and the model judge can
err. Title-case name checks also have false rejections. Translation adds latency.
Question coverage, ordinary Arabic research quality and mixed-language evidence
need independent human evaluation. A successful one-sentence fixture is not a
paper-level accuracy result. English generation also retains model-quality limits.

## Updating an existing installation

The updater now accepts explicitly recorded previous-release hashes in addition
to the uploaded baseline and the incoming version. It still refuses unknown local
edits before any source writes. Replaced source is backed up and private runtime
directories remain protected. The developer helper `scripts.prepare_update`
creates a manifest from a Git baseline and a previous shipped manifest, includes
new source files, and refuses deleted/reverted shipped source without manual review.
See [upgrade instructions](upgrading.md). End users do not need Git to apply an update.


## Final executed checks

| Check | Result |
|---|---|
| Python 3.12.14 suite | 353 passed, 4 skipped |
| Python 3.11.16 base/development suite | 352 passed, 5 skipped, 1 upstream deprecation warning |
| Ruff lint / format | Passed; 87 Python files formatted |
| Docker build | Passed; non-root image |
| Restarted source API | Functional PDF upload/retrieval/cited-chat/cleanup probe passed |
| Dashboard | Existing dashboard health check passed; previous real browser screenshots retained |

Skipped PDF/reference fixtures were not included in the source upload. The extra
Python 3.11 skip is optional Ragas. Windows and GitHub CI have not been run for
this revision. No new visual browser verification is claimed for this continuation.

The final real ArabicMMLU question returned `extractive` after rejecting the
changed place names, in approximately 128 seconds on this CPU. This is a safe
failure, **not** a solved research answer. It used 6000 context characters, three
ordinary statements, a 150-second generation budget and real Qwen3-1.7B Q8.
[Recorded paper failure](validation-results/guarded-paper-summary.json).
The example configuration remains 8000 characters/four statements; the smaller
CPU diagnostic settings are not a general quality recommendation.

The final container passed the PDF upload/retrieval/cited-chat/cleanup smoke as
UID 1000. The archived updater was tested end to end against both supported
installations: 76 changed source files from the original uploaded ZIP, and 19
from the previous release. Each preserved all three private sentinels, produced
a valid backup, changed nothing on a second run and refused later unknown edits.
The archive's 121 source inventory entries passed SHA-256 checks.
