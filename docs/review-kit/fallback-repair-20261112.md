# Four fallback repairs — 12 November 2026

Starting candidate: `7def51e44aa0ec4b8827d7b3a4ab174fcedf603c`.
The two pre-existing untracked diagnostic-archive files and their local recovery
branch are preserved unchanged. Raw archives, wire traces and recovered job logs
stay local and are not included in source updates or review packets.

## Existing hosted run recovered before a new evaluation

Run `43086599560`, source `d2e5b89`, attempt 1 completed with failure at
05:01 China time on 12 November. The model-job log is now accessible and has
been saved locally. Evaluation, compact export, source/model identity checks
and review-packet creation succeeded. Artifact creation then failed on five
requests receiving HTML rather than JSON; runtime cleanup completed.

The log records all eight output modes: seven explicit excerpt fallbacks and
one generated English task/evaluation distinction. Only the Windows acceptance
artifact is listed, and the result branch is absent. Exact model answers, wire
responses and the generated ZIP are not recoverable from the available artifacts
or console log. Those answers are not reconstructed or inferred from output
modes. The separate metadata summary records this limitation. No duplicate run
was started while the old recording's status was unresolved.

## General repairs

- Bind ordinary provider terminology from the current English statement, and
  reverse-bind it using only the actual Arabic candidate. Missing/repeated terms,
  altered numbers and negative semantic verdicts remain refusals.
- Ask attribute questions for qualified noun phrases rather than repeating a
  dataset's subject or adding an existence/availability assertion. The existing
  content guard still rejects an added existence claim.
- Replace arbitrary, unbounded pre-entity words in count decoding with the same
  finite qualifier vocabulary already used by numerical validation. Both counts,
  counted entities and source bounds remain required. This removes a grammar
  path that generated runaway words and unterminated JSON.
- After overlap deduplication, retain only candidates for available quote IDs.
  When every entity in a two-count distinction has one unambiguous value and
  source, require those native references in the model's response. Do not append
  or repair missing citations after generation. Ambiguous candidate sets retain
  ordinary selection and validation. Native provenance, numerical relationships,
  completeness and semantic verdicts remain independently necessary.

## Software checkpoint and live gate

525 tests passed, 5 skipped; Ruff lint and formatting clean. Sixteen added
regression cases cover provider terminology and blind reverse input, changed
numbers/entities, repeated terms, unbounded grammar growth, source deduplication,
two-count omissions/swaps, missing citations, malformed slots, and a negative
semantic judge. No paper identities, review expectations or fixed paper numbers
were added to production code. The abstract-routing correction is retained.

The next isolated eight-question application run uses the existing verified 4B
profile and the same 6000-character evidence, 8192-token context, four-statement,
300/600-second timeout and temperature-zero settings as the recovered local
recording. The software checkpoint does not accept its future answers. Each
failed attempt will be retained locally with its exact source commit, verified
model and settings. Only permitted answer/PDF/CSV review materials will be
published. All questions are exposed development controls; human review and
stable release remain pending.
