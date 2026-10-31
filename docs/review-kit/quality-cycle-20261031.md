# Numerical completeness and Arabic quality cycle — 2026-10-31

The starting candidate is `4e7aa6a43dedba2b63807b3531bfc9e6bdd22143`.
Its Quality run 41184110709 passed all four Windows/Linux Python jobs and the
Docker build. A clean local baseline reproduced 406 passed and 5 skipped.
No repository AGENTS.md exists in this checkout. TASKS.md, the current
architecture, cloud continuation and submitted-review follow-up were read.

The submitted 3 partial / 1 fail AI-assisted verdicts still describe recording
39219463133 at d66c62d only. They are not independent human review and do not
grade the changes or new outputs in this cycle. Prior PDFs, indexes, settings,
review artifacts and the checked-in old ZIP remain untouched.

## Implementation

- Reuse the existing abstract/prose routing correction. Explicit abstract
  requests now select that document's abstract evidence without appending
  competing result-table or appendix passages. Ordinary overview and numbered
  table behavior remain available. Anchors retain real page/chunk identities.
- Expand explicit coordinated count nouns into separately required answer
  slots in English and Arabic. Values are generated from source evidence;
  no paper identifiers, expected values or review guides enter production code.
  Over-budget requirements are disclosed, never collapsed to conceal omissions.
- Reject missing source-expressed count roles, changed count/entity bindings,
  changed prompting-condition/percentage bindings, omitted reported comparison
  ratios and numerical citations that contribute only a different comparison.
  These limited deterministic checks supplement the unchanged quote, native-PDF,
  output-language, reverse-translation and semantic validation requirements.
- Verify numerical relationships in forward Arabic and reverse English claims.
  Preserve the strict round-trip content guard: faithful paraphrases can still
  be refused; that limitation has not been hidden or relaxed.
- Retain raw generation/translation responses and explicit validation rejections
  when tracing is enabled. Record exact source commit and context/token/timeout
  settings in new run reports.
- Export an ungraded portable review packet with START_HERE.html, the exact new
  answers, byte-verified original PDFs, frozen review expectations, UTF-8 CSV and
  a per-file SHA-256 manifest. Refuse existing packet/archive paths.

## Real-model recording

`data/evaluation/quality-cycle-20261031.json` retains the four exact submitted
questions unchanged and adds four English/Arabic source-review controls about
question language/format and distinct tasks versus individual evaluations.
All are development controls, not unseen evaluation. Expected points are never
passed to generation. Both PDFs are English-language papers about Arabic NLP;
this run does not establish native Arabic source-prose performance.

The hosted recording workflow keeps the existing checksum-pinned Instruct GGUF
and pinned Ollama container. It records all eight cases through RAGService,
learned multilingual embeddings, native source verification and the ordinary
generation/translation pipeline. Experimental question-coverage judging stays
explicitly disabled, matching the reviewed profile; structural count slots and
the new numerical guards are enforced regardless of that optional switch.

The workflow exports the full trace and a portable ZIP, and publishes outputs
to its isolated `ci-evaluation-RUN_ID` branch. Exact runtime, model revision,
GGUF checksum, source commit, settings, modes, warnings, source checks and new
answers must be inspected in that recording. Successful software tests or a
completed recording job do not establish answer-quality acceptance.

The current cloud checkout has no local Ollama executable or running model
service. Hosted verified execution is used rather than substituting mock model
answers. Model and CI outcomes will be recorded after the pushed run completes.
Independent human source review and actual laptop acceptance remain pending;
no stable release is authorized or published.

## Cross-platform and source follow-up

The first implementation recording uses commit `2a085bf` and model workflow
41185838438. Its Quality run 41185838432 passed Linux and Docker but exposed
two Windows regressions: an implicit cp1252 test read of Arabic JSON and native
backslashes in the new ZIP manifest. Both were corrected with explicit UTF-8
and POSIX archive keys. The failed run is retained, not relabeled a pass.

Direct original-PDF checks also found that inline `Abstract—` and IEEE Roman
section headings were not classified by parser v4. Parser v5 recognizes those
boundaries without rewriting source characters. Fresh ArabicNumBench abstract
evidence is now confined to its 1,623 source characters on PDF page 1, excluding
index terms, introduction and related work. Existing stored indexes keep their
extraction version and are not automatically reindexed or migrated. Compound
model-strategy counts are also excluded from bare model counts.

After these fixes, local validation is **424 passed, 5 skipped**; Ruff lint and
format checks pass. A new hosted run is required for the corrected commit;
the earlier model run cannot validate the parser or Windows follow-up changes.

The final pre-recording pass also rejects qualitative prose in a requested
numerical score slot, and makes compact-report/trace reads explicitly UTF-8 on
Windows. Its local checks are **427 passed, 5 skipped**, with clean Ruff lint,
format and diff checks. No previously failed CI result has been discarded.

### Decimal multipliers

A further deterministic regression exposed a pre-existing tokenizer defect:
`numeric_atoms("2.8x")` returned the integer `2` by regex backtracking, while
Arabic `2.8 ضعف` returned `2.8`. That could reject a faithful multiplier
translation and conflate different decimal ratios. Ordinary generation and
translation now share the decimal/grouping/multiplier tokenizer with the new
relation checks. The tokenizer still excludes model identifiers and recognizes
Arabic digits and grouping marks. A regression also rejects claiming `2x`
from a premise reporting `2.8x`. Local validation is **428 passed, 5 skipped**.
Earlier recordings remain associated with their own code; the final recording
must use the commit containing this correction.

The round-trip content check also canonicalizes exact numeric unit spellings
(`x`, `times`, `-fold`; `%`, `percent`, `per cent`). It preserves an explicit
unit token and separately checks numeric values, so dropped units or changed
ratios still fail. This is formatting equivalence, not a general synonym
exception. Existing negation, content, identifier and semantic checks remain.
Local validation after the unit regressions is **429 passed, 5 skipped**.
