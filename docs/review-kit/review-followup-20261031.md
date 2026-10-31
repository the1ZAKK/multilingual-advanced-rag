# Submitted source review — 2026-10-31

The returned UTF-8 CSV passed the existing review validator against the exact
four-case packet: IDs, questions, source bindings, response fingerprints, and
all other immutable fields match. Only reviewer, reviewer_type, verdict, and
notes were editable. The uploaded CSV is retained unchanged in ignored local
reports; its SHA-256 is recorded in the aggregate JSON.

Outcome: **3 partial, 1 fail, 0 pass**. All four rows are explicitly
`ai_assisted` with reviewer `ChatGPT`. This is not evidence of completed
external independent human review. The reviewer reports inspecting PDF renders;
the observations below are the submitted review findings, not a new model run.

The reviewed answers are GitHub recording 39219463133, source commit d66c62d,
before the abstract-routing correction at 1ad5f63. Do not apply these verdicts
to later answers without re-reviewing those exact outputs.

| Case | Verdict | Submitted finding / follow-up acceptance check |
| --- | --- | --- |
| mmlu-size-en | partial | Include the missing **35 evaluated models**, alongside 40 tasks, 14,575 questions, and 62.3%, all supported by PDF page 1. |
| mmlu-size-ar | fail | Provide a complete Arabic answer with all four requested values from PDF page 1. Exclude unrelated exam-option excerpts. Do not substitute the 59.9/62.4 table averages on pages 16–17 for the abstract's 62.3%. |
| numbers-prompt-en | partial | Facts are complete, but remove S6 from the specific 2.8x CoT comparison: its quoted page-3 aggregate is 2.7x (77% versus 29%). Retain evidence supporting 80.06% versus 28.76% and reported 2.8x. |
| numbers-prompt-ar | partial | Render the supported facts in Arabic: 71 models, 10 providers, 59,010 individual test cases, 80.06% versus 28.76%, and reported 2.8x. Do not confuse 210 distinct tasks with individual evaluations. |

## Follow-up work

1. Re-run both Arabic cases with the existing abstract-routing correction and
   verify the exact new answers, not only retrieval or generated mode.
2. Address completeness for multiple requested quantities; preserve strict
   number, source, and quotation checks.
3. Check each citation's relevance to the particular numerical comparison.
   Joint support from other citations does not justify an unrelated extra source.
4. Export a fresh ungraded packet bound to the fixed commit and exact new
   outputs. Preserve these submitted verdicts and the earlier failures.
5. Keep stable-release acceptance pending: software CI and AI-assisted review
   do not replace the agreed human source review or target-installation checks.
