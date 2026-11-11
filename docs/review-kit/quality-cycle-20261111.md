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
