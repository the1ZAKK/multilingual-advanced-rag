# Factual facets and Arabic fidelity cycle — 12 November 2026

Starting candidate: `fe63d3d05d9e10c89862792cc07205cb94172554`.
The checkout had no unrelated edits or repository/ancestor AGENTS.md. Required
review follow-up, submitted-review aggregate, continuation guidance, TASKS.md
and the latest failed model recording were inspected before editing. Local
baseline: 444 passed, 5 skipped; Ruff lint and formatting passed. The previous
Windows/Linux Python matrix and Docker checks passed.

The original 3 partial / 1 fail reviews are AI-assisted judgments on recording
39219463133 at d66c62d only. They do not grade this cycle or count as human
review. Existing PDFs, user indexes, settings, backups and reviews are untouched.

## Reproduction before changes

A task-owned local Ollama 0.40.5 CPU runtime was installed from its official
release archive, SHA-256
`dc5eabc292f9949546c90a97c820f924b035e05243b0de13c7f366182562169c`.
The unchanged Qwen2.5-7B-Instruct Q4_K_M comparison profile was downloaded and
independently checksum-verified before import. Its published revision, GGUF
hash, installed digest and import parameters are retained in the runtime report.
No model default or user service was changed; the task service uses port 11435.

Five real semantic-audit controls reproduced a false refusal of a named
benchmark's explicitly reported task count. Adding the source's preceding
declaration did not resolve it, and even the directly worded dataset sentence
was refused on an invented exactness requirement. A concise quantity-and-entity
phrase was accepted; a deliberately changed entity was rejected. These are
exposed diagnostic controls, not independent or unseen evaluation.

## Source-derived facet plans

- Derive each explicit count facet's candidate quantities, role and quote IDs
  directly from native evidence. Match requested evaluation/generation predicates
  where the source supplies them, retaining alternatives when unresolved.
- Guide count prose to begin with a source quantity and counted entity,
  retaining all bounds and qualifications. The model still generates its prose;
  no paper-specific answers or review expectations enter the production plan.
- Select citations for the requested prompting-condition pair. A generic few-shot
  aggregate cannot substitute for a specific few-shot CoT comparison. Keep native
  quotation bytes and reference identities unchanged.
- Scope language/format questions to data-attribute passages and ask for the
  attributes rather than adding unrequested benchmark definitions or sizes.
- Run deterministic numeric, citation and completeness checks before model audits
  so a missing count receives actionable repair feedback immediately. Every
  existing semantic verdict remains required; candidates are not support verdicts.
- Reject a known source quantity relabelled as another recognized entity, even
  when the claimed entity has no separately expressed source count.

The first implementation has 11 new independent-value regressions, including
wrong role, wrong subset/predicate, wrong comparison, irrelevant background,
omitted count, and an explicitly negative semantic judge. Full model recording,
Arabic fidelity checks and an ungraded new packet remain required after the
validated source checkpoint. Software success does not establish answer quality.

## Translation-stage repair and runtime diagnostics

The initial learned-retrieval application probe at `0a9771c` returned a model
HTTP 500 and disclosed an excerpt fallback. The failed output and settings are
preserved; separate direct schema controls returned HTTP 200. No answer-quality
success is inferred from those controls. New bounded HTTP-error tracing retains
the status and response body for diagnosing subsequent server failures.

Ordinary translation now gets a bounded repair of the failed statement before
the application retries the entire answer. Both attempts use the same strict
identifier, numerical, scientific-concept, reverse-content and semantic checks
and the original total deadline. Repair feedback is untrusted and supplied only
to the forward translator; the reverse translator receives only its Arabic
candidate. Failed translations remain in the wire trace.

A small source-activated technical lexicon preserves Modern Standard Arabic and
multiple-choice meaning across documents. It contains no paper names, numerical
answers or expected points. Missing standard-language qualifications are refused
before a reverse translator can silently insert them. Reverse lexical bindings
are activated only by the actual Arabic candidate.

## Local runtime failure and hosted rerun

The complete eight-question learned run at `529f9e7` returned eight explicitly
labelled excerpt fallbacks: the verified local Ollama worker returned HTTP 500
with "unexpected EOF" before generation. The same native prompt was retried
with batch sizes 512, 64 and 32; all failed. Small direct schema/audit controls
work. The cause of the worker exit is unconfirmed; this is a runtime failure,
not evidence that any answer-quality requirement passed.

The runtime log also showed that the provider ignored the count schema pattern
because it lacked an end anchor. Both anchors are now supplied. Deterministic
numeric and semantic validation remain required independently of decoding.
The next hosted recording uses the existing pinned runtime and verified model
on the larger runner. The publication job now preserves the entire wire trace
losslessly with byte hashes alongside the ungraded packet, including HTTP
errors and rejected proposals, without depending on temporary artifact retention.

Real nested-schema probes then confirmed a second compatibility issue: this
provider also ignores regex shorthand `\s`, even with both anchors. Its pinned
llama.cpp b11669 converter explicitly reports that escape as unsupported. The
pattern now uses an explicit space/tab class. Probe outputs and provider warnings
are retained, including the unsuccessful intermediate anchored pattern. The
hosted workflow also retains server diagnostics to expose future converter
warnings and worker exits.

The local 4B learned run at the same implementation now produces a fully
source-audited English ArabicMMLU answer with all four requested quantities.
Its first Arabic question attempt nevertheless emitted an Arabic entity name
inside an English grounding-stage count, then the repair hit a worker HTTP 500.
That failed output remains recorded. Count decoding now binds each native
quantity to its own generic English entity before the separately audited Arabic
translation. Wrong quantity/entity pairings and untranslated intermediate count
prose have additional negative regressions. All existing post-generation checks
are retained. This is an observed grounding-stage defect, not an Arabic
answer-quality pass; the complete rerun is still required.

The local server accumulated hundreds of MiB of optional cross-request prompt
snapshots before worker exits. The precise termination cause remains unconfirmed.
The final owned-runtime experiment sets the pinned llama.cpp server's documented
`LLAMA_ARG_CACHE_RAM=0`, retaining full context and all validators. This affects
prompt reuse storage, not evidence size, model weights or KV precision. The
recording workflow passes it explicitly into the owned container and records the
client declaration separately from server introspection; the retained server log
must confirm the effective cap. Existing user services and settings are untouched.

With the optional cache disabled, the server confirms prompt caching is disabled
and the 4B model reaches Arabic source/translation audits without the earlier
worker crash. The English control passes again. Arabic count statements preserve
their values, but the score's language-centric qualification is mistranslated as
a model "central in" a language. The existing round-trip guard correctly refuses
that proposal and its bounded repair; this is an observed meaning error.

Source-activated language-centric terminology now covers six languages using the
same generic rule. It supplies the actual qualification in Arabic and binds it
back only when the Arabic candidate contains that phrase. Merely mentioning a
language or a central model cannot pass the qualification guard. Independent
French controls check that the rule preserves language identity, and missing or
switched qualifications remain rejected. No number, paper identity or expected
answer is encoded. Existing numerical, reverse-content and semantic checks remain
required; these changes still need a fresh real-model recording.

The final recording also preserves server diagnostics losslessly on its isolated
results branch, with exact byte hashes in the trace manifest. Earlier hosted
recordings retain server logs in their original artifacts when available. This
keeps runtime failures and effective cache/schema settings inspectable after
artifact expiry, alongside every rejected proposal. The ungraded review packet
continues to contain only the recorded source-bound answers and review materials.
