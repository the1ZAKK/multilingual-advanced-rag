# Compound questions and polarity checks — 19 October 2026

This continuation repairs concrete answer-format and safety defects. It does not
establish reliable general research answers or declare the project fully finished.
The restarted cloud retained the source and completed run artifacts; its processes
had stopped. Ollama, the API and dashboard were restarted and checked separately.

## Changes

Explicit coordinated interrogatives now receive one required answer slot per part.
Examples include “How is the content curated, and what mechanism is used?” and
“Does it assess retrieval? Distinguish current scope from future work.” Ordinary
conjunctions such as “models and datasets” do not trigger splitting. The supported
structural patterns cover two to four explicit parts, not arbitrary natural-language
decomposition. If recognized parts exceed the statement budget, the assistant
returns a labeled fallback rather than silently dropping parts.

Each part is generated separately with prioritized, existing quote IDs. Quote
ranking suppresses repeated paper names; quotes, page bindings and IDs are never
rewritten. Arabic questions over English passages keep full evidence when the only
lexical overlap is a Latin paper name. The single-question system prompt remains
unchanged. Every claim still passes normal citation and semantic checks, and all
calls share the existing total generation deadline.

A deterministic veto rejects a narrow, explicit English contradiction: the same
subject both “does not assess” and “assesses” substantially overlapping objects.
It runs before model-based semantic approval and participates in the existing
repair/fallback flow. Distinct subjects, future-tense predicates and different
objects are not conflated by the covered patterns. This is a conservative guard,
not general inference: paraphrased contradictions, implicit roles and Arabic
negation still need independent source review. Ambiguous overlap can cause abstention.

## Real-model findings

The frozen seven-case suite still uses original SHA-verified ArabLegalEval,
AraTable and AceGPT PDFs, learned MiniLM, hybrid retrieval, no cross-encoder,
Qwen3 1.7B Q8, 6000 context characters, three statements and a 150-second total
generation deadline. Gold answers are never sent to the generator. These are
previously exposed development controls, not an unseen benchmark.

Before the polarity guard, the completed multipart run showed:

| Case | AI-assisted source review |
|---|---|
| Legal problem/source material | Partial: identifies the gap but not the requested source material |
| Legal current/future retrieval | Fail: contains conflicting negative and positive claims |
| Same question in Arabic | Fail to deliver a verified Arabic answer; English-source fallback |
| Three table tasks | Fail to generate; semantic rejection and source fallback |
| Table curation/mechanism | Pass: LLM generation, human filtering/verification, self-deliberation |
| Unsupported process reward model | Pass: evidence-limited refusal, no fabricated introduction |
| Named zero-shot table comparison | Pass: both models and all four values match Table 5 |

This is **3 pass, 1 partial, 3 fail** under explicit **AI-assisted** review, with
zero independent human reviews. Valid quote provenance is not answer correctness.
The model judge incorrectly approved the conflicting legal answer. That observation
motivated the deterministic polarity regression instead of weakening validation.

The polarity-guard retest rejected both conflicting attempts and returned labeled
source excerpts. This passes the safety regression but **does not pass** the
original request for a complete generated answer. Existing evaluation-index file
fingerprints were unchanged. Subsequent runtime retests and software verification
are recorded in the compact results below; timeouts remain failures to answer.

[Run identifiers, settings, outcomes and review notes](validation-results/compound-20261019.json).
Full source-containing traces remain in ignored local `reports/`.

## Model access and remaining acceptance gates

The cloud has sufficient RAM for an optional 4B comparison. Official metadata was
retrieved for `Qwen/Qwen3-4B-GGUF`, revision
`bc640142c66e1fdd12af0bd68f40445458f3869b`, file
`Qwen3-4B-Q4_K_M.gguf`, 2,497,280,256 bytes, publisher SHA-256
`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
The native Hugging Face download failed at the network-restricted CDN
`us.aws.cdn.hf.co`; the direct download route was also denied. **No 4B model was
verified, imported or evaluated.** No laptop settings or default model were changed.
Public model metadata does not require a new user token.

For this optional comparison, add `us.aws.cdn.hf.co` to the cloud environment's
allowed domains, retaining all existing domains, then retry the verified download.
The current complete allowlist cannot be read, so it was not replaced with an
incomplete guessed list. A saved configuration or domain entry alone does not
prove a download works. No TLS or artifact-integrity verification was disabled.

Remaining gates are complete multilingual answers, broader independent paper
review, human review, and the user's own Windows installation. Repository CI is
separate: the prior `49bab7d` branch commit's Windows/Ubuntu Python 3.11/3.12 jobs
and Docker build passed in [GitHub Actions](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/39052779188).
That does not demonstrate live model quality or execution on the user's laptop.

## Software and runtime verification

- Python 3.12: **384 passed, 4 skipped**. Python 3.11: **383 passed, 5 skipped**,
  with one upstream HTTPX/Starlette deprecation warning. Artifact-dependent skips
  and the extra optional Ragas skip are not counted as passes.
- Ruff lint, formatting and Git whitespace checks passed.
- The restarted API passed health, synthetic upload, all three retrieval modes,
  cited extractive chat and deletion of only its temporary probe document.
- Real Chromium exercised the dashboard, submitted an offline question and downloaded
  a review export with the actual request, answer and citations. The test clicked the
  visible checkbox label and selected the download button by its accessible role.
- The post-restart method retest returned both requested relations with valid native
  quotes. The task-list question timed out and returned source excerpts. Earlier
  successful method runs and later timeouts are kept separate; they are not combined
  into a final-version answer-accuracy score. CPU performance and model quality remain
  practical constraints on the existing deadline.
- Existing evaluation-index fingerprints remained unchanged during targeted retests.
  The source package excludes active settings, indexes, PDFs, model weights and caches.
- A separate cloud application index now holds the three SHA-verified papers using
  learned MiniLM embeddings. It is separate from immutable evaluation runs and the
  laptop corpus. The API's learned-profile upload/retrieval/chat/cleanup probe passed;
  health confirmed three papers, a loaded learned embedder, and the installed local
  Qwen tag available. The earlier browser export check used the explicitly offline
  profile; it is not reported as learned generation.
- The exact source archive was tested against the uploaded baseline, published main
  and previous validation branch. Preflight/application, valid backups, unchanged
  private-file sentinels, idempotence and unknown-edit refusal passed for all three.
