# 3LM contribution-query diagnosis

For the later local resumption, actual additional-paper failures, repairs and current tests, see [the October 16 evaluation](resume-evaluation-20261016.md). The 3LM results below do not establish generalization.

Local diagnosis on 2026-10-10 (Asia/Shanghai). Nothing was pushed or deployed.

The later Arabic, models/results, and unsupported-claim failures and their independent
semantic checks are documented in [the grounding and language follow-up](grounding-language-diagnosis.md).
This earlier report does not establish correctness for those broader queries.

**Follow-up correction:** the initial checks below were insufficient to establish
semantic completeness. Valid source quotes and whole-answer keywords allowed
incorrect contribution prose to pass. The follow-up section records the new
reproduction, source-derived fix, and stronger per-item verification; it supersedes
the earlier final-validity claim.

## Source and questions

The actual uploaded PDF is `3LM-Bridging Arabic, STEM, and Code.pdf`, 23 pages,
SHA-256 `661a0d336318fa5a8d80294b8b2a3bbf0dd79ffa1a0691cd05b54e9f5cdcd254`.
The original local PDF was located in the user's `Desktop/source papers` directory.
Its bytes match the uploaded document ID exactly. Pages 1 and 2 were rendered and
visually inspected, alongside native, geometrically sorted, and indexed extraction.

The authors' contribution statement starts at the bottom of page 1's right column
and continues at the top of page 2's left column. It describes:

1. Three comprehensive STEM/code benchmarks, with native Arabic curation, synthetic
   generation, and careful translation with human verification (pages 1-2).
2. Evaluation of over 40 state-of-the-art Arabic and multilingual LLMs (page 2).
3. Analysis of cross-task correlations and robustness, with insights into model
   behavior and relationships between cognitive capabilities (page 2).

These are contributions, not the three benchmark types. The problem is the gap in
native, scientifically grounded Arabic evaluation for STEM and programming, given
existing benchmarks' emphasis on linguistic/cultural/religious tasks or translations.

The exact reproduced dashboard questions were:

> According to the 3LM paper, what problem does the work address, and what are its main contributions? Cite supporting passages and page numbers.

> List all three contributions explicitly stated by the authors of the 3LM paper. Number them 1–3 and cite supporting passages and pages. If the retrieved evidence does not contain all three, state which are missing.

## Failure chain

The baseline used the persisted upload, learned MiniLM embeddings, real FAISS/BM25,
the configured mMARCO cross-encoder, and local `qwen3:1.7b` through Ollama.

* **Reading order:** `get_text(sort=True)` interleaved the two columns on page 2.
  Sentences beginning with “Second” and “Third” were mixed line-by-line with unrelated
  Related Work. Native text order and the rendered page show the real continuation.
* **Chunking:** fixed 900-character windows with 150-character overlap cut words,
  sentences, and sections. The page-1 contribution chunk `p1:3000-3775` ended during
  First. Page-preserving chunking inherently separated the continuation on page 2.
* **Ranking:** the exact failed question returned pages **3, 18, 1, 17, 1** under
  hybrid top-five retrieval. Appendix prompts on pages 17-18 matched words such as
  “list”, “three”, “output”, and “supporting”. The page-2 continuation did not reach
  Qwen. Appendix text was inspected as data, and no appendix instruction was executed.
* **Evidence selection:** it retained independent top-five chunks, without neighboring
  page expansion. The original excerpt fallback also truncated each passage to 800
  characters, further hiding available text.
* **Generation/validation:** the exact question's first wire response invented a quote
  ending and flattened PDF line breaks. Validation raised
  `ValueError: Quote is not present in the cited passage`. The retry kept **only S1**,
  dropping the contribution lead. It returned `answer` and `details` without
  `citations`, raising **`KeyError: 'citations'`**. The final output was extractive
  with the generic warning reported by the user. This was invalid grounded output,
  not an Ollama connectivity failure. A shorter preliminary question even produced
  `generation_mode=ollama` with the wrong three contributions after that same lossy
  retry; mode alone was therefore explicitly rejected as a correctness measure.

The actual original wire responses, evidence, and errors are preserved locally in
`reports/3lm-diagnosis/baseline-generation-wire.json`, `baseline-generation.log`,
`baseline-exact.json`, and the rendered/extracted page files. These diagnostic files
contain document/query content and are opt-in local reports, ignored by `.gitignore`;
production logs do not record full questions, answers, or passages.

## Changes

English two-column pages now read down the left column and then the right, with
full-width blocks separating bands. Predominantly Arabic pages retain native glyph
order. Text is kept unchanged within blocks, including line-end hyphens. The parser
records section metadata, recognizes the authored threefold contribution paragraph,
and makes bounded sentence/paragraph windows without crossing known section headings.
Offsets and IDs refer to this extraction's text, not PDF byte offsets.

For contribution questions, chat examines the configured retrieval candidate pool,
prioritizes the explicit authored statement, and takes neighboring chunks within the
same document/section across a page break. Problem summaries additionally prefer the
abstract. Neighbors preserve actual chunk IDs and pages; they have no invented dense,
BM25, RRF, or reranking scores. Expansion is deduplicated and bounded by the existing
context budget. It stops at section boundaries. Incidental appendix matches do not
displace the explicit contribution statement. The supplied appendix schemas remain
untrusted document data in both initial and retry prompts.

Qwen receives a catalogue of immutable paragraph/sentence quotes, each with a unique
source-bound reference such as `S2:Q3`. A constrained schema requires independently
cited statements. The server resolves references to original substrings, verifies
them, and renders source markers and page numbers. It neither normalizes a failed
quote into acceptance nor invents missing citations. Legacy raw-quote validation
continues to reject altered or fabricated quotes. Every rendered answer still passes
the citation validator. The three-contributions request requires three separately
cited statements in authored order; the problem summary requires the problem plus
those three. Each contribution must cite its corresponding authored source sentence.
The first also cites the page-2 fragment completing its page-1 source. For this explicit
threefold intent, named first/second/third slots bind decoding to those exact source groups;
Qwen writes the summaries, while references stay constrained to the actual document.
A problem statement must cite and describe the supplied evaluation-gap passage.
Contradictory claims that the authors state one, two, or four contributions are rejected.
Missing Second/Third evidence is explicitly reported and is not sent to Qwen as though
complete. Retries preserve **all** evidence and include the precise validation error.

The fallback includes complete selected passages rather than silently truncating them.
Warnings now identify the actual generation/validation failure. During verification,
wrong per-item supporting references were rejected and corrected on retry, retaining
the complete evidence. Citation validation remains a structural/exact-source check,
not a general semantic entailment oracle; prose was independently compared with the PDF.

## Migration and preservation

The API worker was stopped for migration. SQLite's backup API saved the entire original
corpus to `data/index/backups/20261009T174711146608Z.sqlite3` (UTC filename). A second
backup at `20261009T175053021758Z.sqlite3` precedes the final heading-metadata refinement.
The original backup contains the original 148-chunk 3LM representation.

`scripts/reindex_document.py` verifies the original PDF hash, embeds replacement chunks,
constructs a candidate index, and commits the per-document replacement atomically.
The 3LM document now has **140 chunks** and extraction version `columns-sentences-v2`.
Its ID, name, 23-page count, and original 1,515,570 PDF bytes are preserved. Its source
bytes are stored in `source_pdfs` inside SQLite. The two other documents and their
vectors are preserved. FAISS's checksum cache and BM25 are rebuilt from the authoritative
corpus. The restarted API serves the migrated index; no new upload or deletion was needed.

The contribution evidence is now `p1:3524-3819` and `p2:0-751` under the same document ID.
Old offset-based chunk IDs cannot be reused in new evaluation annotations. Consult the
backup for old citations and reannotate datasets against current IDs. Future uploads
retain source PDF bytes, and future explicit reindexes can use them without `--pdf`.
Other legacy documents are not silently migrated. Stop the API for CLI migrations to
avoid another process serving stale in-memory indexes. To roll back, stop the worker,
restore the original SQLite backup, and restart; checksum mismatch rebuilds derived caches.

## Retrieval and reranking evaluation

Ground truth consists of the two contribution chunks, manually checked against PDF
pages 1-2. A candidate rank of “outside” means not in the 30-result candidate pool.

| Question | Mode | Page-1 lead rank | Page-2 continuation rank | Recall@5 | Selected evidence recall |
|---|---|---:|---:|---:|---:|
| Problem + contributions | Dense | 3 | outside | 0.5 | 1.0 |
| Problem + contributions | BM25 | 1 | 26 | 0.5 | 1.0 |
| Problem + contributions | Hybrid | 1 | outside | 0.5 | 1.0 |
| Problem + contributions | Hybrid rerank | 11 | outside | 0.0 | 1.0 |
| Three contributions | Dense | 1 | outside | 0.5 | 1.0 |
| Three contributions | BM25 | 2 | 17 | 0.5 | 1.0 |
| Three contributions | Hybrid | 1 | 29 | 0.5 | 1.0 |
| Three contributions | Hybrid rerank | 3 | 11 | 0.5 | 1.0 |

The real cross-encoder takes approximately 1.96-1.97 seconds of additional warm CPU
reranking time for these queries. It does not improve Recall@5 and worsens the summary
question. Hybrid remains the default; broader candidate inspection and same-section
neighbor expansion recover the continuation independently of that rank instability.
These are two local cases, not a broad multilingual benchmark or statistical quality claim.
Full rankings and timings are in `reports/3lm-diagnosis/retrieval-evaluation.json`.

## Reproduction and tests

The browser-control environment exposed no browsers, and an in-app browser could not
be opened. Dashboard tests therefore use Streamlit's **AppTest**, executing the actual
`app/streamlit_app.py`, submitting the exact chat questions through its chat input,
and using its actual HTTP calls against the running API and real Ollama. This is not a
mocked live generation check or a claim of pixel-based browser verification.

Run from the project directory with the virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_3lm.py --phase followup-verified --repeats 3 --pdf 'C:\Users\brimz\Desktop\source papers\3LM-Bridging Arabic, STEM, and Code.pdf'
.\.venv\Scripts\python.exe scripts/evaluate_contribution_retrieval.py --pdf 'C:\Users\brimz\Desktop\source papers\3LM-Bridging Arabic, STEM, and Code.pdf'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src api app tests scripts
```

`scripts/trace_generation.py --phase NAME` provides an opt-in local wire trace without
overwriting the preserved baseline. `slots.json` records individual repeated
dashboard responses and independent checks against native PDF text, source identities,
page numbers, all three concepts, authored contribution quote coverage, and numbering.
Intermediate phases are retained too: manual review caught an incorrect “two
contributions” phrase despite complete concept coverage, and an earlier problem quote
combined the Abstract heading with an added paragraph separator. Those findings led to
the count/gap checks and paragraph-separated quote catalogue, without relaxing exact
source validation.
These source/keyword checks supplement manual prose review; they are not LLM judge scores.

Initial validation reported **6/6 live dashboard submissions passed** using the earlier
checks. That result did not establish per-item semantic completeness: the follow-up
reproduction below exposed accepted wrong prose despite valid quotes and references.
The initial named-slot runs had no validation retries. Initial regression suite:
**54 passed, 1 skipped** (optional Ragas dependency), with **Ruff clean**. The existing
FastAPI test-client deprecation warning remains. A byte-for-byte backup comparison verifies
the other documents, their chunks, and their vectors are unchanged, and hashing the stored
source-PDF blob reproduces the original document ID. API and dashboard remain local.

Regression tests cover two-column extraction, page-continuation expansion, sentence
offsets, source-bound references, per-item contribution coverage, missing evidence,
appendix schema injection, full-evidence retries, repeated dashboard submissions,
and atomic source-preserving reindexing including failure and restart cases. The small
fixture contains the actual authored contribution paragraph, not an invented benchmark.

## Follow-up: author contributions versus benchmark components

The actual PDF was re-read, its SHA-256 rechecked, and both rendered pages inspected.
The "contributions are threefold" sentence at the bottom of page 1 continues at the
top of page 2. Its full content is preserved in `followup-source.json`. Native STEM,
synthetic STEM, and translated code benchmarks are components of contribution 1.
The authors' Second contribution evaluates over 40 Arabic and multilingual models
in scientific/programming domains. Third analyzes correlations and robustness, with
insights into behavior and relationships between cognitive capabilities.

The live dashboard baseline (`followup-baseline.json`) reproduced a materially wrong
summary with `generation_mode=ollama` and valid source references: the problem was
repeated in the First slot, benchmarks appeared in Second, and evaluation appeared
in Third. Analysis was omitted. The renderer numbered the problem as item 1, so the
answer displayed four numbered items. The separate three-contributions query had
the right three broad categories but omitted contribution 1's construction methods
and human verification. Neither result establishes full authored-list completeness.

A direct validator reproduction also accepted native/synthetic/code benchmark
components as First/Second/Third when supplied with the correct forced references.
The legacy raw `answer`/`citations` path accepted a fourth numbered item because it
bypassed structured per-item validation. Source captions repeated when multiple
different verified quotes shared a label (including the reported S8 pattern).

The repair derives complete ordinal-led author sentences from actual evidence in
document/page/offset order, including cross-page continuations. The response schema
binds both text and references to those dynamically derived statements. The validator
requires that full source wording; it does not claim that lexical similarity or valid
quote IDs prove paraphrase meaning. Ambiguous documents, absent ordinals, missing
continuations, and cut-off sentences fail closed. Unknown references, altered quotes,
conflicting quote fields and uncited claims retain their strict rejection behavior.
Raw legacy answers cannot bypass this authored-list path. A requested problem appears
as a separately cited, unnumbered paragraph; contributions are exactly items 1–3.

This behavior has **no 3LM answer, document ID, model count, benchmark name, or action
word hardcoded in production**. Full author statements are intentional source output,
with `generation_mode=source-contributions` and a dashboard caption identifying the
original source language. Qwen receives the complete evidence and source-derived slot
texts; problem prose can still be generated. Other questions retain normal generation.
Retries preserve all sources and log the precise validation error, including
`Unfaithful First contribution: require the complete source-derived author statement;
quote IDs alone do not validate paraphrase meaning` for component/partial prose.

The dashboard groups source captions by label and source identity, keeping every
distinct verified quote in an expandable view. The API still validates every quote
before presentation; grouping never hides an invalid second quote. Existing saved
chat messages remain historical responses; fresh submissions use the repaired path.

No further extraction/index migration was necessary. The prior source-preserving
140-chunk migration remains in place, with the original uploaded bytes and other
documents intact. Only the local API worker was restarted to load the fix.

The new diagnostic oracle assesses **each numbered item** against independently
annotated assertions from the full PDF passage: suite/domain/construction quality
and methods for First; population, scale, and scientific/programming evaluation for
Second; both analysis dimensions and behavioral/cognitive relationships for Third.
It also checks exact native-PDF quotes, per-item citation alignment, continuation
pages, exactly three numbered contributions, and a separately stated evaluation gap.
Its adversarial cases include keyword-rich benchmark-component substitutions,
swapped and duplicate items, partial contributions, and four-item answers; accurate
paraphrases are accepted by the diagnostic rubric. These annotations are evaluation
fixtures, not production answer text or a general semantic entailment oracle.

Final follow-up verification: **6/6 actual dashboard submissions passed** (three of
each exact question), with the real HTTP API and local Qwen. Every independent per-item
assertion, full-PDF supporting fragment, exact quote/page check, numbering check, and
source-caption check passed. There were no warnings. `followup-verified.json` stores
all responses and item-level results. `followup-baseline-review.json` records the new
oracle rejecting both baseline answers and their precise `Unfaithful First` validation
failure. The actual fixed Qwen request/response is in
`followup-source-generation-wire.json`; its schema binds source-derived texts and refs,
and it passed without a retry. `followup-preservation.json` verifies original bytes,
hash, 140 chunks, and byte-for-byte equality of the other document/chunk/vector rows.

Final tests: **103 passed, 1 skipped** (optional Ragas dependency), **Ruff clean**.
Regressions include a wholly different water-sensor author list, correct-ref/wrong-prose
and partial-source failures, full-evidence retries with precise errors, honest fallback,
the legacy/conflicting-schema bypasses, and three distinct S8 quotes displayed under
one caption while an invalid later S8 quote is rejected. Browser control remains
unavailable; these are real Streamlit AppTest dashboard executions, not a claim of
pixel-based browser verification. API and dashboard remain local. Nothing was pushed
or deployed.
