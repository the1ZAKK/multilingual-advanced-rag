# Grounding and language diagnosis

**Later generalization check (2026-10-16):** [Additional-paper evaluation](resume-evaluation-20261016.md) exposes unresolved author-list, table-binding, Arabic explanation and semantic-judge failures. The successful 3LM runs below are historical case-specific results, not a general answer-quality claim. Current regression results are 310 passed, 1 skipped; see the later report for repairs and limitations.

This follow-up supersedes conclusions based only on `generation_mode`, source quote identity, or keyword coverage. The three exact queries are saved in `reports/grounding-language/queries.json`. The original summary and explicit-contributions questions are saved separately in `original-queries.json`.

## Independent reference

The uploaded **3LM: Bridging Arabic, STEM, and Code through Benchmarking** PDF was inspected before production changes. Page 1 ends during the first contribution; page 2 begins with its translation/human-verification continuation, followed by the second and third contributions. The three author-stated contributions are benchmark construction, evaluation of over 40 Arabic and multilingual models, and cross-task correlation/robustness analysis. Native STEM, synthetic STEM, and translated code are benchmark components within the first contribution.

The model and numerical reference was established from original table geometry, captions, methods, and rendered pages, independently of the index or generated answers. It contains 672 bound model/variant/size/column/value tuples from the four main performance tables. HumanEval(+), MBPP(+), and machine translation have benchmark/process roles. Qwen3-235B-A22B has both a curation role and evaluated result rows. Models are not excluded merely because they also curate data.

The independent inspection found no direct statement establishing introduction of an Arabic process reward model. This supports an evidence-local insufficiency answer, not a claim that retrieval proves absence everywhere in the paper. Source references and images are in `reports/grounding-language/source-reference.md`, `source-reference.json`, and `page*.png`.

## Failures and fixes

The six exact baseline dashboard runs all displayed `ollama` and all had valid citation provenance, yet all failed independent answer quality. The two Arabic runs additionally failed language following. An earlier diagnostic invocation corrupted Arabic text through PowerShell stdin; those two records were excluded and replaced by exact UTF-8 reproductions. They remain preserved as diagnostic artifacts.

English-only contribution-intent checks let the Arabic query enter generic generation. The previous schema also required English author wording and instructed the model to retain it. Shared Arabic/English intent handling now selects the actual contribution section and its neighboring continuation. Explicit document identifiers scope selection to the named uploaded paper. Requested language is resolved separately and enforced on answer prose, while original citation quotes retain their source language.

The model/results path formerly passed linear PDF tokens to Qwen without row/column relationships. The fix uses retained PDF word coordinates, merged model cells, parameter-size rows, hierarchical headers, captions, and direct metric declarations. Ambiguous layouts, captionless tables, and illustrative prompt/template tables yield no trusted result facts. Legitimate appendix result tables remain eligible. Complete printed model catalogs and representative rows chosen by column maxima are rendered from those relationships. This covers representative main performance results, not every numerical finding or unsupported table layout.

Positive introduction claims now require a direct source candidate and a separate semantic entailment audit. Related benchmarks, methods, appendix prompts, and mentions cannot establish model introduction. Where retrieved passages provide no direct support, the response explicitly explains insufficiency and distinguishes it from proof of whole-paper absence. Context citations are labeled as contextual references.

The direct-candidate gate is conservative and lexical. Synonym-heavy or cross-language positive evidence can fail closed before semantic auditing; the English positive-evidence regression does not establish recall for every language or phrasing. An insufficiency response describes retrieved support, never a whole-document absence finding.

The first attempted Arabic fix produced Arabic prose but changed code generation into classification, correlations into recommendations, and “over 40” into “around 40.” Qwen's own cross-language audit falsely accepted all three translations. These runs were rejected by independent source checks. Later token and batched-span attempts dropped source actions or coordination even with the right terms present. Their complete outputs and precise failures remain in the diagnostic artifacts.

Translation now binds source-activated general scientific terms and numeric relations in ordered source spans. Qwen translates the intervening prose; the renderer retains bound terms and conjunctions, then checks language, concepts and numerical bounds. Independent reverse translation receives only the actual Arabic candidate and a shared vocabulary mapping, never the original English source. A separate audit checks proposition roles and completeness against the full author statement. Neither terminology presence nor the audit's verdict proves final answer quality: the assembled sentences and saved dashboard outputs are independently checked against the original page-1/page-2 author list. No paper answer, model roster or result gold is embedded in production code.

Native PDF citation checks also caught whole-chunk quotes assembled from reordered reading blocks. Table citations now use unchanged contiguous substrings present in both the native PDF page and the indexed passage. Original quote/reference validation remains strict. The dashboard groups repeated source labels for display and preserves each distinct verified quote. Ordinary questions use paragraphs; enumeration is limited to requested lists and author contribution lists. Appendix prompts remain document data in generation, translation, and audit stages, and cannot spoof an authored contribution section.

## Reranking

The actual configured multilingual cross-encoder was evaluated alongside dense, BM25, hybrid, and an offline lexical comparison. For the Arabic question, hybrid placed the page-1 author statement at rank 11 and the page-2 continuation at rank 21; cross-encoder reranking moved them to 13 and 7. Neither top five contained the complete list. For models/results, reranking improved the model-definition passage and one numerical table, but the candidate pool did not contain all four main result tables. Reranking improves ranking within its candidate pool; it cannot recover absent table relations or establish entailment. Detailed ranks are preserved in `retrieval-comparison.json`.

## Preservation and migration

This follow-up performs no further reindexing. Table geometry and native page text are cached in memory from the retained original PDF; indexed chunks and vectors are unchanged. The earlier reading-order migration replaced only this document's chunks (148 to 140), atomically retained its document identity/name/source bytes, and left other documents and vectors intact. Both pre-migration SQLite backups remain available. Read-only table/source/backup fingerprints before and after this follow-up are recorded under `reports/grounding-language/preservation-*.json`.

The original PDF hash is `661a0d336318fa5a8d80294b8b2a3bbf0dd79ffa1a0691cd05b54e9f5cdcd254`. A separate Chain-of-Thought paper was tested in an isolated index; it was not uploaded into or substituted for the live corpus.

## Verification artifacts

The dashboard tests execute the actual Streamlit application with HTTP calls to the running local API and Ollama. Browser inventory was empty, and attempts to open the local dashboard in IAB and Chrome both returned "Browser is not available". These are AppTest dashboard-path checks, not visual browser verification. Saved responses report answer quality, language following, citation provenance, and generation mode separately. The independent control verifies PaLM 540B/GSM8K standard prompting 17.9 versus Chain-of-Thought prompting 56.9 on the actual second PDF, without requiring three contributions or a numbered answer.

The second-paper control tests table/entity/condition generalization. It does not establish Arabic translation fluency on every paper. The final dashboard runner additionally checks the actual rendered source captions in each assistant message for duplicates while retaining distinct quotes under one source.

All final runs passed independent language, semantic completeness, native-PDF/selected-chunk citation provenance, and actual displayed source-deduplication checks. Each query was repeated three times with `auto` and three times with the explicit requested language.

| Question | Runs | Semantic quality | Language | Citation validity | Generation mode |
|---|---:|---:|---:|---:|---|
| Exact Arabic authored contributions | 6 | 6/6 | 6/6 | 6/6 | `translated-contributions` |
| Exact models/results | 6 | 6/6 | 6/6 | 6/6 | `source-tables` |
| Exact PRM verification | 6 | 6/6 | 6/6 | 6/6 | `insufficient-evidence` |
| Original summary | 6 | 6/6 | 6/6 | 6/6 | `source-contributions` |
| Original explicit three-contributions list | 6 | 6/6 | 6/6 | 6/6 | `source-contributions` |

The numerical answer has 106 correctly bound claims per run, checked against the 672 independently annotated source tuples. Its complete printed model/size catalogs and representative main performance rows exclude benchmarks, curation processes, sizes and scores from model identities. The PRM answer explicitly names the queried object, requires direct evidence, labels citations as contextual, and avoids a whole-paper absence claim. Independent reviewers manually read every distinct answer variant in addition to the automated semantic regressions.

The final regression suite passed **306 tests, with 1 optional RAGAS test skipped**; Ruff checks were clean. Unit regressions cover missing author-list continuations, benchmark-versus-contribution distinctions, Arabic quantity and scientific-meaning drift, wrong model roles, swapped numerical relationships, unsupported positive claims, direct positive evidence, whole-paper absence overclaims, appendix prompts, strict native quotes, and displayed citation deduplication. Portable translations use different domains and source-derived counts; they do not embed a 3LM answer. The actual second-paper control passed all six isolated service runs with the final code.

Final counts are in `reports/grounding-language/final-summary.json`; actual dashboard outputs and independent checks are in `final-verified.json`, `final-verified-quality.json`, `original-verified.json`, and `original-verified-quality.json`. The separate manual audit is in `final-independent-review.md`, `final-independent-dashboard-audit.json`, and `original-independent-dashboard-audit.json`. Precise model/validation requests and failures are retained in `generation-wire.jsonl` and the API stderr logs. The final API remains running locally; no production files changed after the full test run.

The final read-only preservation comparison is equal for every corpus table (including chunk vectors and retained PDF bytes) and both backup hashes. This follow-up did not reindex or replace any document.

No push or deployment was performed.
