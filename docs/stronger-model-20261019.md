# Stronger-model tests — 19 October 2026

The stronger model was downloaded, checksum-verified, imported and exercised on a
GitHub-hosted Linux runner. This resolves the previous inability to test it; it
does not resolve the cloud instance's CDN policy or certify answer quality.

## Reproducible runtime

- Official `Qwen/Qwen3-4B-GGUF`, revision `bc640142c66e1fdd12af0bd68f40445458f3869b`.
- `Qwen3-4B-Q4_K_M.gguf`, 2,497,280,256 bytes.
- SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
- Ollama 0.40.5; imported model `qwen3-hf:4b`.
- Learned multilingual MiniLM, hybrid retrieval, no cross-encoder, 6,000 context
  characters, four answer statements, 450-second total deadline for the first run.
- Source `7e0520a`; [completed run](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/39154135524).

The runner verifies the entire model file before import and every PDF hash and
frozen native-page reference before ingestion. Model generation never receives
reference answers or expected points.

## First run: answer quality remains inadequate

Nine questions covered four papers: two fresh development papers and two previously
exposed controls. The fresh papers contain English and bilingual Arabic/English
passages; they are not two fully Arabic-language papers. This is an AI-curated
development suite, not an independent scientific benchmark.

| Question | AI-assisted source review |
|---|---|
| Legal problem and source material | Partial: the source-material slot contains a figure caption |
| Legal current/future retrieval | Partial: correct initial exclusion, wrong future-work detail |
| Same legal question in Arabic | Fail: 450-second timeout and English-source fallback |
| Table curation and evaluation mechanism | Fail: statement-budget rejection and fallback |
| NPT and rank-four two-qutrit PPT results | Partial: substitutes a different PPT statement |
| NPT result in Arabic | Fail: semantic rejection and fallback |
| Culture coverage counts | Pass: 13 countries, 12 domains, 54 subtopics |
| Native Arabic country/region example | Fail: language rejection and fallback |
| Unsupported process reward model claim | Pass: evidence-limited refusal |

That is **2 pass, 3 partial, 4 fail**, reviewed by AI against original PDF text.
Independent human reviews remain **zero**. A successful workflow means the attempts
were recorded, not that the generated answers passed. Exact citation provenance
and Arabic script checks cannot establish completeness or semantic correctness.

[Compact runtime, answers, failure reasons and review notes](validation-results/qwen4-first-run.json).
Full ungraded human-review sheets and complete returned citations are retained in
`reports/qwen4-initial-review/`; no human verdicts have been invented. The 4B model
has **not** been selected as the application default.

## Targeted follow-up

An experimental question-coverage audit can be enabled with
`RAG_QUESTION_COVERAGE_AUDIT=true`; the application default is false because a
1.7B judge produced false-negative verdicts even while its reason said the claim
answered the question. The hosted follow-up procedures explicitly enable it.
This trial audit supplements claim support. Coverage is checked against the whole answer for that question part;
each sentence still has to be supported by its own citations. A supported but
irrelevant caption cannot pass merely because its quote is exact. This remains a
model-based audit and can err; focused rejection/repair tests do not substitute
for real-paper evaluation.

Explicit abstract requests prioritize scoped abstract chunks. For papers with no
printed Abstract heading, scoped, unheaded first-page chunks can be included as
source context. The system does not relabel them or invent a source section.

The original uploaded 3LM PDF has SHA-256
`661a0d336318fa5a8d80294b8b2a3bbf0dd79ffa1a0691cd05b54e9f5cdcd254`.
The official ACL publisher PDF is a different 22-page edition with SHA-256
`380f4f636c530e91ffa6f304a4e5e0640ef305b8389b1be4213307f06a98c284`.
Native reference checks differed, so the attempted original-paper hosted run
stopped before model execution. The original reference file and document identity
were preserved. Publisher-edition references require separate preflight and a
separately named suite; no checksum check was disabled.

## Windows evidence and remaining limits

Hosted Windows Server 2025 / Python 3.12 passed real PDF upload/indexing, learned
retrieval, cited English/Arabic excerpt chat, Streamlit AppTest against the live
API, and restart/persistence checks. The test uses an isolated temporary index and
terminates only its own processes. It does not exercise model generation on Windows
or establish that the user's physical laptop is working.

[Windows acceptance result](validation-results/windows-learned-acceptance.json).
The user's machine, independent human source review and complete reliable Arabic
answers remain acceptance gates. No research-answer readiness or full-project
completion is claimed.

## Repeat the recording procedure

In GitHub Actions, choose **Record model tests and Windows acceptance**, then
**Run workflow**. Select `full` for the original nine cases, `regression` for six
exposed failures, `arabic` for the two targeted Arabic cases, or `publisher` for the two original questions against the inspected
publisher edition. All are development suites. The optional model procedure is
manual so a documentation push does not repeat a multi-gigabyte model download.
Ordinary Windows/Linux unit checks and Docker builds still run on pushes.

For local evaluation, use a newly named output directory:

```bash
python -m scripts.fetch_review_sources --dataset data/evaluation/completion-review.json --output reports/new-papers
RAG_OLLAMA_MODEL=qwen3-hf:4b RAG_LLM_TOTAL_TIMEOUT=600 python -m scripts.validate_workflow --dataset data/evaluation/completion-review.json --sources reports/new-papers --output reports/new-model-run --profile learned --context-chars 6000 --max-statements 4
python -m scripts.review_packet --dataset data/evaluation/completion-review.json --run reports/new-model-run/answers.json --output reports/new-human-review
```

The configured Ollama model must already exist. `scripts.prepare_model` can verify
and import the pinned model where official downloads are allowed; it does not alter
the application's default generator. Each source/output path above must be new.
Do not fill human verdicts from model output, a mode flag, or software test results.

## Repairs motivated by follow-up failures

The publisher-edition run exposed conference furniture inside a page-spanning
author unit: a folio followed by a paired Proceedings/pages header and copyright
line. Display assembly now masks recognized publication metadata at page edges
without changing offsets, original chunks, native PDF text or quote IDs. Unknown
intervening text and bare numbers are retained. Every non-furniture character
still requires source coverage. Final citation validation checks **every required
source-bound fragment**, replacing a legacy check that incorrectly demanded one
contiguous quote containing the whole reordered cross-page sentence. Dropping a
required continuation quote still fails. This does not require reindexing user data.

An observed faithful ordinary Arabic translation also failed the conservative
word-set back-translation veto: “first edition” became “first version,” and “assess
the ability” became “measure the ability.” Two narrowly scoped phrase families
now canonicalize publication-version wording and the assessment verb immediately
before “the ability.” Negation, facts, numbers, identifiers, unrelated measurements
and exact native source quotations remain unchanged and checked. General synonyms
are not accepted automatically. Real-model follow-up remains necessary; these
normalizations alone do not prove Arabic answer quality.

## Paired six-case findings

The six exposed failures were run again on the verified 4B Q4_K_M model and the
existing cached 1.7B Q8_0 model. Both used learned MiniLM, hybrid retrieval, 6,000
context characters, four statements and a 600-second total deadline. The same
non-contribution code path and frozen expectations were used; the machines and
quantization differ, so elapsed times are not a fair hardware/model speed comparison.
The old code versions enabled the trial coverage gate unconditionally. Current
application defaults leave that gate off; hosted trial workflows explicitly enable it.

| Diagnostic run | Pass | Partial | Fail | Independent human reviews |
|---|---:|---:|---:|---:|
| 4B, six exposed failures | 0 | 2 | 4 | 0 |
| 1.7B, same six | 0 | 0 | 6 | 0 |

These are **AI-assisted judgments**, not an automatic benchmark score. Partial
means a correct subset, not approval of every remaining clause or citation.
The 4B legal-source answer still cites evaluation criteria instead of establishing
source origins. Its future-scope answer still substitutes a different future plan.
The trial coverage judge accepted those omissions and also falsely rejected a
supported Arabic statement. It is insufficient as a general completeness validator
and has not been enabled by default. The 1.7B run additionally produced malformed
JSON and false semantic/coverage rejections. No complete Arabic answers were
delivered in these paired runs.

Every returned quotation in all twelve attempts was checked against the exact
original PDF page. That confirms provenance, not claim entailment or completeness.
The abstract-priority change supplied the requested physics abstract, but a small
judge still rejected a supported validity statement and the Arabic physics answer
still selected irrelevant witness details. The updated back-translation phrases
were introduced **after** these runs and require their own targeted retest.

[Settings, raw rejection reasons, generated answers and AI review counts](validation-results/model-comparison-20261019.json).
The two repaired source-rendering questions cannot be extrapolated to general QA.

The example environment now matches the tested CPU profile: hybrid retrieval with
`RAG_RERANKER_BACKEND=none`. Cross-encoder reranking is still optional. Existing
user `.env` files are preserved by source updates. No generator model default was
changed. Native publication ligatures (`ﬁ`, `ﬀ`, etc.) are expanded only in display/
translation text, activating the existing terminology glossary; mathematical
symbols/subscripts and all exact citation quotes remain unchanged.

## Final targeted results

| Frozen regression | Final outcome | AI-assisted source review |
|---|---|---|
| 3LM problem/contributions, English | `source-contributions` | Pass: problem and all three full author units |
| 3LM contributions, Arabic | `translated-contributions` | Pass: all three substantive contributions and qualifiers |
| Ordinary Arabic legal current/future scope | `extractive` | Fail: coverage judge rejected before translation |
| Ordinary Arabic NPT result | `extractive` | Fail: semantic judge rejected before translation |

The publisher questions passed on source `1b4d85f` in
[the real-model run](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/39212487984).
Both answers cite exact native publisher-PDF quotes on PDF pages 1–2 (printed
proceedings pages 42–43). The Arabic prose remains stylistically awkward, but the
requested facts, over-40 qualifier, scientific/programming domains and analytical
methods are retained. This validates that narrow source-rendering/translation
route, **not general Arabic QA**. Earlier publisher attempts failed: first at
legacy quote coverage, then by displaying conference furniture despite successful
mode flags. The final repair also handles blank lines between publisher blocks.

[Final publisher answers and citation/source review](validation-results/publisher-final-20261019.json).
The original uploaded 3LM PDF was additionally checked locally for complete
three-unit source assembly and exact native quotes after the changes. That check
used its original bytes and did not involve model generation or reindex existing data.

The ordinary Arabic retest used source `00a15a1` with the experimental coverage
audit explicitly enabled and a 600-second total deadline. Both attempts still
failed before the guarded ordinary-translation stage. Therefore the faithful
phrase normalization is unit-verified against the observed round trip but has
**not** been shown to produce a successful end-to-end answer for those questions.
[Exact modes, timing and rejection reasons](validation-results/arabic-final-20261019.json) ·
[recorded run](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/39211804667).
No default generator upgrade is justified by these results.

All source checks and semantic judgments above are AI-assisted. No independent
person or physical user laptop was available. The review packets contain blank
human verdicts, exact source identities, expected points and complete returned
answers/citations; they omit retrieved full-context evidence and include no PDFs,
models, indexes or settings. Human review must still inspect the original sources.
Embedding model identifiers were recorded, but their default upstream revision
was not explicitly pinned in these development runs; no publication-grade
reproducibility or independent benchmark accuracy is claimed.

The current manual workflow uses a 600-second total deadline and enables the trial
coverage gate explicitly. Selecting `full` repeats the original questions, not
the earlier 450-second code/protocol. Compare source SHA and recorded settings
when interpreting runs. Source updates and manifests preserve `.env`, uploaded
documents, indexes, model caches and unknown local edits; none of the model-test
outputs justify silently changing a user's installed model or settings.

## Engineering verification

The final local suite passed **403 tests**, with four optional tests skipped
because they require private Windows-path PDF/reference fixtures. Ruff lint and
format checks passed. The source archive was exercised against the original
uploaded source, the earlier published main, the previous validation branch and
main immediately before completion work: backups were valid, reapplication was
idempotent, three private-file sentinels were preserved for each baseline, and
unknown README edits were refused. This is updater verification on isolated test
copies, not a backup or update of the user's physical laptop.
