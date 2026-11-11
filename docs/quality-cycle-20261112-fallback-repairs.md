# Four-fallback repair cycle — 12 November 2026

Recovered from `7def51e` on `arabic-quality-candidate`; all prior changes and attempts
were preserved. The semantic audit's short evidence explanation already precedes
its verdict in the saved source. Every explicit verdict flag remains required.

## Hosted recovery and the review restriction

The existing hosted run [43086599560](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/43086599560)
at `d2e5b893087c08cea4fdd5cd59bc08e6aae3e14a`, attempt 1, finished with
failure at 2026-11-11 21:01:39 UTC (12 November 05:01:39 China time). Its model
evaluation, compact export, identity checks and packet creation completed. Artifact
creation failed after five requests received HTML instead of JSON. The complete
model-job log was recovered and preserved locally; the Windows acceptance archive
was recovered and its published digest verified. Only that Windows artifact is
listed and the expected result branch returns 404. The model answers, packet and
wire recording cannot be recovered from available artifacts or console text.

The [recovery summary](validation-results/hosted-recovery-43086599560-summary.json)
records exact identity, failure evidence and console modes/latencies: seven excerpt
fallbacks and one generated English task/evaluation answer. Exact answers were not
reconstructed. New model runs began only after that recovery limit was established.

Auto-review rejected public upload of diagnostic-archive commit `1b00f7d` because
it would disclose model outputs, traces, logs and source-related data; the initial
rejection also described the remote as unverified. Verifying repository identity
did not override the disclosure decision. That action remains blocked and was not
retried. Both original archive files remain local and byte-identical to the saved
recovery commit. This packet uses fresh permitted final-answer exports; it contains
no restricted archive contents, raw logs, wire responses or rejected proposals.

## Causes and repairs

- Arabic comparison: protect source-activated provider, prompting and metric nouns
  without changing approaches into models or providers into suppliers. Reverse
  bindings inspect only the actual Arabic candidate. For an unambiguous native
  accuracy comparison, request its two condition/value phrases and reported ratio;
  unrelated cost ratios and invented bounds remain invalid.
- Arabic language/format: request qualified attribute phrases, avoiding a redundant
  existence assertion. The guard still rejects an added existence claim.
- English task/evaluation counts: replace runaway unbounded count qualifiers with
  the finite vocabulary already recognized by numerical validation. Filter removed
  quote IDs after overlap deduplication and require the unambiguous native count
  references in the provider response; missing citations are never appended.
- Arabic task/evaluation counts: generate two qualified count/entity phrases without
  redundant assessment verbs. Both counts and their distinct entities remain required.

The `db564bb` comparison gate improved the Arabic target, but regressed its English
control: the provider's schema converter emitted a literal tab inside JSON. Strict
parsing refused it. `c620cfd` uses canonical plain spaces in that comparison format;
it does not sanitize malformed JSON or relax parsing. Its new negative transport
regression requires refusal before semantic judgment.

The subsequent `c620cfd` gate returned valid comparison JSON but the judge
incorrectly demanded counts from other answer slots within the comparison claim.
The negative completeness verdict was enforced. `e72ae99` clarifies that the full
question provides entity/reference context while the unchanged requested_facet
identifies this claim’s scope. Both fields and all native premises are retained;
missing qualifications within the assigned facet and missing whole-answer slots
still fail. Three new HTTP-path regressions verify scope, negative completeness
refusal and an omitted count. No verdict is inferred from its explanation.

Numerical roles, source bounds, native citations, completeness, independent reverse
translation and semantic decisions remain necessary. No benchmark identities,
expected answers or fixed paper quantities were added to production code.

## Recorded outcomes

These are generation/fallback observations, not human source-review grades.

| Former fallback | Latest recorded mode | Actual tested source commit |
| --- | --- | --- |
| Arabic counts and accuracy comparison (`numbers-prompt-ar`) | `translated-answer` | `e72ae99e30f3823dc54fd9ec9f3a272761eb8f19` |
| Arabic language and question format (`mmlu-format-ar`) | `translated-answer` | `31f7d359cded69c04c0a0446b52b9611a517da99` |
| English task/evaluation distinction (`numbers-task-evaluations-en`) | `ollama` | `31f7d359cded69c04c0a0446b52b9611a517da99` |
| Arabic task/evaluation distinction (`numbers-task-evaluations-ar`) | `translated-answer` | `31f7d359cded69c04c0a0446b52b9611a517da99` |

| Actual source commit | Recording | Outcome |
| --- | --- | --- |
| `9e9703ac5cd4299f707834dc899fcdd99aad2eaf` | `e5c3c2da-ba4f-4e6e-9d33-c6736bd81465` | 6 generated, 2 explicit fallbacks |
| `31f7d359cded69c04c0a0446b52b9611a517da99` | `2690ed61-64ac-4de7-b1ec-f40d3141bc17` | 5 generated, 1 explicit fallbacks |
| `db564bb33783f839068eda366614f7a1c011be16` | `0f3fca89-2304-4ba0-9f37-e1d9f9b04696` | 1 generated, 1 explicit fallbacks |
| `c620cfdb78f45a104cde2ea7eced3faa0912ae95` | `8d5b9d7a-73b5-41bc-bdc1-7ae90c43e320` | 1 generated, 1 explicit fallbacks |
| `e72ae99e30f3823dc54fd9ec9f3a272761eb8f19` | `e2e49e7a-babf-4e43-94ee-fc053e385d1e` | 2 generated, 0 explicit fallbacks |

The final `e72ae99` two-case gate is the observed check on the audit-context
correction; its actual modes are included above rather than inferred from tests.

The eight-case view selects each question's latest recorded output across those
commits. It is not an eight-question run at the final commit. The original full
eight-case run, six-case affected/control run and final two-case comparison run
are exported separately. The intermediate `db564bb` and `c620cfd` recordings, including their
English control failures, remain local with exact identity and available final outputs.
All 20 attempts retain their original commit, model and settings.

All new runs use `qwen3-instruct-hf:4b`, Unsloth's Qwen3-4B-Instruct-2507 Q4_K_M,
revision `a06e946bb6b655725eafa393f4a9745d460374c9`, GGUF SHA-256
`3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`.
Ollama 0.40.5, 4 threads, 8192 context tokens, 6000 evidence characters, at most
four statements, temperature 0, 300-second request/600-second answer deadlines,
and extra experimental question-coverage auditing disabled match the recovered
4B settings. Learned multilingual MiniLM hybrid retrieval uses no reranker.
Runtime/model hashes were checked before each isolated sequential run; prompt
cache was disabled. The ordinary application model default was not changed.

549 local tests passed, 5 skipped; Ruff lint and formatting passed. The final
code's [Quality workflow](https://github.com/the1ZAKK/multilingual-advanced-rag/actions/runs/43114105765)
passed all four Windows/Linux Python 3.11/3.12 jobs and Docker. Negative regressions
cover count omissions/swaps, invalid references, wrong nouns/conditions, missing
or unrelated ratios, malformed JSON, and negative semantic decisions.

## Updated review packet

Download [RAG-PDF-Review-Kit-fallback-repairs-20261112.zip](review-kit/RAG-PDF-Review-Kit-fallback-repairs-20261112.zip),
extract it, and open `START_HERE.html`. The index links each question to its actual
recording. Each subfolder includes exact final answers, native quotations/pages,
the two original public PDFs, the frozen AI-curated reference guide and its own
ungraded UTF-8 CSV. All 16 review rows are `not_reviewed`; human reviews remain zero.

SHA-256: `50508d7133cb422ef807d0d56438f08dad18734deea01c402887e58c1b757f31`.
Verification checked all constituent manifests, exact answers/questions, model and
settings bindings, original PDF hashes, native quotations/pages, CSV fingerprints,
blank reviewer fields, safe paths, allowed file types and ZIP CRC. The
[aggregate record](validation-results/fallback-repair-cycle-20261112-summary.json)
retains every run's exact identity and outcomes, including intermediate failures.
The [packet verification record](validation-results/fallback-repair-packet-20261112.json)
records all three source-labelled parts and the latest per-question index.

Generated mode is an operational improvement, not proof of fluent or faithful
Arabic. AI-assisted inspection flags an extraneous `الكلمة` before the latest
Arabic accuracy comparison and awkward count agreement such as `210 مهام` in
the Arabic distinction. Those outputs need language revision; Arabic fluency
acceptance remains unmet. Exact recorded answers have not been edited. These
findings are separate from the blank independent-review CSVs and are not human
verdicts. Count/metric phrasing still warrants source-aware language review.
These are exposed development controls on English research PDFs. Independent
human review, native Arabic source-prose evaluation and actual user-laptop
acceptance remain open. The next step is independent review of the new packet's
answers and CSVs. No stable release or human-review pass is claimed.
