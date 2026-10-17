# Current implementation

```mermaid
flowchart TD
  PDF[Validated PDF] --> Layout[Reading order / sections / sentence windows]
  Layout --> Embed[Normalized multilingual embeddings]
  Embed --> DB[(SQLite metadata / original PDF / chunks / vectors)]
  DB --> Dense[FAISS with document ID selector]
  DB --> Sparse[Arabic-aware BM25 with document scope]
  Q[Question + language + selected documents] --> Dense
  Q --> Sparse
  Dense --> RRF[Reciprocal Rank Fusion]
  Sparse --> RRF
  RRF --> CE[Optional multilingual cross-encoder]
  CE --> Context[Budgeted source context / adjacent sections]
  Context --> Lists[Explicit source author lists]
  Context --> Tables[Geometry-bound table extraction / focused selection]
  Context --> Claims[Ordinary question generation]
  Lists --> Source[Source rendering or guarded translation]
  Tables --> Views[Bounded source table views]
  Claims --> Qwen[Local Ollama / Qwen]
  Qwen --> Verify[Quote provenance / per-sentence claims / language / semantic audit]
  Verify --> Pivot[Optional Arabic translation of verified English claims]
  Pivot --> Reverse[Independent reverse translation / number and identifier guards]
  Reverse --> Answer[Cited response labeled translated-answer]
  Verify --> Answer
  Reverse --> Fallback
  Verify --> Fallback[Explicit original-language excerpts on failure]
  Context --> Refuse[Evidence-local insufficiency for unsupported claims]
```

`src/rag/service.py` coordinates the independent components. Document filters run
inside FAISS and BM25 **before top-k truncation**, including reranked candidates.
Filename inference is only a convenience; explicit dashboard selection is more
reliable. Unknown scope IDs are errors, never an implicit search of all documents.

Ordinary overview questions can add the selected paper's Abstract as context.
Those anchors carry score zero: they are source context, not manufactured
retrieval matches. Standard `/search` metrics do not include this chat-only
context expansion, so retrieval metrics and final answer quality remain separate.

The source-list route recognizes explicit declarations and source-counted ordinals,
inline numbers, Roman numerals and bullets. It does not contain paper-specific gold
answers. It refuses incomplete/ambiguous source lists. Source-authored units stay
complete; translated units receive additional language/meaning checks.

Tables use PDF word/cell geometry, native source fragments and conservative role
checks. Nearby captions above a table take priority over the following table's
caption; accepted captions cannot be reused by another table on that page.
Unsupported layouts fail closed. Explicit table IDs, prompting conditions and
model names narrow the view. Requested weaker models remain in comparisons;
broad summaries have row/column/table caps and label omissions. These are source
views, not a claim that every model or result in a paper has been enumerated.

Ordinary generation uses a shorter task-specific prompt and bounded structured
statements. Multi-sentence generated items become individually cited/audited
claims, not a single terminal citation over uncited prose. Reference IDs and
exact original-PDF quotations remain immutable. Semantic judging sees source
metadata for reference resolution; metadata is not factual evidence. Requirement
flags are not verdict labels. False optional switches are omitted from judge
input, while the server still enforces supported/complete/direct verdict checks.

Generation, repair, translation and semantic audits share an overall deadline.
The default four-statement budget applies to ordinary generated prose, not to
source-counted author lists. Context size and time limits are configurable for
CPU memory/latency. Models can still fail or disagree: this is not a guaranteed
entailment oracle. No response-mode label is a correctness grade.

For ordinary Arabic questions with English-only evidence, generation first produces
English claims and audits them against their own sources. Only verified claims
are translated. Each Arabic sentence preserves the original refs and identifiers,
passes numeric-value and identifier/name checks, and receives an independent reverse translation
whose complete meaning is checked against the original claim and source premises.
The reverse translator never receives the original English claim. A conservative
content-word veto rejects changed round-trip meaning even if the judge approves;
it also rejects some faithful paraphrases. All calls share
the original time budget; failure still falls back. This adds latency and does
not guarantee question coverage or judge reliability.

Responses distinguish `ollama`, `translated-answer`, `source-contributions`, `translated-contributions`,
`source-tables`, `insufficient-evidence`, and `extractive`. Excerpts may retain the
source language when no verified translation exists; this is disclosed rather
than labeled a successful Arabic answer. The dashboard shows mode descriptions,
unique source captions, all distinct verified quotations, and retrieval metadata.
