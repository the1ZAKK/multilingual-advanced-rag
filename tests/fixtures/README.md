# Arabic fixture

`arabic.pdf` is a synthetic one-page Arabic fixture with text describing hybrid retrieval.
It embeds a subset of the system Noto Sans Arabic font (SIL Open Font License; fonts remain
under their original license). It is not a scanned document, scientific benchmark, or
screenshot. The fixture supports portable tests without OS-specific font dependencies.
Arabic native text order is tested explicitly; geometric sorting changes its word order.

`grounding-language-reference.json` is evaluation-only source data independently audited
from the uploaded 3LM PDF before the grounding/language fixes. It contains complete
author contribution spans and 672 numerical result tuples with model/variant/size,
benchmark, language, condition, metric, value, table and page. The semantic oracle
checks claim relations and completeness separately from exact citation provenance.
Production code must not import this fixture or use it as an answer template.

`cot-table2-page21.json` preserves native text from page21 of the actual local
*Chain-of-Thought Prompting Elicits Reasoning* paper, with independently reviewed
Table2 metadata: PaLM540B, GSM8K accuracy17.9standard versus56.9chain-of-thought.
This control prevents the 3LM contribution format/semantics from being applied to
every paper or question. Source hashes are included in both reference fixtures.
