# PDF answer review kit

The latest continuation packet is
[RAG-PDF-Review-Kit-fallback-repairs-20261112.zip](RAG-PDF-Review-Kit-fallback-repairs-20261112.zip).
Choose the download button on its GitHub file page, extract it, and open
`START_HERE.html` in a browser.

It preserves separate recordings at `9e9703a` (eight cases), `31f7d35` (six
affected/control cases), and `e72ae99` (two comparison cases). Its index selects
the latest recorded answer per question across the actual commits; it does not
represent an eight-case run at the final commit. Exact model, runtime and settings
are included for each recording. Earlier failures remain in their own recordings.

Each folder contains exact final answers, source references, two original public
PDFs, a frozen AI-curated reference guide and an ungraded `review.csv`. All 16 rows
are `not_reviewed`; reviewer fields are blank and human reviews remain zero.
Raw diagnostic archives, wire traces, rejected proposals and logs are excluded.
See the [cycle report](../quality-cycle-20261112-fallback-repairs.md) for outcomes,
limitations and verification. Generated mode is not release or source-review approval.

Edit only reviewer, reviewer_type, verdict, and notes in review.csv.
Save as UTF-8 CSV and return the file in ChatGPT for review processing.

The [earlier four-case kit](RAG-PDF-Review-Kit.zip) is retained unchanged: run
39219463133 at `d66c62d`, before the abstract-routing correction. Its answers and
judgments must not be relabelled as results from the newer recordings.
