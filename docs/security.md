# Security and privacy

Bind to loopback for local use. The API has no authentication or tenancy isolation; do not
expose it to untrusted networks. A hosted deployment requires authentication, authorization,
reverse-proxy body limits, TLS, rate limits and abuse controls. Configure request limits
at the gateway as well as per-file limits in the application: the multipart parser may spool
a request before application-level file validation, including requests without Content-Length.
The dashboard is an API client and does not provide an access-control boundary.

PDFs are checked by extension, signature, byte size, encryption status and page count.
Extraction runs in-process, not in a sandboxed parser service. Malicious or highly complex
PDFs can consume CPU/memory; use a resource-isolated worker before accepting public uploads.
Raw filenames are reduced to basenames, never used for storage paths. New uploads and
explicitly migrated legacy uploads retain original PDF bytes in SQLite for verified
reindexing. SQLite also contains extracted passages and vectors; protect its directory.
Deletion removes document/chunk/source-PDF records and rebuilds FAISS/BM25, but SQLite freelist space,
backups and OS snapshots can retain prior bytes. Secure erasure is not implemented.

No API keys, model weights, corpus indexes, uploaded personal documents or environment values
are committed. Model downloads use verified TLS and standard Hugging Face loading with
`trust_remote_code=False`. Pin model revisions for experiments and inspect licenses. The
FAISS cache is checksum-checked against its manifest and corpus before loading. This detects
accidental corruption, not malicious changes by someone with write access to the index
folder; only load locally generated indexes from trusted storage. BM25 is rebuilt without
pickle deserialization. SQLite is the authoritative recovery source.

Prompt layers separate system instructions from JSON-encoded question/evidence. Documents
are explicitly marked untrusted. Generated citations must use assigned source labels and
exact quotes present in those passages. Unknown labels, invented quotes and mismatched
markers are rejected; chat falls back to source excerpts. These checks prove provenance,
**not entailment**. A valid quotation can still fail to support a claim. Prompt injection is
mitigated, not solved. Human review remains necessary for legal/scientific conclusions.
Explicit threefold contribution answers therefore require the complete, dynamically
reconstructed author statements as well as their exact supporting references. Those
source statements are labeled separately from model paraphrases; valid source IDs
alone cannot validate altered contribution prose. A raw legacy answer cannot bypass
this per-item path. Dashboard citation grouping never removes distinct quotes before
validation.
Generator output is escaped in the UI. Local Ollama requests do not use environment proxies;
all generation remains on the configured local server unless the user changes that address.

Default evidence thresholds are heuristics, not calibrated confidence. A shared term can
admit weak evidence. Tune thresholds on held-out annotations and audit refusal behavior.
No content/query logging is enabled; logs contain event names, document hashes and error
stack traces. Ollama and other local services may have separate logging policies.
The opt-in diagnosis scripts write full questions, passages and model wire responses
under the locally ignored `reports/` directory. They do not enable production content
logging. Protect or remove those reports according to the source document's privacy needs.
