# Troubleshooting

- **Model download denied / offline:** permit `huggingface.co` and the specific redirect
  destinations required by the selected model (e.g. `cdn-lfs.huggingface.co`,
  `cdn-lfs-us-1.hf.co`, `cas-bridge.xethub.hf.co`, `cas-server.xethub.hf.co`). Do not disable TLS.
  Cache public models on a permitted machine and use a local directory, or choose the
  explicit hash backend with a separate data directory. BM25-only retrieval can work without
  embedding model loading on an existing corpus. Model availability is not implied by `/health`.
- **Read-only HOME cache:** set `UV_CACHE_DIR` and `HF_HOME` to writable project cache paths.
- **Backend 503:** inspect structured logs; confirm `.[models]` dependencies, cached model,
  memory and `RAG_DEVICE`. A failed ingestion does not partially commit a document.
- **Model identity changed:** use a new `RAG_DATA_DIR`; reupload PDFs. A model/revision switch
  is rejected for an existing corpus. There is no implicit migration/reembedding command.
- **Ollama unavailable:** use `ollama list`, verify the configured exact model tag, and
  test `GET /api/tags` on the configured URL. Install/pull Qwen locally. Chat remains extractive.
- **Scanned/empty PDF:** run OCR with an Arabic/English OCR tool before ingestion. Never
  infer missing text from images. Empty pages are reported. Broken font mappings and complex
  layouts can produce poor text even when nonempty; inspect passages before trusting results.
- **Arabic reading order:** the parser uses native text order for Arabic pages; PDF authoring
  tools differ. Column layouts and mixed bidi content need manual inspection. No heuristic
  reversal is applied to real Arabic content. Normalization affects search only, not quotes.
- **Reranking disabled:** select lexical/cross-encoder in configuration; requesting
  `hybrid-rerank` with `none` returns an explicit error.
- **No scores / unknown annotations:** fetch current chunk IDs from `/search`, annotate the
  current corpus and regenerate dataset IDs after chunk/parser changes.
- **SQLite / multiple workers:** run one API process per data directory. The in-process lock
  is not a distributed lock. Separate API processes sharing SQLite can have stale indexes.
- **Windows FAISS:** use 64-bit Python 3.11/3.12 and official `faiss-cpu` wheels; if no compatible
  wheel exists for your architecture, use Docker or WSL2. Do not install unofficial binaries.
- **Docker permission/runtime:** use Docker Desktop with Linux containers. Docker access in
  this cloud can differ from your laptop. Confirm host Ollama binding and firewall rules.
- **GPU:** choose official PyTorch CUDA wheels compatible with driver; verify
  `torch.cuda.is_available()`. The lock file deliberately selects CPU torch.

Corporate/cloud Docker builds may need the supplied proxy hostname to resolve in the build
network, plus a trusted CA bundle. The Dockerfile supports an optional BuildKit secret
`--secret id=corporate_ca,src=/path/to/trusted-ca-bundle.pem` and standard `--build-arg
HTTPS_PROXY --build-arg HTTP_PROXY --build-arg NO_PROXY` arguments. Resolve DNS using your
platform's supported network settings. The CA is mounted only for package installation, not
copied into the image. Never use trusted-host, no-verify flags or untrusted substitute CAs.
The cloud build was tested with the machine's existing system CA bundle.
