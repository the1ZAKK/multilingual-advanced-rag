import hashlib
import json
import logging
import os
import re
import threading
from time import perf_counter

import faiss
import numpy as np
import pymupdf

from rag.config import Settings
from rag.embeddings.backend import Embedder, make_embedder
from rag.generation.ollama import GenerationUnavailable, OllamaGenerator, extractive
from rag.ingestion.pdf import parse_pdf
from rag.ingestion.tables import extract_result_tables
from rag.ingestion.text import normalize
from rag.models import ChatRequest, ChatResponse, DocumentInfo, Hit, SearchRequest, SearchResponse
from rag.query import (
    asks_problem,
    contribution_question,
    models_results_question,
    overview_question,
    resolve_language,
    verification_question,
)
from rag.reranking.backend import Reranker
from rag.retrieval.index import SearchIndex, rrf
from rag.storage.sqlite import Store

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        self.settings = settings
        self.embedder = embedder or make_embedder(settings)
        self.lock = threading.RLock()
        self.store = Store(settings.data_dir, self.embedder.key)
        self.index = SearchIndex(*self.store.corpus(), bm25_casefold=settings.bm25_casefold)
        self.reranker = Reranker(settings)
        self.generator = OllamaGenerator(settings)
        self._table_cache = {}
        self._page_cache = {}
        self._restore_cache()

    def _fingerprint(self):
        digest = hashlib.sha256(self.embedder.key.encode())
        for chunk in self.index.chunks:
            digest.update(chunk.model_dump_json().encode())
        digest.update(self.index.vectors.tobytes())
        return digest.hexdigest()

    def _restore_cache(self):
        path = self.settings.data_dir / "vectors.faiss"
        try:
            manifest = json.loads((self.settings.data_dir / "vectors.json").read_text())
            if (
                manifest["corpus"] == self._fingerprint()
                and manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                index = faiss.read_index(str(path))
                if (
                    index.ntotal == len(self.index.chunks)
                    and index.d == self.index.vectors.shape[1]
                ):
                    self.index.faiss = index
                    return
        except (OSError, ValueError, KeyError, RuntimeError):
            pass
        self._save_cache()

    def _save_cache(self):
        # DB commit is authoritative. A failed cache write cannot invalidate ingestion.
        directory = self.settings.data_dir
        try:
            tmp = directory / "vectors.faiss.tmp"
            faiss.write_index(self.index.faiss, str(tmp))
            manifest = {
                "corpus": self._fingerprint(),
                "sha256": hashlib.sha256(tmp.read_bytes()).hexdigest(),
            }
            os.replace(tmp, directory / "vectors.faiss")
            temp_meta = directory / "vectors.json.tmp"
            temp_meta.write_text(json.dumps(manifest))
            os.replace(temp_meta, directory / "vectors.json")
        except OSError:
            logger.warning("FAISS cache write failed; SQLite remains intact", exc_info=True)

    def documents(self):
        with self.lock:
            return self.store.documents()

    def ingest(self, data: bytes, filename: str) -> DocumentInfo:
        info, chunks = parse_pdf(data, filename, self.settings)
        with self.lock:
            existing = next(
                (d for d in self.store.documents() if d.document_id == info.document_id), None
            )
            if existing:
                return existing.model_copy(update={"duplicate": True})
            vectors = self.embedder.encode([c.normalized for c in chunks])
            if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
                raise RuntimeError("Embedding backend returned invalid vectors")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = np.asarray(vectors / np.maximum(norms, 1e-12), dtype=np.float32)
            all_vectors = np.vstack([self.index.vectors, vectors]) if self.index.chunks else vectors
            combined = self.index.chunks + chunks
            order = sorted(range(len(combined)), key=lambda i: combined[i].chunk_id)
            candidate = SearchIndex(
                [combined[i] for i in order],
                all_vectors[order],
                bm25_casefold=self.settings.bm25_casefold,
            )
            self.store.add(info, chunks, vectors, data)
            # Stable DB ordering is also the FAISS row mapping after restart.
            self.index = candidate
            self._save_cache()
            logger.info(
                "document_indexed", extra={"document_id": info.document_id, "chunks": len(chunks)}
            )
            return info

    def reindex_document(self, document_id: str, data: bytes | None = None) -> DocumentInfo:
        """Explicit atomic per-document migration; keep identity, name and other documents."""
        with self.lock:
            existing = next(
                (d for d in self.store.documents() if d.document_id == document_id), None
            )
            if existing is None:
                raise ValueError("Unknown document")
            data = data if data is not None else self.store.source(document_id)
            if data is None:
                raise ValueError("Legacy upload has no retained PDF; supply the original PDF")
            if hashlib.sha256(data).hexdigest() != document_id:
                raise ValueError("PDF hash does not match the uploaded document; migration refused")
            info, chunks = parse_pdf(data, existing.name, self.settings)
            vectors = self.embedder.encode([c.normalized for c in chunks])
            if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
                raise RuntimeError("Embedding backend returned invalid vectors")
            vectors = np.asarray(
                vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12),
                dtype=np.float32,
            )
            keep = [i for i, c in enumerate(self.index.chunks) if c.document_id != document_id]
            combined = [self.index.chunks[i] for i in keep] + chunks
            all_vectors = np.vstack([self.index.vectors[keep], vectors]) if keep else vectors
            order = sorted(range(len(combined)), key=lambda i: combined[i].chunk_id)
            candidate = SearchIndex(
                [combined[i] for i in order],
                all_vectors[order],
                bm25_casefold=self.settings.bm25_casefold,
            )
            self.store.add(info, chunks, vectors, data, replace=True)
            self.index = candidate
            self._table_cache.pop(document_id, None)
            self._page_cache.pop(document_id, None)
            self._save_cache()
            logger.info("document_reindexed", extra={"document_id": document_id})
            return info

    def delete(self, document_id: str) -> bool:
        with self.lock:
            kept = [i for i, c in enumerate(self.index.chunks) if c.document_id != document_id]
            candidate = SearchIndex(
                [self.index.chunks[i] for i in kept],
                self.index.vectors[kept],
                bm25_casefold=self.settings.bm25_casefold,
            )
            if not self.store.delete(document_id):
                return False
            self._table_cache.pop(document_id, None)
            self._page_cache.pop(document_id, None)
            self.index = candidate
            self._save_cache()
            return True

    def search(self, request: SearchRequest) -> SearchResponse:
        start = perf_counter()
        query = normalize(request.query)
        if not query:
            raise ValueError("Query must contain non-whitespace text")
        rerank_ms = 0
        with self.lock:
            allowed = set(request.document_ids) if request.document_ids is not None else None
            known = {d.document_id for d in self.store.documents()}
            if allowed is not None and not allowed <= known:
                raise ValueError("Unknown document identifier in retrieval scope")
            depth = max(self.settings.retrieval_depth, request.top_k)
            dense = []
            if request.mode != "bm25" and self.index.chunks:
                dense = self.index.dense(self.embedder.encode([query], query=True), depth, allowed)
            sparse = (
                self.index.sparse(request.query, depth, allowed) if request.mode != "dense" else []
            )
            hits = (
                dense
                if request.mode == "dense"
                else sparse
                if request.mode == "bm25"
                else rrf([dense, sparse], self.settings.rrf_k)
            )
            if request.mode == "hybrid-rerank":
                if self.settings.reranker_backend == "none":
                    raise ValueError(
                        "Reranking disabled. Configure lexical or cross-encoder backend."
                    )
                hits, rerank_ms = self.reranker.rerank(query, hits[:depth])
        return SearchResponse(
            hits=hits[: request.top_k],
            latency_ms=(perf_counter() - start) * 1000,
            rerank_latency_ms=rerank_ms,
            embedding_backend=self.settings.embedding_backend,
            reranker_backend=self.settings.reranker_backend
            if request.mode == "hybrid-rerank"
            else "none",
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        start = perf_counter()
        documents = (
            set(request.document_ids)
            if request.document_ids is not None
            else self.named_documents(request.query)
        )
        retrieval_request = (
            request.model_copy(update={"document_ids": sorted(documents)}) if documents else request
        )
        if (
            contribution_question(request.query)
            or models_results_question(request.query)
            or verification_question(request.query)
        ):
            # Inspect the candidate pool before intent-aware evidence selection;
            # a reranker's top five can otherwise discard the authors' section.
            retrieval_request = retrieval_request.model_copy(
                update={"top_k": max(request.top_k, min(self.settings.retrieval_depth, 50))}
            )
        hits = self.search(retrieval_request).hits
        language = resolve_language(request.query, request.language)
        if documents:
            hits = [h for h in hits if h.chunk.document_id in documents]
        supported = [
            h
            for h in hits
            if (h.bm25_score or 0) > 0
            or (h.dense_score is not None and h.dense_score >= self.settings.min_dense_score)
        ]
        evidence = self.select_evidence(supported, request.query, documents)
        self.bind_native_pages(evidence)
        warning = None
        if not evidence:
            answer = (
                "لا توجد أدلة كافية في الوثائق للإجابة."
                if language == "ar"
                else "Insufficient evidence in the indexed documents to answer."
            )
            citations, mode = [], "insufficient-evidence"
        elif request.use_llm:
            try:
                generated = None
                if models_results_question(request.query):
                    from rag.generation.table_answer import table_answer

                    evidence = self.table_evidence(evidence, documents, query=request.query)
                    generated = table_answer(evidence, language)
                    if generated is None:
                        logger.warning(
                            "table_answer_unavailable: no unambiguous source-bound table relations"
                        )
                        raise GenerationUnavailable(
                            "No unambiguous source-bound numerical table relationships; showing evidence excerpts"
                        )
                if generated is None:
                    generated = self.generator.generate(request.query, language, evidence)
                answer, citations = generated
                mode = getattr(generated, "mode", "ollama")
            except GenerationUnavailable as exc:
                answer, citations = extractive(evidence, language)
                warning, mode = str(exc), "extractive"
        else:
            answer, citations = extractive(evidence, language)
            mode = "extractive"
        if (
            mode == "extractive"
            and language == "ar"
            and any(resolve_language(h.chunk.text) != "ar" for h in evidence)
        ):
            notice = (
                "No verified Arabic answer was produced; the excerpts retain their source language."
            )
            warning = f"{warning} {notice}" if warning else notice
        return ChatResponse(
            answer=answer,
            citations=citations,
            evidence=evidence,
            generation_mode=mode,
            warning=warning,
            latency_ms=(perf_counter() - start) * 1000,
        )

    def select_evidence(self, hits, query="", document_ids=None):
        """Budget ranked seeds with adjacent same-section chunks, across page breaks.

        A neighbor is context, not an independent retrieval-score match. Keep its
        real chunk identity/page and don't expand across a section boundary.
        """
        with self.lock:
            if overview_question(query):
                named = document_ids or self.named_documents(query)
                scope = named or ({hits[0].chunk.document_id} if hits else set())
                abstracts = [
                    c
                    for c in self.index.chunks
                    if c.document_id in scope and c.section == "Abstract"
                ]
                if not abstracts and re.search(r"\babstract\b|ملخص", query, re.IGNORECASE):
                    # Some scientific papers have no printed Abstract heading.
                    # An explicit abstract request may anchor first-page source
                    # chunks, but does not relabel them as abstract or rewrite text.
                    abstracts = [
                        c
                        for c in self.index.chunks
                        if c.document_id in scope and c.page == 1 and c.section is None
                    ]
                if abstracts:
                    # The source's abstract is an explicit overview, unlike an
                    # appendix question that merely repeats the query vocabulary.
                    anchors = [
                        Hit(chunk=c, score=0)
                        for c in sorted(abstracts, key=lambda c: (c.page, c.start))[:3]
                    ]
                    anchor_ids = {h.chunk.chunk_id for h in anchors}
                    hits = anchors + [h for h in hits if h.chunk.chunk_id not in anchor_ids]
            if contribution_question(query):
                anchors = [h for h in hits if h.chunk.section == "Contributions"]
                named = document_ids or self.named_documents(query)
                if named:
                    # Explicit document identity + author section is stronger than
                    # multilingual lexical overlap. This is source selection, not
                    # a paper-specific answer or an entailment decision.
                    anchors = [
                        Hit(chunk=c, score=0)
                        for c in self.index.chunks
                        if c.document_id in named and c.section == "Contributions"
                    ]
                if anchors:
                    docs = {h.chunk.document_id for h in anchors}
                    # The explicit authors' statement is stronger evidence for this
                    # intent than incidental word matches in appendix prompts.
                    extra = [
                        h
                        for h in hits
                        if h.chunk.document_id in docs
                        and h.chunk.section
                        and (h.chunk.section == "Abstract" or "Introduction" in h.chunk.section)
                    ]
                    abstracts = [h for h in extra if h.chunk.section == "Abstract"]
                    extra = abstracts[:1] or extra[:1]
                    hits = anchors + extra if asks_problem(query) else anchors
            corpus = sorted(self.index.chunks, key=lambda c: (c.document_id, c.page, c.start))
            positions = {c.chunk_id: i for i, c in enumerate(corpus)}
            selected, seen, budget = [], set(), self.settings.max_context_chars
            ranked = {h.chunk.chunk_id: h for h in hits}
            for hit in hits:
                pos = positions.get(hit.chunk.chunk_id)
                neighbors = []
                if pos is not None and hit.chunk.section is not None:
                    for i in (pos - 1, pos, pos + 1):
                        if 0 <= i < len(corpus):
                            c = corpus[i]
                            if (
                                c.document_id == hit.chunk.document_id
                                and c.section == hit.chunk.section
                                and abs(c.page - hit.chunk.page) <= 1
                            ):
                                neighbors.append(
                                    ranked.get(c.chunk_id)
                                    or hit.model_copy(
                                        update={
                                            "chunk": c,
                                            "dense_score": None,
                                            "bm25_score": None,
                                            "rerank_score": None,
                                            "rrf_score": None,
                                        }
                                    )
                                )
                group = neighbors or [hit]
                # Ensure the ranked seed has priority if the full neighborhood doesn't fit.
                new = [h for h in group if h.chunk.chunk_id not in seen]
                if sum(len(h.chunk.text) for h in new) > budget:
                    new = [hit] if hit.chunk.chunk_id not in seen else []
                for candidate in new:
                    if len(candidate.chunk.text) <= budget:
                        selected.append(candidate)
                        seen.add(candidate.chunk.chunk_id)
                        budget -= len(candidate.chunk.text)
            return selected

    def named_documents(self, query: str) -> set[str]:
        """Resolve an explicit distinctive filename identifier, without guessing a paper.

        Short acronym identifiers and long title phrases are stable uploaded data.
        Common title words alone cannot scope retrieval; ambiguous matches stay open.
        """
        query_tokens = set(re.findall(r"[\w]+", query.casefold()))
        matches = set()
        stop = {
            "paper",
            "pdf",
            "the",
            "and",
            "of",
            "for",
            "with",
            "from",
            "arabic",
            "english",
            "model",
            "models",
        }
        for document in self.documents():
            stem = os.path.splitext(document.name)[0]
            printed_tokens = re.findall(r"[\w]+", stem)
            tokens = [token.casefold() for token in printed_tokens]
            acronym = {
                t for t in tokens if any(ch.isdigit() for ch in t) and any(ch.isalpha() for ch in t)
            }
            # A distinctive acronym/CamelCase identifier is uploaded metadata,
            # not a fixed paper-name lookup. Ordinary title-case words do not
            # scope retrieval, and ambiguous identifier matches remain open.
            acronym.update(
                token.casefold()
                for token in printed_tokens
                if len(token) >= 3
                and token.casefold() not in stop
                and (token.isupper() or re.search(r"[A-Z].*[A-Z]", token))
            )
            significant = [t for t in tokens if len(t) > 3 and t not in stop]
            explicit = bool(acronym & query_tokens) or (
                len(significant) >= 3 and len(set(significant) & query_tokens) >= 3
            )
            if explicit:
                matches.add(document.document_id)
        return matches if len(matches) == 1 else set()

    def source_pages(self, document_id):
        """Read retained original pages without changing the document/index."""
        with self.lock:
            if document_id not in self._page_cache:
                data = self.store.source(document_id)
                if data:
                    with pymupdf.open(stream=data, filetype="pdf") as source:
                        self._page_cache[document_id] = [
                            page.get_text(sort=False) for page in source
                        ]
                else:
                    self._page_cache[document_id] = []
            return self._page_cache[document_id]

    def bind_native_pages(self, evidence):
        for hit in evidence:
            pages = self.source_pages(hit.chunk.document_id)
            if pages and 0 < hit.chunk.page <= len(pages):
                hit.native_page_text = pages[hit.chunk.page - 1]

    def table_evidence(self, evidence, documents=None, query=""):
        """Read retained PDFs on demand; preserve cell geometry without reindexing.

        Table answers are rendered from source relations, so their native indexed
        passages need not fit the model's prompt budget. Nothing is sent to Qwen
        on this path. Ambiguous extracted tables carry diagnostics, never facts.
        """
        with self.lock:
            docs = documents or {h.chunk.document_id for h in evidence[:1]}
            selected = []
            for document in sorted(docs):
                if document not in self._table_cache:
                    data = self.store.source(document)
                    self._table_cache[document] = extract_result_tables(data) if data else []
                    if data:
                        with pymupdf.open(stream=data, filetype="pdf") as source:
                            self._page_cache[document] = [
                                page.get_text(sort=False) for page in source
                            ]
                    else:
                        self._page_cache[document] = []
                tables = [t for t in self._table_cache[document] if t.get("status") == "ok"]
                from rag.retrieval.table_selection import select_tables

                tables = select_tables(tables, query)
                pages = {t["page"] for t in tables}
                method_chunks = []
                for c in self.index.chunks:
                    if c.document_id != document:
                        continue
                    prose = re.sub(r"-\s*\n", "", c.text)
                    prose = " ".join(prose.split())
                    stem = re.search(r"STEM\s+benchmarks.{0,400}\baccuracy\b", prose, re.I)
                    code = re.search(
                        r"(?:for\s+)?code\s+benchmarks[,\s]+pass\s*@\s*1\s+evaluation", prose, re.I
                    )
                    if stem:
                        method_chunks.append((c, "accuracy", "STEM"))
                    if code:
                        method_chunks.append((c, "pass@1", "code"))
                method_ids = {c.chunk_id for c, _, _ in method_chunks}
                chunks = sorted(
                    (
                        c
                        for c in self.index.chunks
                        if c.document_id == document
                        and (c.page in pages or c.chunk_id in method_ids)
                    ),
                    key=lambda c: (c.page, c.start),
                )
                for chunk in chunks:
                    selected.append(Hit(chunk=chunk, score=0))
                if chunks:
                    labels = {h.chunk.chunk_id: f"S{i}" for i, h in enumerate(selected, 1)}
                    enriched = []
                    for table in tables:
                        paths = " ".join(" ".join(c["path"]) for c in table["columns"])
                        # A declared benchmark metric can bind matching table
                        # headers; unknown metrics stay unknown. This does not
                        # use answer text, paper identity, table numbers or values.
                        scope = (
                            "code"
                            if re.search(r"HumanEval|MBPP", paths, re.I)
                            else (
                                "STEM"
                                if re.search(r"MCQ", paths, re.I)
                                and re.search(r"Completion", paths, re.I)
                                else None
                            )
                        )
                        native_pages = self._page_cache.get(document, [])
                        bindings = [
                            {
                                "metric": metric,
                                "scope": kind,
                                "label": labels[c.chunk_id],
                                "quote": c.text,
                                "native_page_text": native_pages[c.page - 1]
                                if native_pages
                                else None,
                            }
                            for c, metric, kind in method_chunks
                            if kind == scope
                        ]
                        enriched.append(
                            {
                                **table,
                                "document_id": document,
                                "metric_bindings": bindings,
                                "native_page_text": native_pages[table["page"] - 1]
                                if native_pages
                                else None,
                            }
                        )
                    selected[-len(chunks)].tables = enriched
            self.bind_native_pages(selected)
            return selected or evidence
