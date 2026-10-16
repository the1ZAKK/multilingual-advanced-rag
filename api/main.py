import json
import logging
import re
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from rag.config import Settings
from rag.evaluation.benchmark import EvaluationRequest, evaluate
from rag.ingestion.pdf import InvalidDocument
from rag.models import ChatRequest, ChatResponse, SearchRequest, SearchResponse
from rag.service import RAGService


class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(
            {
                "level": record.levelname,
                "event": record.getMessage(),
                "document_id": getattr(record, "document_id", None),
            },
            ensure_ascii=False,
        )


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger("rag").addHandler(handler)
logging.getLogger("rag").setLevel(logging.INFO)


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service or RAGService(config)
        yield

    app = FastAPI(title="Multilingual Advanced RAG", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_limits(request, call_next):
        # Content-Length is an early check, not the only check. Each file is read with a hard cap.
        value = request.headers.get("content-length")
        if value:
            try:
                size = int(value)
            except ValueError:
                return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
            if (
                size < 0
                or size > config.max_file_mb * 1024 * 1024 * config.max_upload_files + 1024 * 1024
            ):
                return JSONResponse({"detail": "Request exceeds upload limit"}, status_code=413)
        start = perf_counter()
        response = await call_next(request)
        response.headers["X-Response-Time-Ms"] = f"{(perf_counter() - start) * 1000:.2f}"
        return response

    @app.exception_handler(ValueError)
    async def invalid_input(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(RuntimeError)
    async def unavailable(request, exc):
        logging.getLogger("rag").error("pipeline_unavailable", exc_info=exc)
        return JSONResponse(
            {
                "detail": "Pipeline unavailable. Check server logs, model installation and configuration."
            },
            status_code=503,
        )

    @app.get("/health")
    def health():
        svc = app.state.service
        return {
            "status": "ok",
            "documents": len(svc.documents()),
            "embedding_backend": config.embedding_backend,
            "embedding_model": config.embedding_model,
            "embedding_model_loaded": getattr(svc.embedder, "_model", None) is not None,
            "reranker_backend": config.reranker_backend,
            "ollama": svc.generator.status(),
        }

    @app.post("/documents/upload")
    async def upload(files: list[UploadFile] = File(...)):
        if len(files) > config.max_upload_files:
            for file in files:
                await file.close()
            raise HTTPException(413, "Too many files")
        results, errors = [], []
        for file in files:
            try:
                data = await file.read(config.max_file_mb * 1024 * 1024 + 1)
                if len(data) > config.max_file_mb * 1024 * 1024:
                    raise InvalidDocument("File exceeds configured size limit")
                info = await run_in_threadpool(
                    app.state.service.ingest, data, file.filename or "upload.pdf"
                )
                results.append(info.model_dump())
            except InvalidDocument as exc:
                errors.append({"filename": file.filename, "error": str(exc)})
            finally:
                await file.close()
        # Explicit partial success: clients must inspect both arrays.
        return JSONResponse(
            {"documents": results, "errors": errors}, status_code=200 if results else 422
        )

    @app.get("/documents")
    def documents():
        return app.state.service.documents()

    @app.delete("/documents/{document_id}")
    def delete(document_id: str):
        if not re.fullmatch(r"[a-f0-9]{64}", document_id):
            raise HTTPException(422, "Invalid document identifier")
        if not app.state.service.delete(document_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": document_id}

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest):
        return app.state.service.search(request)

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        return app.state.service.chat(request)

    @app.post("/evaluate")
    def evaluation(request: EvaluationRequest):
        return evaluate(app.state.service, request)

    return app


app = create_app()
