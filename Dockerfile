FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface
WORKDIR /app
RUN --mount=type=secret,id=corporate_ca \
    if [ -f /run/secrets/corporate_ca ]; then export SSL_CERT_FILE=/run/secrets/corporate_ca PIP_CERT=/run/secrets/corporate_ca; fi; \
    pip install --no-cache-dir uv==0.12.19
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY api ./api
COPY app ./app
COPY README.md LICENSE ./
RUN --mount=type=secret,id=corporate_ca \
    if [ -f /run/secrets/corporate_ca ]; then export SSL_CERT_FILE=/run/secrets/corporate_ca; fi; \
    uv sync --locked --extra models --no-dev && useradd --create-home rag && \
    chmod -R a+rX /app/src /app/api /app/app /app/README.md /app/LICENSE && \
    mkdir -p /app/data/index /app/.cache/huggingface && chown -R rag:rag /app/data /app/.cache
USER rag
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
