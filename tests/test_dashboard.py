from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import create_app


def test_dashboard_chat_and_comparison(service, settings, pdf_bytes, monkeypatch):
    service.ingest(pdf_bytes(), "sample.pdf")
    real_client = httpx.Client
    with TestClient(create_app(settings, service)) as backend:

        def handle(request):
            result = backend.request(
                request.method,
                request.url.path,
                content=request.content,
                headers={"content-type": request.headers.get("content-type", "application/json")},
            )

            return httpx.Response(
                result.status_code, content=result.content, headers=dict(result.headers)
            )

        monkeypatch.setattr(
            httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle))
        )
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app/streamlit_app.py"))
        app.run(timeout=15)
        assert not app.exception
        next(c for c in app.checkbox if c.label == "Generate with local Ollama").uncheck().run()
        app.chat_input[0].set_value("What is BM25 retrieval?").run(timeout=15)
        assert not app.exception
        assert len(app.chat_message) == 2
        assert "Source excerpts" in app.session_state["messages"][-1]["text"]
        app.text_input[0].set_value("BM25 retrieval")
        next(b for b in app.button if b.label == "Compare").click().run(timeout=15)
        assert not app.exception
        assert len(app.expander) >= 3


def test_dashboard_keeps_failed_chat_visible_after_rerun(monkeypatch):
    real_client = httpx.Client

    def handle(request):
        if request.url.path == "/chat":
            return httpx.Response(500, json={"detail": "backend test failure"})
        return httpx.Response(200, json=[])

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle))
    )
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app/streamlit_app.py")).run(timeout=15)
    app.chat_input[0].set_value("Question during backend failure").run(timeout=15)
    assert not app.exception
    messages = app.session_state["messages"]
    assert len(messages) == 2
    assert "backend test failure" in messages[-1]["text"]
    assert "response" not in messages[-1]
    assert messages[-1]["request"]["query"] == "Question during backend failure"
    assert any("backend test failure" in item.value for item in app.markdown)
