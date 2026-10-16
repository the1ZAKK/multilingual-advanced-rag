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
        app.checkbox[0].uncheck().run()
        app.chat_input[0].set_value("What is BM25 retrieval?").run(timeout=15)
        assert not app.exception
        assert len(app.chat_message) == 2
        assert "Source excerpts" in app.session_state["messages"][-1]["text"]
        app.text_input[0].set_value("BM25 retrieval").run()
        next(b for b in app.button if b.label == "Compare").click().run(timeout=15)
        assert not app.exception
        assert len(app.expander) >= 3
