"""Isolated real-server acceptance on Windows/Linux; no user data or models changed."""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pymupdf


@contextmanager
def running(root, output, index, api_port, ui_port, backend="hash"):
    env = {
        **os.environ,
        "RAG_DATA_DIR": str(index),
        "RAG_EMBEDDING_BACKEND": backend,
        "RAG_RERANKER_BACKEND": "lexical" if backend == "hash" else "none",
        "RAG_API_URL": f"http://127.0.0.1:{api_port}",
        "PYTHONPATH": os.pathsep.join([str(root), str(root / "src")]),
    }
    processes, logs = [], []
    try:
        commands = [
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
                "--workers",
                "1",
            ],
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(root / "app/streamlit_app.py"),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(ui_port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
        ]
        for name, command in zip(["api", "dashboard"], commands, strict=True):
            log = (output / f"{name}-{time.time_ns()}.log").open("w", encoding="utf-8")
            logs.append(log)
            processes.append(
                subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            )
        with httpx.Client(trust_env=False, timeout=5) as client:
            for url in [
                f"http://127.0.0.1:{api_port}/health",
                f"http://127.0.0.1:{ui_port}/_stcore/health",
            ]:
                deadline = time.monotonic() + 90
                while True:
                    if any(p.poll() is not None for p in processes):
                        raise RuntimeError("Acceptance service exited; inspect its captured log")
                    try:
                        if client.get(url).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError("Acceptance service readiness timed out")
                    time.sleep(0.25)
        yield env
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for log in logs:
            log.close()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run(output, learned=False):
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    if output.exists():
        raise ValueError("Choose a new acceptance output directory; prior results are preserved")
    output.mkdir(parents=True)
    checks = []
    backend = "sentence-transformer" if learned else "hash"
    with tempfile.TemporaryDirectory(prefix="rag acceptance ") as temp:
        index = Path(temp) / "isolated index"
        api_port, ui_port = free_port(), free_port()
        while ui_port == api_port:
            ui_port = free_port()
        base = f"http://127.0.0.1:{api_port}"
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_textbox(
                pymupdf.Rect(50, 50, 550, 700),
                "Oriole retrieval combines dense vectors and BM25 keyword search.",
            )
            english = pdf.tobytes()
        arabic = (root / "tests/fixtures/arabic.pdf").read_bytes()
        originals = {
            hashlib.sha256(english).hexdigest(): english,
            hashlib.sha256(arabic).hexdigest(): arabic,
        }
        with running(root, output, index, api_port, ui_port, backend) as env:
            with httpx.Client(trust_env=False, timeout=180) as client:
                result = client.post(
                    base + "/documents/upload",
                    files=[
                        ("files", ("english.pdf", english, "application/pdf")),
                        ("files", ("arabic.pdf", arabic, "application/pdf")),
                    ],
                )
                result.raise_for_status()
                infos = result.json()["documents"]
                assert {d["document_id"] for d in infos} == set(originals)
                assert all(d["chunks"] > 0 and d["pages"] == 1 for d in infos)
                health = client.get(base + "/health").json()
                assert health["embedding_backend"] == backend
                if learned:
                    assert health["embedding_model_loaded"] is True
                checks.append("English and native-Arabic PDF upload/index identities")
                english_id = hashlib.sha256(english).hexdigest()
                for mode in ["dense", "bm25", "hybrid"]:
                    response = client.post(
                        base + "/search",
                        json={"query": "BM25", "mode": mode, "document_ids": [english_id]},
                    )
                    response.raise_for_status()
                    assert response.json()["hits"] and all(
                        h["chunk"]["document_id"] == english_id for h in response.json()["hits"]
                    )
                checks.append("Scoped dense/BM25/hybrid retrieval")
                for info in infos:
                    with pymupdf.open(stream=originals[info["document_id"]], filetype="pdf") as pdf:
                        native = pdf[0].get_text()
                    query = (
                        "BM25" if info["document_id"] == english_id else native.strip().split()[0]
                    )
                    response = client.post(
                        base + "/chat",
                        json={
                            "query": query,
                            "document_ids": [info["document_id"]],
                            "use_llm": False,
                        },
                    )
                    response.raise_for_status()
                    answer = response.json()
                    assert answer["generation_mode"] == "extractive" and answer["citations"]
                    assert all(
                        c["document_id"] == info["document_id"]
                        and c["page"] == 1
                        and c["quote"] in native
                        for c in answer["citations"]
                    )
                checks.append("English/Arabic cited chat with exact original-page quotes")
            # AppTest runs actual UI code against these real servers, not API mocks.
            previous = {key: os.environ.get(key) for key in ["RAG_API_URL"]}
            os.environ["RAG_API_URL"] = env["RAG_API_URL"]
            try:
                app = AppTest.from_file(str(root / "app/streamlit_app.py")).run(timeout=60)
                next(c for c in app.checkbox if c.label == "Generate with local Ollama").set_value(
                    False
                )
                app.chat_input[0].set_value("What combines BM25 and dense vectors?").run(timeout=60)
                assert not app.exception
                messages = app.session_state["messages"]
                assert messages[-1]["response"]["citations"]
                assert messages[-1]["response"]["generation_mode"] == "extractive"
                checks.append("Real Streamlit AppTest chat against live API")
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        previous_ports = {api_port, ui_port}
        api_port, ui_port = free_port(), free_port()
        while api_port in previous_ports or ui_port in previous_ports or api_port == ui_port:
            api_port, ui_port = free_port(), free_port()
        base = f"http://127.0.0.1:{api_port}"
        with running(root, output, index, api_port, ui_port, backend):
            with httpx.Client(trust_env=False, timeout=180) as client:
                docs = client.get(base + "/documents").json()
                docs = docs["documents"] if isinstance(docs, dict) else docs
                assert {d["document_id"] for d in docs} == set(originals)
                response = client.post(
                    base + "/chat",
                    json={"query": "BM25", "document_ids": [english_id], "use_llm": False},
                )
                response.raise_for_status()
                assert response.json()["citations"]
            checks.append("API/dashboard restart and persisted index/cited chat")
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        profile=backend + "; no model generation",
        checks=checks,
        passed=True,
        actual_user_laptop=False,
    )
    with (output / "acceptance.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["offline", "learned"], default="offline")
    args = parser.parse_args()
    run(args.output.resolve(), args.profile == "learned")


if __name__ == "__main__":
    main()
