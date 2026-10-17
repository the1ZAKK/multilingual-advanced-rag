"""Functional probe with a synthetic PDF; deletes only the document it created."""

import argparse
import uuid

import httpx
import pymupdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    document_id = None
    with httpx.Client(timeout=120, trust_env=False) as client:
        response = client.get(base + "/health")
        response.raise_for_status()
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 72, 523, 700),
            "Hybrid retrieval combines FAISS vectors with BM25 keywords. " + str(uuid.uuid4()),
        )
        try:
            response = client.post(
                base + "/documents/upload",
                files={"files": ("smoke.pdf", doc.tobytes(), "application/pdf")},
            )
            response.raise_for_status()
            info = response.json()["documents"][0]
            if info["duplicate"]:
                raise RuntimeError(
                    "Unexpected collision; refusing to delete a pre-existing document"
                )
            document_id = info["document_id"]
            for mode in ["dense", "bm25", "hybrid"]:
                response = client.post(
                    base + "/search", json={"query": "BM25 keywords", "mode": mode}
                )
                response.raise_for_status()
                assert any(
                    h["chunk"]["document_id"] == document_id for h in response.json()["hits"]
                )
            response = client.post(
                base + "/chat", json={"query": "BM25 keywords", "use_llm": False}
            )
            response.raise_for_status()
            assert response.json()["citations"]
            print("Functional health/upload/retrieval/cited-chat probe passed")
        finally:
            doc.close()
            if document_id:
                client.delete(base + "/documents/" + document_id).raise_for_status()


if __name__ == "__main__":
    main()
