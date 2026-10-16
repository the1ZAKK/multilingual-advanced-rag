import html
import json
import os

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag.response_modes import MODE_DESCRIPTIONS

load_dotenv(override=False)

st.set_page_config(page_title="Multilingual Research Assistant", page_icon="📚", layout="wide")
BASE = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
st.title("Multilingual Research Assistant")
st.caption("Arabic & English · Hybrid retrieval · Verifiable page citations")


def api(method, path, **kwargs):
    try:
        with httpx.Client(timeout=180, trust_env=False) as client:
            response = client.request(method, BASE + path, **kwargs)
        if response.status_code >= 400:
            st.error(f"API {response.status_code}: {response.text}")
            return None
        return response.json()
    except httpx.HTTPError as exc:
        st.error(f"Backend unavailable: {exc}")
        return None


def rtl_text(text):
    # Escape all document and generated text; bidi is automatic and HTML is never trusted.
    st.markdown(
        '<div dir="auto" style="white-space:pre-wrap">' + html.escape(text) + "</div>",
        unsafe_allow_html=True,
    )


def passages(hits):
    for hit in hits:
        chunk = hit["chunk"]
        with st.expander(
            f"{chunk['document_name']} · page {chunk['page']} · score {hit['score']:.4f}"
        ):
            rtl_text(chunk["text"])
            st.caption(f"Chunk ID: {chunk['chunk_id']}")
            st.json({k: hit[k] for k in ["dense_score", "bm25_score", "rrf_score", "rerank_score"]})


def citation_sources(citations):
    # Group only for presentation, after API validation. Keep every distinct
    # verified quote; sharing a label never makes a different quote valid.
    sources = {}
    for citation in citations:
        identity = (
            citation["label"], citation["document_id"], citation["chunk_id"], citation["page"]
        )
        source = sources.setdefault(identity, {**citation, "quotes": []})
        if citation["quote"] not in source["quotes"]:
            source["quotes"].append(citation["quote"])
    return list(sources.values())


with st.sidebar:
    st.subheader("Runtime")
    if st.button("Check backend & Ollama"):
        status = api("GET", "/health")
        if status:
            st.json(status)
    language = st.selectbox("Answer language", ["auto", "ar", "en"])
    use_llm = st.checkbox("Generate with local Ollama", value=True)
    st.caption("Without Ollama, chat displays cited source excerpts.")

docs_tab, chat_tab, search_tab, eval_tab = st.tabs(
    ["Documents", "AI Chat", "Retrieval Comparison", "Evaluation"]
)
with docs_tab:
    uploads = st.file_uploader("Arabic or English PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Index documents", disabled=not uploads):
        with st.spinner("Extracting, embedding and indexing…"):
            result = api(
                "POST",
                "/documents/upload",
                files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in uploads],
            )
        if result:
            for doc in result["documents"]:
                st.success(
                    f"{doc['name']}: {doc['chunks']} chunks"
                    + (" (already indexed)" if doc["duplicate"] else "")
                )
                for warning in doc["warnings"]:
                    st.warning(warning)
            for error in result["errors"]:
                st.error(error)
    documents = api("GET", "/documents")
    for doc in documents or []:
        col1, col2 = st.columns([5, 1])
        col1.write(f"{doc['name']} — {doc['pages']} pages, {doc['chunks']} chunks")
        if col2.button("Delete", key=doc["document_id"]):
            if api("DELETE", "/documents/" + doc["document_id"]):
                st.rerun()
with chat_tab:
    st.session_state.setdefault("messages", [])
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("response", {}).get("generation_mode") == "source-tables":
                # Native Markdown renders source table relationships clearly.
                # HTML stays disabled for all document and answer content.
                st.markdown(message["text"], unsafe_allow_html=False)
            else:
                rtl_text(message["text"])
            if "response" in message:
                result = message["response"]
                st.caption(f"Generation mode: {result['generation_mode']}")
                description = MODE_DESCRIPTIONS.get(result["generation_mode"])
                if description:
                    st.caption(description)
                if result["warning"]:
                    st.warning(result["warning"])
                for c in citation_sources(result["citations"]):
                    st.caption(f"[{c['label']}] {c['document_name']}, page {c['page']}")
                    with st.expander(f"Verified quotes for {c['label']}"):
                        for quote in c["quotes"]:
                            rtl_text(quote)
                passages(result["evidence"])
    question = st.chat_input("Ask a research question / اطرح سؤالاً")
    if question:
        st.session_state.messages.append({"role": "user", "text": question})
        with st.spinner("Retrieving evidence…"):
            result = api(
                "POST", "/chat", json={"query": question, "language": language, "use_llm": use_llm}
            )
        if result:
            st.session_state.messages.append(
                {"role": "assistant", "text": result["answer"], "response": result}
            )
        st.rerun()
with search_tab:
    query = st.text_input("Comparison question")
    modes = st.multiselect(
        "Pipelines",
        ["dense", "bm25", "hybrid", "hybrid-rerank"],
        default=["dense", "bm25", "hybrid"],
    )
    if st.button("Compare", disabled=not query.strip()):
        for mode, col in zip(modes, st.columns(max(len(modes), 1))):
            with col:
                st.subheader(mode)
                result = api("POST", "/search", json={"query": query, "mode": mode, "top_k": 5})
                if result:
                    st.caption(
                        f"{result['latency_ms']:.1f} ms; reranker: {result['reranker_backend']}"
                    )
                    passages(result["hits"])
with eval_tab:
    st.info(
        "Upload manually annotated cases with current chunk IDs. Demo scores are not scientific benchmarks."
    )
    dataset = st.file_uploader("Evaluation JSON", type=["json"])
    selected = st.multiselect(
        "Evaluate pipelines",
        ["dense", "bm25", "hybrid", "hybrid-rerank"],
        default=["dense", "bm25", "hybrid"],
        key="eval_modes",
    )
    answers = st.checkbox("Evaluate answer lexical proxies (extractive mode)")
    if st.button("Run evaluation", disabled=dataset is None):
        try:
            payload = json.loads(dataset.getvalue())
            result = api(
                "POST",
                "/evaluate",
                json={"cases": payload["cases"], "modes": selected, "generate_answers": answers},
            )
            if result:
                st.session_state["benchmark"] = result
        except (ValueError, KeyError) as exc:
            st.error(f"Invalid dataset: {exc}")
    if "benchmark" in st.session_state:
        result = st.session_state["benchmark"]
        st.caption(result["note"])
        frame = pd.DataFrame(result["summary"]).set_index("mode")
        st.dataframe(frame)
        st.bar_chart(frame[["recall@5", "mrr", "ndcg@5"]])
        st.bar_chart(frame[["query_latency_ms", "rerank_latency_ms"]])
        st.download_button(
            "JSON", json.dumps(result, ensure_ascii=False, indent=2), "benchmark.json"
        )
        st.download_button("CSV", pd.DataFrame(result["rows"]).to_csv(index=False), "benchmark.csv")
