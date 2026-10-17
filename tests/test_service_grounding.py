from rag.ingestion.structure import section_marks
from rag.models import Chunk, DocumentInfo, Hit
from rag.service import RAGService


def test_numerical_chat_reaches_table_renderer_and_falls_back_cleanly(service, pdf_bytes):
    from rag.models import ChatRequest

    service.ingest(pdf_bytes("Models evaluated report accuracy results."), "Study.pdf")
    result = service.chat(ChatRequest(query="Which models and results are reported?"))
    assert result.generation_mode == "extractive"
    assert "No unambiguous source-bound numerical table relationships" in result.warning
    assert result.citations


def test_extractive_reordered_blocks_keep_native_contiguous_quotes():
    from rag.generation.ollama import extractive

    first = "The first column describes the dataset."
    second = "The second column reports the results."
    hit = Hit(
        chunk=chunk("d", "Study.pdf", "Results", first + "\n\n" + second),
        score=1,
        native_page_text=second + "\n" + first,
    )
    answer, citations = extractive([hit], "en")
    assert first in answer and second in answer
    assert len(citations) >= 2
    assert all(c.quote in hit.native_page_text and c.quote in hit.chunk.text for c in citations)


def chunk(document, name, section, text, page=1):
    return Chunk(
        chunk_id=f"{document}:{page}:{section}",
        document_id=document,
        document_name=name,
        page=page,
        text=text,
        normalized=text,
        start=0,
        end=len(text),
        section=section,
    )


def bare_service(chunks):
    import threading
    from types import SimpleNamespace

    service = object.__new__(RAGService)
    service.lock = threading.RLock()
    service.settings = SimpleNamespace(max_context_chars=12000)
    service.index = SimpleNamespace(chunks=chunks)
    docs = {
        c.document_id: DocumentInfo(
            document_id=c.document_id, name=c.document_name, pages=2, chunks=1
        )
        for c in chunks
    }
    service.store = SimpleNamespace(documents=lambda: list(docs.values()))
    return service


def test_arabic_authored_list_uses_explicit_document_section_even_when_retrieval_misses():
    authored = chunk(
        "alpha",
        "8AB-Bridging domains.pdf",
        "Contributions",
        "The contributions are threefold. First, we introduce a dataset.",
    )
    decoy = chunk("beta", "Other study.pdf", "Contributions", "Unrelated contributions.")
    service = bare_service([authored, decoy])
    selected = service.select_evidence(
        [Hit(chunk=decoy, score=1)], "ما المساهمات الثلاث التي يذكرها مؤلفو ورقة 8AB؟"
    )
    assert [h.chunk.document_id for h in selected] == ["alpha"]
    assert selected[0].chunk.text == authored.text


def test_common_title_words_do_not_accidentally_scope_documents():
    service = bare_service(
        [
            chunk("a", "Arabic models.pdf", "Abstract", "A"),
            chunk("b", "English models.pdf", "Abstract", "B"),
        ]
    )
    assert service.named_documents("Which Arabic models were evaluated?") == set()


def test_full_title_reference_scopes_another_actual_paper_title_without_paper_specific_rule():
    service = bare_service(
        [
            chunk("a", "14-Chain-of-Thought Prompting Elicits Reasoning.pdf", "Abstract", "A"),
            chunk("b", "8AB.pdf", "Abstract", "B"),
        ]
    )
    assert service.named_documents(
        "What results does Chain-of-Thought Prompting Elicits Reasoning report?"
    ) == {"a"}


def test_ambiguous_filename_identifiers_do_not_choose_arbitrarily():
    service = bare_service(
        [
            chunk("a", "8AB original.pdf", "Abstract", "A"),
            chunk("b", "8AB revised.pdf", "Abstract", "B"),
        ]
    )
    assert service.named_documents("Summarize 8AB") == set()


def test_distinctive_uploaded_camelcase_identifier_scopes_without_an_answer_lookup():
    service = bare_service(
        [
            chunk("a", "12-OrbitBench -- Evaluating spatial reasoning.pdf", "Abstract", "A"),
            chunk("b", "Weather study.pdf", "Abstract", "B"),
        ]
    )
    assert service.named_documents("What limitations does OrbitBench discuss?") == {"a"}
    assert service.named_documents("What does the Evaluating paper report?") == set()


def test_shared_uploaded_camelcase_identifier_remains_ambiguous():
    service = bare_service(
        [
            chunk("a", "OrbitBench original.pdf", "Abstract", "A"),
            chunk("b", "OrbitBench revised.pdf", "Abstract", "B"),
        ]
    )
    assert service.named_documents("Summarize OrbitBench") == set()


def test_arabic_authored_section_marker_keeps_native_offsets():
    text = "المقدمة\nمساهمات هذه الدراسة ثلاث: أولا، نقدم مجموعة بيانات.\n"
    marks = section_marks(text, "Introduction")
    assert (text.index("مساهمات"), "Contributions") in marks


def test_appendix_prompt_cannot_spoof_an_authored_contribution_section():
    text = "Our contributions are threefold: First, we introduce a hypothetical model.\n"
    assert section_marks(text, "Appendix") == [(0, "Appendix")]
    arabic = "مساهماتنا ثلاث: أولا، نقدم مثالا وهميا.\n"
    assert section_marks(arabic, "A.2 Prompt examples") == [(0, "A.2 Prompt examples")]


def test_method_paragraph_can_declare_two_distinct_metrics_without_conflation():
    methods = chunk(
        "d",
        "Study.pdf",
        "Experimental Setup",
        "STEM benchmarks use accuracy. For code benchmarks, pass@1 evaluation was adopted.",
    )
    table_page = chunk(
        "d", "Study.pdf", "Results", "Native Synthetic MCQ Completion HumanEval MBPP", page=2
    )
    service = bare_service([methods, table_page])
    service._table_cache = {
        "d": [
            {
                "status": "ok",
                "page": 2,
                "columns": [{"path": ["Native", "MCQ"]}, {"path": ["Synthetic", "Completion"]}],
            },
            {"status": "ok", "page": 2, "columns": [{"path": ["HumanEval"]}, {"path": ["MBPP"]}]},
        ]
    }
    service._page_cache = {"d": [methods.text, table_page.text]}
    selected = service.table_evidence([Hit(chunk=methods, score=1)], {"d"})
    tables = next(h.tables for h in selected if h.tables)
    assert [t["metric_bindings"][0]["metric"] for t in tables] == ["accuracy", "pass@1"]
    assert all(t["metric_bindings"][0]["quote"] == methods.text for t in tables)
