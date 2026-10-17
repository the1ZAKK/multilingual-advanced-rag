import pytest

from rag.query import (
    asks_problem,
    claim_target,
    contribution_question,
    models_results_question,
    numbered_list_question,
    resolve_language,
    validate_output_language,
    verification_question,
)


@pytest.mark.parametrize(
    "question",
    [
        "List the authors' main contributions.",
        "What is the paper's contribution?",
        "What does this work contribute?",
        "ما هي المساهمات التي يذكرها المؤلفون؟",
        "اذكر إسهامات البحث.",
        "وَبِالْإِسْهَامَاتِ الأساسية لهذه الدراسة؟",
        "ما مساهمة هذه الورقة؟",
        "ما الإسهام الرئيسي؟",
        "ما مساهماتهم في هذه الدراسة؟",
    ],
)
def test_authored_contribution_intent_in_both_languages(question):
    assert contribution_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "Which models were evaluated and what were their results?",
        "What is the contribution of temperature to the final score?",
        "What does the word contribution mean?",
        "Define contribution.",
        "ما تعريف كلمة مساهمة؟",
        "هل يقدّم البحث نموذج مكافأة؟",
    ],
)
def test_incidental_contribution_or_other_intent_is_not_author_list(question):
    assert not contribution_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "What problem does the work address?",
        "What research gap is identified?",
        "ما المشكلة التي تعالجها الورقة؟",
        "اذكر الفجوة التي يعالجها البحث.",
        "ما تحديات الدراسة ومساهماتها؟",
    ],
)
def test_problem_intent(question):
    assert asks_problem(question)


@pytest.mark.parametrize(
    "question",
    [
        "Which models were evaluated and what numerical results were reported?",
        "Give LLM performance scores.",
        "ما النماذج المقيمة وما نتائجها؟",
        "اذكر نماذج اللغة مع النتائج الرقمية.",
        "قارن أداء النموذج في المقاييس المطلوبة.",
    ],
)
def test_model_result_intent(question):
    assert models_results_question(question)


@pytest.mark.parametrize(
    "question",
    ["Which models were evaluated?", "What were the benchmark sizes?", "ما المساهمات؟"],
)
def test_model_result_intent_requires_both_roles_and_results(question):
    assert not models_results_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "Does the paper introduce a process reward model?",
        "Is HumanEval a model?",
        "Has the study trained a new sensor classifier?",
        "Verify whether the work proposes a retrieval method.",
        "هل تقدّم الورقة نموذج مكافأة؟",
        "أيمكن لهذه الدراسة إثبات ذلك؟",
        "تحقق مما إذا كان البحث يقدم هذه الطريقة.",
    ],
)
def test_generic_claim_verification_intent(question):
    assert verification_question(question)


@pytest.mark.parametrize(
    "question",
    ["What models are evaluated?", "Explain the experiments.", "ما نتائج النموذج؟"],
)
def test_information_questions_do_not_become_polar_claims(question):
    assert not verification_question(question)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Does ExamplePaper introduce an Arabic process reward model? Cite evidence.",
            "Arabic process reward model",
        ),
        ("Did this study train a new sensor classifier?", "new sensor classifier"),
        ("هل تقدم ورقة ExamplePaper نموذج مكافأة عربي؟", "نموذج مكافاة عربي"),
        ("هل تقترح هذه الدراسة طريقة استرجاع؟", "طريقة استرجاع"),
        ("Does this paper introduce a model or a dataset?", None),
        ("What model does the paper introduce?", None),
        ("Is the evidence enough?", None),
    ],
)
def test_claim_target_is_generic_and_fails_on_ambiguous_objects(question, expected):
    assert claim_target(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Number them 1–3.",
        "Use a numbered list.",
        "Enumerate the main findings.",
        "List all three stated contributions.",
        "اذكر المساهمات في قائمة مرقمة.",
        "رقّمها من 1 إلى 3.",
    ],
)
def test_numbering_is_explicit(question):
    assert numbered_list_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "Does the work introduce a reward model?",
        "Which models were evaluated and what are their scores?",
        "What is the number of models?",
        "ما عدد النماذج؟",
        "Summarize the paper.",
    ],
)
def test_other_questions_do_not_force_numbering(question):
    assert not numbered_list_question(question)


@pytest.mark.parametrize(
    ("question", "requested", "expected"),
    [
        ("ما مساهمات الدراسة؟", "auto", "ar"),
        ("Qwen3 ما نتائج هذا النموذج؟", "auto", "ar"),
        ("Compare Qwen3 and PaLM 540B.", "auto", "en"),
        ("١٢٣ ؟ ،", "auto", "en"),
        ("What is the score؟", "auto", "en"),
        ("What does علم mean?", "auto", "en"),
        ("ما المساهمات؟", "en", "en"),
        ("What are the contributions?", "ar", "ar"),
    ],
)
def test_language_resolution_respects_preference_and_meaningful_script(
    question, requested, expected
):
    assert resolve_language(question, requested) == expected


@pytest.mark.parametrize(
    ("answer", "language"),
    [
        ("تقيّم الدراسة نماذج Qwen3 وGPT-4o في مهام STEM وMCQ [S1] (page 2).", "ar"),
        ("تستخدم الدراسة HumanEval+ وMBPP+ كمعايير لتقييم توليد الشفرة.", "ar"),
        ("لا تثبت الأدلة المسترجعة أن الورقة تقدم نموذج PRM جديداً.", "ar"),
        ("The paper evaluates Qwen3 on the HumanEval+ benchmark [S1] (page 2).", "en"),
        ("The model's Arabic name is علم, and its performance is reported in the table.", "en"),
    ],
)
def test_output_language_allows_model_ids_acronyms_and_brief_foreign_names(answer, language):
    validate_output_language(answer, language)


def test_source_derived_plain_names_and_attached_arabic_clitics_are_allowed():
    validate_output_language(
        "النماذج: Jais وFanar وLlama وGemma.",
        "ar",
        identifiers=("Jais", "Fanar", "Llama", "Gemma"),
    )
    validate_output_language("النماذج: Qwen3 وGPT-4o وLlama-3.1.", "ar")


@pytest.mark.parametrize(
    ("answer", "language"),
    [
        ("First, we present three comprehensive benchmarks spanning science and coding.", "ar"),
        ("الإجابة: The authors evaluate forty language models and analyze robustness.", "ar"),
        (
            "تقدم الدراسة معايير جديدة لتقييم قدرات النماذج اللغوية في العلوم والبرمجة.\n"
            "Second, the authors evaluate forty language models.",
            "ar",
        ),
        ("These are the results. تقدم الورقة ثلاثة معايير جديدة.", "en"),
        ("لا توجد أدلة كافية لإثبات هذا الادعاء.", "en"),
        ("Qwen3 PRM 1.7B [S1]", "ar"),
        ("الإجابة: THE AUTHORS EVALUATE FORTY LANGUAGE MODELS.", "ar"),
    ],
)
def test_wrong_language_and_short_heading_evasion_are_rejected(answer, language):
    with pytest.raises(ValueError, match="language mismatch"):
        validate_output_language(answer, language)


def test_invalid_language_is_rejected():
    with pytest.raises(ValueError):
        resolve_language("question", "fr")
    with pytest.raises(ValueError):
        validate_output_language("answer", "auto")
