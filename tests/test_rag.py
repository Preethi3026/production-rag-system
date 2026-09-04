import re

import pytest

import src.rag as rag_module
from src.retriever import MIN_SCORE, load_resources, search
from src.rag import generate_answer, validate_citations


@pytest.fixture(scope="module")
def resources():
    return load_resources()


def test_resources_load(resources):
    model, index, records = resources

    assert model is not None
    assert index is not None
    assert records is not None

    assert index.ntotal == len(records)
    assert index.ntotal == 121


def test_ollama_connection():
    response = rag_module.ollama_client.list()

    model_names = [
        model.model
        for model in response.models
    ]

    assert any(
        name.startswith(rag_module.OLLAMA_MODEL)
        for name in model_names
    )


def test_faiss_dimension_matches_embedding_model(resources):
    model, index, records = resources

    assert index.d == 384


def test_retrieval_returns_results(resources):
    model, index, records = resources

    results = search(
        "What percentage of Fortune 500 companies use Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert len(results) > 0

    result = results[0]

    assert "text" in result
    assert "page" in result
    assert "chunk_index" in result
    assert "source" in result
    assert "score" in result


def test_retrieval_scores_are_ordered(resources):
    model, index, records = resources

    results = search(
        "What are the main components of Azure Synapse?",
        model,
        index,
        records,
        top_k=5,
    )

    assert len(results) > 0

    scores = [result["score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_retrieval_scores_meet_threshold(resources):
    model, index, records = resources

    results = search(
        "What are the main components of Azure Synapse?",
        model,
        index,
        records,
        top_k=5,
    )

    for result in results:
        assert result["score"] >= 0.60


def test_retrieval_filters_results_below_threshold(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert all(
        result["score"] >= MIN_SCORE
        for result in results
    )


def test_retrieval_respects_top_k(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=3,
    )

    assert len(results) <= 3


def test_retrieval_results_have_required_fields(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=3,
    )

    required_fields = {
        "score",
        "id",
        "page",
        "chunk_index",
        "text",
        "source",
    }

    assert results

    for result in results:
        assert required_fields.issubset(result.keys())


def test_retrieval_scores_are_finite(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=3,
    )

    assert results

    for result in results:
        assert result["score"] == result["score"]
        assert result["score"] != float("inf")
        assert result["score"] != float("-inf")


def test_retrieval_results_are_sorted_descending(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results

    scores = [
        result["score"]
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_retrieval_results_are_independent(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=3,
    )

    assert results

    if len(results) >= 2:
        original_second_id = results[1]["id"]

        results[0]["id"] = "temporary-test-value"

        assert results[1]["id"] == original_second_id


def test_search_does_not_mutate_records(resources):
    model, index, records = resources

    original_records = [
        record.copy()
        for record in records
    ]

    search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=3,
    )

    assert records == original_records


def test_retrieval_top_k_cannot_exceed_available_records(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=len(records) + 100,
    )

    assert len(results) <= len(records)


def test_retrieval_returns_empty_list_when_no_results_meet_threshold(
    resources,
    monkeypatch,
):
    model, index, records = resources

    monkeypatch.setattr(
        "src.retriever.MIN_SCORE",
        1.1,
    )

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results == []


def test_retrieval_result_ids_are_unique(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results

    ids = [
        result["id"]
        for result in results
    ]

    assert len(ids) == len(set(ids))


def test_retrieval_result_text_is_valid(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results

    for result in results:
        assert isinstance(result["text"], str)
        assert result["text"].strip()


def test_retrieval_result_source_is_valid(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results

    for result in results:
        assert isinstance(result["source"], str)
        assert result["source"].strip()


def test_retrieval_result_page_is_valid(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results

    for result in results:
        assert isinstance(result["page"], int)
        assert result["page"] > 0


def test_retrieval_result_chunk_index_is_valid(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results

    for result in results:
        assert isinstance(result["chunk_index"], int)
        assert result["chunk_index"] >= 0


def test_retrieval_result_id_is_valid(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results

    for result in results:
        assert isinstance(result["id"], str)
        assert result["id"].strip()


def test_retrieval_result_score_is_numeric(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results

    for result in results:
        assert isinstance(result["score"], (int, float))


def test_retrieval_result_score_is_in_valid_range(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results

    for result in results:
        assert -1.0 <= result["score"] <= 1.0


def test_search_returns_list(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert isinstance(results, list)


def test_search_results_are_dictionaries(resources):
    model, index, records = resources

    results = search(
        "What is Microsoft Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results

    for result in results:
        assert isinstance(result, dict)


def test_unanswerable_question_returns_no_results(resources):
    model, index, records = resources

    results = search(
        "Who is the president of the United States?",
        model,
        index,
        records,
        top_k=5,
    )

    assert results == []


def test_current_question_is_used_for_unanswerable_retrieval(resources):
    model, index, records = resources

    current_question = "Who is the president of the United States?"

    results = search(
        current_question,
        model,
        index,
        records,
        top_k=5,
    )

    assert results == []


def test_conversation_history_does_not_contaminate_retrieval(
    monkeypatch,
):
    current_question = "Who is the president of the United States?"

    history = [
        {
            "question": "What percentage of Fortune 500 companies use Azure?",
            "answer": (
                "According to [Document 1], Azure is used by "
                "95% of Fortune 500 companies."
            ),
        }
    ]

    captured_queries = []

    def fake_search(
        query,
        model,
        index,
        records,
        top_k=5,
    ):
        captured_queries.append(query)
        return []

    def fake_chat(
        model,
        messages,
        options,
    ):
        return {
            "message": {
                "content": (
                    "I couldn't find the answer "
                    "in the provided documents."
                )
            }
        }

    monkeypatch.setattr(
        rag_module,
        "search",
        fake_search,
    )

    monkeypatch.setattr(
        rag_module.ollama_client,
        "chat",
        fake_chat,
    )

    answer = generate_answer(
        current_question,
        top_k=5,
        history=history,
    )

    assert captured_queries == [current_question]

    assert answer == (
        "I couldn't find the answer "
        "in the provided documents."
    )


def test_generate_answer():
    answer = generate_answer(
        "What are the 5 Vs of Big Data?",
        top_k=5,
    )

    assert answer is not None
    assert len(answer.strip()) > 0

    answer_lower = answer.lower()

    assert "volume" in answer_lower
    assert "velocity" in answer_lower
    assert "variety" in answer_lower
    assert "value" in answer_lower
    assert "veracity" in answer_lower


def test_search_rejects_invalid_top_k(resources):
    model, index, records = resources

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        search(
            "What is Azure Synapse?",
            model,
            index,
            records,
            top_k=0,
        )


def test_search_rejects_non_integer_top_k(resources):
    model, index, records = resources

    with pytest.raises(
        ValueError,
        match="top_k must be an integer",
    ):
        search(
            "What is Azure Synapse?",
            model,
            index,
            records,
            top_k="5",
        )


def test_citations_reference_retrieved_documents(resources):
    model, index, records = resources

    results = search(
        "What percentage of Fortune 500 companies use Azure?",
        model,
        index,
        records,
        top_k=5,
    )

    retrieved_documents = len(results)

    assert retrieved_documents > 0

    valid_answer = (
        "Azure is used by 95% of Fortune 500 companies [Document 1]."
    )

    invalid_answer = (
        "Azure is used by 95% of Fortune 500 companies "
        f"[Document {retrieved_documents + 1}]."
    )

    valid_citations = [
        int(number)
        for number in re.findall(
            r"\[Document (\d+)\]",
            valid_answer,
        )
    ]

    invalid_citations = [
        int(number)
        for number in re.findall(
            r"\[Document (\d+)\]",
            invalid_answer,
        )
    ]

    assert all(
        1 <= number <= retrieved_documents
        for number in valid_citations
    )

    assert not all(
        1 <= number <= retrieved_documents
        for number in invalid_citations
    )

    assert not all(
        1 <= number <= retrieved_documents
        for number in invalid_citations
    )


def test_validate_citations():
    assert validate_citations(
        "I couldn't find the answer in the provided documents.",
        5,
    )

    assert validate_citations(
        "Azure is used by 95% of Fortune 500 companies [Document 1].",
        5,
    )

    assert validate_citations(
        "Azure is a cloud platform [Document 2].",
        5,
    )

    assert not validate_citations(
        "Azure is a cloud platform [Document 6].",
        5,
    )

    assert not validate_citations(
        "Azure is a cloud platform [Document 0].",
        5,
    )

    assert not validate_citations(
        "Azure is a cloud platform [Document 2].",
        1,
    )