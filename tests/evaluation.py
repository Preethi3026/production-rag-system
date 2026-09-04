import json
import re
from pathlib import Path

from src.rag import generate_answer


BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR / "evaluation_questions.json"
RESULTS_FILE = BASE_DIR / "evaluation_results.json"


def load_questions():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {QUESTIONS_FILE}"
        )

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "evaluation_questions.json must contain a list."
        )

    return questions


def has_document_citation(answer):
    return bool(re.search(r"\[Document \d+\]", answer))


def is_expected_answer(answer, expected_type):
    if expected_type == "answerable":
        return answer.strip() != ""

    if expected_type == "unanswerable":
        return "I couldn't find the answer in the provided documents." in answer

    return False


def main():
    questions = load_questions()

    results = []

    answerability_passed = 0
    citation_passed = 0

    print("=" * 60)
    print("RAG Evaluation")
    print("=" * 60)
    print(f"Questions: {len(questions)}")
    print()

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        expected_type = item["expected_type"]

        print(f"Question {question_id}: {question}")
        print(f"Expected type: {expected_type}")

        answer = generate_answer(
            question,
            top_k=5,
        )

        answerability_ok = is_expected_answer(
            answer,
            expected_type,
        )

        citation_ok = (
            has_document_citation(answer)
            if expected_type == "answerable"
            else True
        )

        if answerability_ok:
            answerability_passed += 1

        if citation_ok:
            citation_passed += 1

        result = {
            "id": question_id,
            "question": question,
            "expected_type": expected_type,
            "answer": answer,
            "answerability_passed": answerability_ok,
            "citation_passed": citation_ok,
        }

        results.append(result)

        print("Answer:")
        print(answer)
        print(f"Answerability check: {'PASS' if answerability_ok else 'FAIL'}")
        print(f"Citation check: {'PASS' if citation_ok else 'FAIL'}")
        print("-" * 60)

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("Evaluation completed.")
    print("=" * 60)

    print(
        f"Answerability accuracy: "
        f"{answerability_passed}/{len(questions)}"
    )

    print(
        f"Citation coverage: "
        f"{citation_passed}/{len(questions)}"
    )

    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()