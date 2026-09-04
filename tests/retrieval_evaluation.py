import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retriever import load_resources, search


def main():
    model, index, records = load_resources()

    test_questions = [
        {
            "question": "What percentage of Fortune 500 companies use Azure?",
            "expected_retrieval": True,
        },
        {
            "question": "What is Azure Cosmos DB?",
            "expected_retrieval": True,
        },
        {
            "question": "What are the main components of Azure Synapse?",
            "expected_retrieval": True,
        },
        {
            "question": "Who is the president of the United States?",
            "expected_retrieval": False,
        },
    ]

    print("=" * 60)
    print("Retrieval Evaluation")
    print("=" * 60)

    total_passed = 0

    for number, item in enumerate(test_questions, start=1):
        question = item["question"]
        expected_retrieval = item["expected_retrieval"]

        print()
        print(f"Question {number}: {question}")
        print("-" * 60)

        results = search(
            question,
            model,
            index,
            records,
            top_k=5,
        )

        retrieval_ok = bool(results) == expected_retrieval

        print(
            f"Retrieval check: "
            f"{'PASS' if retrieval_ok else 'FAIL'}"
        )

        if retrieval_ok:
            total_passed += 1

        if not results:
            print("No documents retrieved.")
            continue

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"Score={result['score']:.4f} | "
                f"Page={result['page']} | "
                f"Chunk={result['chunk_index']}"
            )

            print(f"   Source: {result['source']}")
            print(f"   Text: {result['text'][:200]}...")
            print()

    print()
    print("=" * 60)
    print("Retrieval Evaluation Summary")
    print("=" * 60)
    print(
        f"Retrieval accuracy: "
        f"{total_passed}/{len(test_questions)}"
    )

    if total_passed == len(test_questions):
        print("Overall result: PASS")
    else:
        print("Overall result: FAIL")


if __name__ == "__main__":
    main()