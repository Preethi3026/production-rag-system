import json
import os
from pathlib import Path

from dotenv import load_dotenv
import faiss
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))

EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "embeddings.json"
INDEX_PATH = BASE_DIR / "data" / "processed" / "faiss.index"


# --------------------------------------------------
# 2. Model
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# --------------------------------------------------
# 3. Load resources
# --------------------------------------------------

def load_resources():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index...")

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print("Loading embedding records...")

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_PATH}"
        )

    with EMBEDDINGS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "embeddings.json must contain a list."
        )

    if len(records) == 0:
        raise ValueError(
            "embeddings.json contains no records."
        )

    if index.ntotal != len(records):
        raise ValueError(
            f"FAISS contains {index.ntotal} vectors, "
            f"but embeddings.json contains "
            f"{len(records)} records."
        )

    expected_dimension = 384

    if index.d != expected_dimension:
        raise ValueError(
            f"FAISS index dimension is {index.d}, "
            f"but expected {expected_dimension}."
        )

    print(
        f"Loaded {len(records)} records "
        f"and {index.ntotal} vectors."
    )

    return model, index, records


# --------------------------------------------------
# 4. Search function
# --------------------------------------------------

def search(
    query,
    model,
    index,
    records,
    top_k=5,
):

    # --------------------------------------------------
    # 4.1 Validate query
    # --------------------------------------------------

    if not isinstance(query, str):
        raise ValueError(
            "Query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    if not isinstance(top_k, int):
        raise ValueError(
            "top_k must be an integer."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    # Do not request more results than available
    top_k = min(top_k, index.ntotal)

    # --------------------------------------------------
    # 4.2 Convert query into embedding
    # --------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    # --------------------------------------------------
    # 4.3 Search FAISS
    # --------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    # --------------------------------------------------
    # 4.4 Build search results
    # --------------------------------------------------

    results = []


    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position == -1:
            continue

        if float(score) < MIN_SCORE:
            continue

        record = records[index_position]

        results.append(
            {
                "score": float(score),
                "id": record["id"],
                "page": record["page"],
                "chunk_index": record["chunk_index"],
                "text": record["text"],
                "source": record["source"],
            }
        )

    return results


# --------------------------------------------------
# 5. Main function
# --------------------------------------------------

def main():

    model, index, records = load_resources()

    print()
    print("==========================================")
    print("Semantic Search")
    print("==========================================")

    query = input(
        "Enter your question: "
    )

    results = search(
        query,
        model,
        index,
        records,
        top_k=5,
    )

    print()
    print("==========================================")
    print("Search Results")
    print("==========================================")

    for number, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"Result {number}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Page: {result['page']}"
        )

        print(
            f"Chunk: {result['chunk_index']}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            "Text:"
        )

        print(
            result["text"]
        )

        print(
            "------------------------------------------"
        )


# --------------------------------------------------
# 6. Run main
# --------------------------------------------------

if __name__ == "__main__":
    main()