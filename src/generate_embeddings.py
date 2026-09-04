import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

INPUT_PATH = Path("data/processed/chunks.json")
OUTPUT_PATH = Path("data/processed/embeddings.json")


# --------------------------------------------------
# 2. Main function
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # 3. Check input file
    # --------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    # --------------------------------------------------
    # 4. Load chunks
    # --------------------------------------------------

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError(
            "chunks.json must contain a list of chunks."
        )

    if len(chunks) == 0:
        raise ValueError(
            "chunks.json contains no chunks."
        )

    print(f"Loaded {len(chunks)} chunks.")

    # --------------------------------------------------
    # 5. Validate chunk structure
    # --------------------------------------------------

    required_fields = {
        "page",
        "chunk_index",
        "text",
        "source",
    }

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk {index} is not a dictionary."
            )

        missing_fields = required_fields - set(chunk.keys())

        if missing_fields:
            raise ValueError(
                f"Chunk {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

    # --------------------------------------------------
    # 6. Load embedding model
    # --------------------------------------------------

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------
    # 7. Extract text
    # --------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------------------
    # 8. Generate embeddings
    # --------------------------------------------------

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # --------------------------------------------------
    # 9. Create output records
    # --------------------------------------------------

    output = []

    for chunk, embedding in zip(chunks, embeddings):

        # Create our own unique ID.
        # chunks.json does NOT contain chunk_id.

        chunk_id = (
            f"{chunk['source']}:"
            f"{chunk['page']}:"
            f"{chunk['chunk_index']}"
        )

        output.append(
            {
                "id": chunk_id,
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "source": chunk["source"],
                "embedding": embedding.tolist(),
            }
        )

    # --------------------------------------------------
    # 10. Create output directory
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 11. Save embeddings.json
    # --------------------------------------------------

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------
    # 12. Verify output
    # --------------------------------------------------

    if len(output) != len(chunks):
        raise ValueError(
            f"Number of embeddings ({len(output)}) "
            f"does not match number of chunks "
            f"({len(chunks)})."
        )

    if len(output) == 0:
        raise ValueError(
            "No embeddings were generated."
        )

    embedding_dimension = len(
        output[0]["embedding"]
    )

    if embedding_dimension != 384:
        raise ValueError(
            f"Expected 384-dimensional embeddings, "
            f"but got {embedding_dimension}."
        )

    for item in output:

        if len(item["embedding"]) != 384:
            raise ValueError(
                "Found an embedding that is not "
                "384-dimensional."
            )

    # --------------------------------------------------
    # 13. Final confirmation
    # --------------------------------------------------

    print()
    print("==========================================")
    print("Embedding generation completed successfully.")
    print("==========================================")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {len(output)}")
    print(f"Embedding dimension: {embedding_dimension}")
    print(f"Output: {OUTPUT_PATH}")


# --------------------------------------------------
# 14. Run main
# --------------------------------------------------

if __name__ == "__main__":
    main()