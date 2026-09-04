import json
from pathlib import Path

import faiss
import numpy as np


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

INPUT_PATH = Path("data/processed/embeddings.json")
INDEX_PATH = Path("data/processed/faiss.index")


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
    # 4. Load embeddings
    # --------------------------------------------------

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "embeddings.json must contain a list of records."
        )

    if len(records) == 0:
        raise ValueError(
            "embeddings.json contains no records."
        )

    print(f"Loaded {len(records)} embedding records.")

    # --------------------------------------------------
    # 5. Validate embeddings
    # --------------------------------------------------

    required_fields = {
        "id",
        "page",
        "chunk_index",
        "text",
        "source",
        "embedding",
    }

    for index, record in enumerate(records):

        if not isinstance(record, dict):
            raise ValueError(
                f"Record {index} is not a dictionary."
            )

        missing_fields = required_fields - set(record.keys())

        if missing_fields:
            raise ValueError(
                f"Record {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if len(record["embedding"]) != 384:
            raise ValueError(
                f"Record {index} does not have a "
                f"384-dimensional embedding."
            )

    # --------------------------------------------------
    # 6. Convert embeddings to NumPy array
    # --------------------------------------------------

    vectors = np.array(
        [record["embedding"] for record in records],
        dtype="float32",
    )

    print(f"Vector shape: {vectors.shape}")

    # --------------------------------------------------
    # 7. Create FAISS index
    # --------------------------------------------------

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # --------------------------------------------------
    # 8. Add vectors to the index
    # --------------------------------------------------

    index.add(vectors)

    # --------------------------------------------------
    # 9. Verify index
    # --------------------------------------------------

    if index.ntotal != len(records):
        raise ValueError(
            f"FAISS contains {index.ntotal} vectors, "
            f"but expected {len(records)}."
        )

    # --------------------------------------------------
    # 10. Create output directory
    # --------------------------------------------------

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 11. Save FAISS index
    # --------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    # --------------------------------------------------
    # 12. Final confirmation
    # --------------------------------------------------

    print()
    print("==========================================")
    print("FAISS index created successfully.")
    print("==========================================")
    print(f"Vectors indexed: {index.ntotal}")
    print(f"Vector dimension: {dimension}")
    print(f"Index: {INDEX_PATH}")


# --------------------------------------------------
# 13. Run main
# --------------------------------------------------

if __name__ == "__main__":
    main()