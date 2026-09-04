from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()


if __name__ == "__main__":
    embedder = Embedder("BAAI/bge-small-en-v1.5")

    result = embedder.embed_query(
        "What is this research paper about?"
    )

    print(f"Embedding dimensions: {len(result)}")
    print(f"First 5 values: {result[:5]}")
