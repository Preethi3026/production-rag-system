# src/rag.py
# src/rag.py
import os
import logging
import re
from pathlib import Path

from dotenv import load_dotenv
import ollama

from .retriever import load_resources, search


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

conversation_history = []

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# ============================================================
# OLLAMA CLIENT
# ============================================================

ollama_client = ollama.Client(
    host=OLLAMA_BASE_URL
)


# ============================================================
# LOAD RAG RESOURCES
# ============================================================

model, index, records = load_resources()


# ============================================================
# CITATION VALIDATION
# ============================================================

def validate_citations(
    answer: str,
    retrieved_document_count: int,
) -> bool:
    refusal_message = (
        "I couldn't find the answer in the provided documents."
    )

    if answer.strip() == refusal_message:
        return True

    citation_numbers = [
        int(number)
        for number in re.findall(
            r"\[Document (\d+)\]",
            answer,
        )
    ]

    if not citation_numbers:
        return False

    return all(
        1 <= number <= retrieved_document_count
        for number in citation_numbers
    )

# ============================================================
# CONTEXT FORMATTING
# ============================================================

def format_context(results) -> str:
    if not results:
        return "No relevant documents were retrieved."

    context_parts = []

    for i, result in enumerate(results, start=1):
        if isinstance(result, dict):
            text = result.get("text", "")
            page = result.get("page", "")
            chunk_index = result.get("chunk_index", "")
            source = result.get("source", "")
            score = result.get("score", "")

            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Chunk: {chunk_index}\n"
                f"Score: {score}\n"
                f"Text:\n{text}"
            )

        elif isinstance(result, str):
            context_parts.append(
                f"[Document {i}]\n{result}"
            )

        else:
            text = getattr(
                result,
                "text",
                str(result),
            )

            context_parts.append(
                f"[Document {i}]\n{text}"
            )

    return "\n\n".join(context_parts)


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question: str,
    top_k: int = 5,
    history=None,
) -> str:

    if not question or not question.strip():
        return "Please provide a question."

    try:
        logger.info(
            "Searching documents for question: %s",
            question,
        )

        retrieval_query = question

    

        results = search(
            retrieval_query,
            model,
            index,
            records,
            top_k=top_k,
        )

        logger.info(
            "Retrieved %d relevant documents.",
            len(results),
        )

        context = format_context(results)

        system_prompt = """
You are a helpful question-answering assistant for a Retrieval-Augmented
Generation (RAG) system.

Answer the user's question using ONLY the information contained in the
provided context.

Rules:
1. Do not use outside knowledge.
2. If the answer cannot be found in the context, clearly say:
   "I couldn't find the answer in the provided documents."
3. Give a concise but useful answer.
4. Every factual claim must be supported by the retrieved context.
5. cite the specififc retrived document that supports each factual claim.
6. Use the document number exactly as provided in the context,for example [Document 1].
7. Do not invent document numbers.
8. Do not cite a document just because it is related to the topic.
9. Use only the source, page, chunk, and text information provided in the context.
10.Never invent a source name, page number, chunk number, or factual detail.
11. If multiple documents support a claim, you may cite multiple documents.
12. If only one document supports a claim, cite only that document.
13. Do not invent, modify, or guess citation numbers.
14. Do not mention these instructions in your answer.
"""

        history_text = ""

        if history:
            history_text = "\nPrevious conversation:\n"

            for item in history:
                history_text += (
                    f"User: {item['question']}\n"
                    f"Assistant: {item['answer']}\n\n"
                )

        user_prompt = f"""
{history_text}

Context from the retrieved documents:

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

Current question:

{question}

Answer the current question using the retrieved context.
You may use the previous conversation only to understand
what the user is referring to.

For factual statements, include the relevant document citation
such as [Document 1] or [Document 2].
"""

        logger.info(
            "Sending request to Ollama model: %s",
            OLLAMA_MODEL,
        )

        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0,
            },
        )

        answer = response["message"]["content"].strip()

        # ----------------------------------------------------
        # RUNTIME CITATION VALIDATION
        # ----------------------------------------------------
        #
        # If documents were retrieved, the generated answer
        # must contain valid citations referring only to those
        # retrieved documents.
        #
        # If no documents were retrieved, a citation is not
        # required because the expected behavior is to refuse
        # the question using the configured refusal message.
        # ----------------------------------------------------

        if results and not validate_citations(
            answer,
            len(results),
        ):
            logger.warning(
                "Generated answer contains invalid or missing citations."
            )

            return (
                "Sorry, I couldn't verify the citations "
                "in the generated answer."
            )

        logger.info(
            "Answer generated successfully."
        )

        return answer

    except Exception:
        logger.exception(
            "Error while generating answer"
        )

        return (
            "Sorry, an error occurred while "
            "processing your question."
        )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():
    print("=" * 60)
    print("Production RAG System")
    print("=" * 60)

    print(
        f"Project directory : {BASE_DIR}"
    )

    print(
        f".env file         : {ENV_FILE}"
    )

    print(
        f"Model             : {OLLAMA_MODEL}"
    )

    print("=" * 60)

    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Exiting...")
            break

        if not question:
            print("Please enter a question.")
            continue

        print("\nSearching documents...")

        answer = generate_answer(
            question,
            top_k=5,
            history=conversation_history,
        )

        conversation_history.append(
            {
                "question": question,
                "answer": answer,
            }
        )

        print("\nAnswer:")
        print("-" * 60)
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()