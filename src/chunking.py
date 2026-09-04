import json
import re
from pathlib import Path


INPUT_PATH = Path("data/processed/pages.json")
OUTPUT_PATH = Path("data/processed/chunks.json")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.split("\n")

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if cleaned_lines and cleaned_lines[-1] != "":
            previous = cleaned_lines[-1]

            if (
                not previous.endswith((".", "!", "?", ":", ";"))
                and not line.startswith(("-", "*", "•"))
            ):
                cleaned_lines[-1] = previous + " " + line
                continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def load_pages() -> list[dict]:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def process_pages() -> list[dict]:
    pages = load_pages()
    chunks = []

    for page in pages:
        page_number = page.get("page_number")
        raw_text = page.get("text", "")

        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            continue

        page_chunks = chunk_text(cleaned_text)

        for chunk_index, chunk in enumerate(page_chunks):
            chunks.append(
                {
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk,
                    "source": page.get("source", ""),
                }
            )

    return chunks


def save_chunks(chunks: list[dict]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    chunks = process_pages()
    save_chunks(chunks)

    print(f"Created {len(chunks)} chunks")
    print(f"Saved to {OUTPUT_PATH}")

    
