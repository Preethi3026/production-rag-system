from pathlib import Path
import json

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text and metadata from every PDF page."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            {
                "page_number": page_number,
                "text": text.strip(),
                "source": pdf_path.name,
            }
        )

    return pages


def save_pages_to_json(pages: list[dict], output_path: str) -> None:
    """Save extracted pages as JSON."""

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(pages, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    pdf_file = "data/raw/research_paper.pdf"
    output_file = "data/processed/pages.json"

    pages = extract_text_from_pdf(pdf_file)

    save_pages_to_json(pages, output_file)

    print(f"Extracted {len(pages)} pages.")
    print(f"Saved extracted text to: {output_file}")