from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import zipfile


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str


def extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Supported types: {supported}")

    if extension in {".txt", ".md"}:
        return content.decode("utf-8-sig").strip()

    if extension == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(BytesIO(content))
        except (PdfReadError, ValueError) as error:
            raise ValueError("Could not read PDF document") from error
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    from docx import Document
    from docx.opc.exceptions import PackageNotFoundError

    try:
        document = Document(BytesIO(content))
    except (PackageNotFoundError, zipfile.BadZipFile) as error:
        raise ValueError("Could not read Word document") from error
    return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        value = normalized[start:end].strip()
        if value:
            chunks.append(TextChunk(index=len(chunks), text=value))
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks
