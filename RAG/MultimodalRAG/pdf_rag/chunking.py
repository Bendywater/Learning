import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

from .document_loader import PageText


# 文本切分
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    page: int
    chunk_index: int
    text: str


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(end - overlap, start + 1)
            continue

        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n{paragraph}".strip() if tail else paragraph

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]

# 这里会把每个PageText切分成多个Chunk
def build_chunks(pages: Iterable[PageText], chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page in pages:
        page_chunks = split_text(page.text, chunk_size=chunk_size, overlap=overlap)
        for idx, text in enumerate(page_chunks):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.doc_id}_p{page.page:04d}_c{idx:04d}",
                    doc_id=page.doc_id,
                    title=page.title,
                    source_path=page.source_path,
                    page=page.page,
                    chunk_index=idx,
                    text=text,
                )
            )
    return chunks


def save_chunks(chunks: Iterable[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks
