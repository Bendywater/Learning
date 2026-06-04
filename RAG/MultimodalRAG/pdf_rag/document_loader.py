import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader


@dataclass
class PageText:
    doc_id: str
    source_path: str
    title: str
    page: int
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# 读PDF文档用PdfReader函数
def load_pdf(pdf_path: Path) -> List[PageText]:
    reader = PdfReader(str(pdf_path))
    pages: List[PageText] = []
    doc_id = pdf_path.stem
    for idx, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue
        pages.append(
            PageText(
                doc_id=doc_id, # 文件名
                source_path=str(pdf_path), # 文件路径
                title=pdf_path.stem, # 文件名
                page=idx, # 页码
                text=text, # 页码对应的文本
            )
        )
    return pages


def load_text_file(text_path: Path) -> List[PageText]:
    text = clean_text(text_path.read_text(encoding="utf-8"))
    if not text:
        return []
    return [
        PageText(
            doc_id=text_path.stem,
            source_path=str(text_path),
            title=text_path.stem,
            page=1,
            text=text,
        )
    ]


def load_documents(pdf_dir: Path, text_dir: Path) -> List[PageText]:
    pages: List[PageText] = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        pages.extend(load_pdf(path))
    for path in sorted(text_dir.glob("*.txt")):
        pages.extend(load_text_file(path))
    for path in sorted(text_dir.glob("*.md")):
        pages.extend(load_text_file(path))
    return pages
