from dataclasses import dataclass
from pathlib import Path


@dataclass
class PDFRAGConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    raw_pdf_dir: Path = project_root / "pdf_rag_data" / "raw_pdfs"
    raw_text_dir: Path = project_root / "pdf_rag_data" / "raw_texts"
    chunk_file: Path = project_root / "pdf_rag_data" / "processed_chunks.jsonl"
    index_dir: Path = project_root / "pdf_rag_index"
    faiss_index_file: Path = index_dir / "chunks.faiss"
    metadata_file: Path = index_dir / "chunks_metadata.jsonl"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_model: str = "glm-4-flash"
    chunk_size: int = 800 # 把pdf切分成800字左右的片段
    chunk_overlap: int = 120 # 相邻片段重叠120字，避免关键上下文被切断
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 6

    def ensure_dirs(self) -> None:
        self.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.raw_text_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_file.parent.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
