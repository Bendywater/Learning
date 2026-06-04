import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk, load_chunks


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(vectors, dtype="float32")


def build_faiss_index(chunks: List[Chunk], embedding_model: str, index_path: Path, metadata_path: Path) -> None:
    if not chunks:
        raise ValueError("No chunks to index. Put PDFs in pdf_rag_data/raw_pdfs first.")

    encoder = EmbeddingModel(embedding_model)
    vectors = encoder.encode([chunk.text for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as f:
        for row_id, chunk in enumerate(chunks):
            data = asdict(chunk)
            data["row_id"] = row_id
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_metadata(metadata_path: Path) -> List[dict]:
    rows: List[dict] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
