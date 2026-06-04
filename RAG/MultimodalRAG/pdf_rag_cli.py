import argparse
import json

from pdf_rag.chunking import build_chunks, save_chunks
from pdf_rag.config import PDFRAGConfig
from pdf_rag.document_loader import load_documents
from pdf_rag.index_store import build_faiss_index
from pdf_rag.retriever import PDFRAGRetriever


def build_command(args: argparse.Namespace) -> None:
    config = PDFRAGConfig(
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    config.ensure_dirs()

    pages = load_documents(config.raw_pdf_dir, config.raw_text_dir)
    if not pages:
        raise SystemExit(
            "没有加载到文档。请把 PDF 放到 pdf_rag_data/raw_pdfs/，"
            "或把 txt/md 放到 pdf_rag_data/raw_texts/ 后重试。"
        )

    chunks = build_chunks(pages, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    save_chunks(chunks, config.chunk_file)
    build_faiss_index(chunks, config.embedding_model, config.faiss_index_file, config.metadata_file)

    print(f"加载页数: {len(pages)}")
    print(f"切分片段: {len(chunks)}")
    print(f"chunk 文件: {config.chunk_file}")
    print(f"Faiss 索引: {config.faiss_index_file}")
    print(f"元数据: {config.metadata_file}")


def ask_command(args: argparse.Namespace) -> None:
    from pdf_rag.generator import ZhipuGenerator

    config = PDFRAGConfig(
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_k=args.rerank_top_k,
    )
    retriever = PDFRAGRetriever(
        config.faiss_index_file,
        config.metadata_file,
        config.embedding_model,
        config.reranker_model if args.use_reranker else "",
    )
    contexts = retriever.retrieve(
        args.query,
        dense_top_k=config.dense_top_k,
        bm25_top_k=config.bm25_top_k,
        rerank_top_k=config.rerank_top_k,
    )

    print("\n=== 检索与重排结果 ===")
    for idx, row in enumerate(contexts, start=1):
        print(
            f"[{idx}] {row['chunk_id']} | page={row['page']} | "
            f"hybrid={row['hybrid_score']:.4f} | rerank={row['rerank_score']:.4f}"
        )
        print(row["text"][:220].replace("\n", " ") + "\n")

    generator = ZhipuGenerator(model_name=config.llm_model)
    answer = generator.generate(args.query, contexts)
    print("\n=== 生成结果 ===")
    print(answer)


def inspect_command(args: argparse.Namespace) -> None:
    config = PDFRAGConfig()
    if not config.metadata_file.exists():
        raise SystemExit("还没有构建索引，请先运行 build。")
    rows = []
    with config.metadata_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"chunk 总数: {len(rows)}")
    for row in rows[: args.limit]:
        print(f"- {row['chunk_id']} | {row['title']} | 第 {row['page']} 页")
        print(row["text"][:160].replace("\n", " ") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="标准 PDF RAG：加载、切分、向量化、检索、重排、生成")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="从 PDF/txt/md 构建 chunk 和 Faiss 索引")
    build_parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    build_parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    build_parser.add_argument("--chunk-size", type=int, default=800)
    build_parser.add_argument("--chunk-overlap", type=int, default=120)
    build_parser.set_defaults(func=build_command)

    ask_parser = subparsers.add_parser("ask", help="提问，执行检索、重排和生成")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    ask_parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    ask_parser.add_argument("--dense-top-k", type=int, default=20)
    ask_parser.add_argument("--bm25-top-k", type=int, default=20)
    ask_parser.add_argument("--rerank-top-k", type=int, default=6)
    ask_parser.add_argument("--use-reranker", action="store_true", help="启用 CrossEncoder 重排，首次运行会下载模型")
    ask_parser.set_defaults(func=ask_command)

    inspect_parser = subparsers.add_parser("inspect", help="查看已经构建的 chunk")
    inspect_parser.add_argument("--limit", type=int, default=5)
    inspect_parser.set_defaults(func=inspect_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
