把权威资料 PDF 放到这个目录，然后运行：

```bash
python pdf_rag_cli.py build
```

构建结果会写入：

- `pdf_rag_data/processed_chunks.jsonl`
- `pdf_rag_index/chunks.faiss`
- `pdf_rag_index/chunks_metadata.jsonl`
