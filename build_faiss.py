import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_docs", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs_path = out_dir / "docs.jsonl"
    index_path = out_dir / "index.faiss"

    # Embedding model
    model = SentenceTransformer(args.embed_model)
    dim = model.get_sentence_embedding_dimension()

    # FAISS index 
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200

    buffer_texts = []
    written = 0

    with open(args.corpus_jsonl, "r", encoding="utf-8") as fin, \
         open(docs_path, "w", encoding="utf-8") as fdocs:

        for i, line in enumerate(fin):
            if args.max_docs > 0 and written >= args.max_docs:
                break

            data = json.loads(line)
            text = data["text"]
            title = data.get("title", "")

            fdocs.write(json.dumps({
                "id": written,
                "title": title,
                "text": text
            }, ensure_ascii=False) + "\n")

            buffer_texts.append(text)
            written += 1

            if len(buffer_texts) == args.batch_size:
                vecs = model.encode(
                    buffer_texts,
                    batch_size=args.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False
                ).astype("float32")
                index.add(vecs)
                buffer_texts.clear()

        if buffer_texts:
            vecs = model.encode(
                buffer_texts,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype("float32")
            index.add(vecs)

    faiss.write_index(index, str(index_path))
    print(f"FAISS index written to {index_path}")
    print(f"Docs written to {docs_path} ({written} documents)")

if __name__ == "__main__":
    main()
