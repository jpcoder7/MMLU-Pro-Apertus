python - << 'PY'
import json
from datasets import load_dataset

wiki_data = "/workspace/rag/wiki/enwiki_20231101_preprocessed.jsonl"

wiki_stream = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)

with open(wiki_data, "w", encoding="utf-8") as f:
    for i, row in enumerate(wiki_stream):
        f.write(json.dumps({
            "id": str(i),
            "title": row.get("title", ""),
            "text": row.get("text", "")
        }, ensure_ascii=False) + "\n")
PY
