import argparse
import json
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--max_docs", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.out_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            data = json.loads(line)
            text = re.sub(r"\s+", " ", data.get("text", "")).strip()
            title = re.sub(r"\s+", " ", data.get("title", "")).strip()

            if not text:
                continue

            record = {
                "id": str(count),
                "title": title,
                "text": text
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if args.max_docs > 0 and count >= args.max_docs:
                break

    print(f"Wrote {count} documents to {output_path}")

if __name__ == "__main__":
    main()
