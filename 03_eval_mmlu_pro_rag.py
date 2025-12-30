"""
touch /workspace/rag/03_eval_mmlu_pro_rag.py
nano /workspace/rag/03_eval_mmlu_pro_rag.py

"""

import argparse
import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.utils import make_table

CHOICE_RE = re.compile(r"\b([A-E])\b")

def load_docs(path: Path):
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs

class RagVllmLM(LM):
    def __init__(
        self,
        pretrained: str,
        index_dir: str,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        max_ctx_chars: int = 2000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
    ):
        super().__init__()
        index_dir = Path(index_dir)

        self.docs = load_docs(index_dir / "docs.jsonl")
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.embedder = SentenceTransformer(embed_model)

        self.top_k = top_k
        self.max_ctx_chars = max_ctx_chars

        self.llm = LLM(
            model=pretrained,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self):
        return 8192

    def tok_encode(self, string: str):
        raise NotImplementedError

    def tok_decode(self, tokens):
        raise NotImplementedError

    def _retrieve(self, query: str) -> str:
        qv = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        _, idxs = self.index.search(qv, self.top_k)

        chunks = []
        used = 0
        for i in idxs[0]:
            d = self.docs[int(i)]
            piece = f"{d['title']}\n{d['text']}"
            if used + len(piece) > self.max_ctx_chars:
                break
            chunks.append(piece)
            used += len(piece)

        return "\n\n".join(chunks)

    def generate_until(self, requests):
        outputs = []

        for r in requests:
            prompt, until, gen_kwargs = r.args
            ctx = self._retrieve(prompt)

            full_prompt = (
                "You are given Wikipedia excerpts.\n"
                "Answer the multiple-choice question.\n"
                "IMPORTANT: Reply with ONLY one letter: A, B, C, D, or E.\n\n"
                "Wikipedia excerpts:\n"
                f"{ctx}\n\n---\n\n"
                f"{prompt}"
            )

            sp = SamplingParams(
                temperature=0.0,
                max_tokens=16,
                stop=until,
            )

            out = self.llm.generate([full_prompt], sp)[0].outputs[0].text
            m = CHOICE_RE.search(out)
            outputs.append(m.group(1) if m else out.strip()[:1])

        return outputs

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=str, required=True)
    ap.add_argument("--pretrained", type=str, default="swiss-ai/Apertus-8B-2509")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--log_samples", action="store_true")
    args = ap.parse_args()

    model = RagVllmLM(
        pretrained=args.pretrained,
        index_dir=args.index_dir,
    )

    print(
        f"vllm (pretrained={args.pretrained}), "
        f"RAG index={args.index_dir}, limit={args.limit}"
    )

    res = evaluator.simple_evaluate(
        model=model,
        tasks=["mmlu_pro"],
        num_fewshot=0,
        batch_size="auto",
        limit=args.limit,
        log_samples=args.log_samples,
    )

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(make_table(res, column="results"))
    if "group_results" in res:
        print(make_table(res, column="group_results"))

    print("Results written to:", out)

if __name__ == "__main__":
    main()
