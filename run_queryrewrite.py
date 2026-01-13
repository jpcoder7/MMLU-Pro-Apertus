import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.utils import make_table


# Normalize stop sequences into a list of strings
def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


# Load RAG documents from a jsonl file
def load_docs(path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


class RagQueryRewriteEvalModel(LM):
    """
    lm-eval wrapper for MMLU with RAG + query rewriting:
    - Rewrite the question into a search query before retrieval
    - Retrieval via FAISS + sentence-transformers
    - Uses HF model to compute loglikelihood for multiple-choice scoring
    """

    def __init__(
        self,
        pretrained: str,
        index_dir: str,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        max_ctx_chars: int = 2000,
        debug_first_n: int = 0,
        max_model_len: int = 8192,
        rewrite_enabled: bool = True,
        rewrite_max_new_tokens: int = 32,
        rewrite_temperature: float = 0.0,
        rewrite_top_p: float = 1.0,
    ):
        super().__init__()
        index_dir = Path(index_dir)

        # Load retrieval assets
        self.docs = load_docs(index_dir / "docs.jsonl")
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.embedder = SentenceTransformer(embed_model)

        self.top_k = int(top_k)
        self.max_ctx_chars = int(max_ctx_chars)
        self.debug_first_n = int(debug_first_n)
        self.debug_count = 0

        # Load model and tokenizer for scoring
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True, use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        self.model.eval()

        self.max_model_len = int(max_model_len)
        self.rewrite_enabled = bool(rewrite_enabled)
        self.rewrite_max_new_tokens = int(rewrite_max_new_tokens)
        self.rewrite_temperature = float(rewrite_temperature)
        self.rewrite_top_p = float(rewrite_top_p)
        self._rewrite_cache = {}

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self):
        return self.max_model_len

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    # Rewrite the question into a shorter search query
    def _rewrite_query(self, question: str) -> str:
        question_text = (question or "").strip()
        if not self.rewrite_enabled or not question_text:
            return question_text
        if question_text in self._rewrite_cache:
            return self._rewrite_cache[question_text]

        rewrite_prompt = (
            "Rewrite the following multiple-choice question into a short, precise search query. "
            "Return only the query text.\n\n"
            f"Question: {question_text}\n\nQuery:"
        )
        input_ids = self.tokenizer(
            rewrite_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=self.rewrite_max_new_tokens,
                do_sample=self.rewrite_temperature > 0.0,
                temperature=self.rewrite_temperature,
                top_p=self.rewrite_top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        rewritten = generated_text.strip().split("\n", 1)[0].strip()
        if not rewritten:
            rewritten = question_text
        self._rewrite_cache[question_text] = rewritten
        return rewritten

    # Retrieve top-k passages using FAISS + embeddings
    def _retrieve(self, raw_query: str) -> str:
        query_text = (raw_query or "").strip()
        if not query_text:
            return ""
        if self.top_k <= 0 or self.max_ctx_chars <= 0:
            return ""

        if len(query_text) > 5000:
            query_text = query_text[:5000]

        query_vec = self.embedder.encode([query_text], normalize_embeddings=True).astype("float32")
        _, indices = self.index.search(query_vec, self.top_k)

        chunks = []
        used_chars = 0
        for doc_index in indices[0]:
            doc = self.docs[int(doc_index)]
            title = str(doc.get("title", "")).strip()
            text = str(doc.get("text", "")).strip()
            if not text:
                continue

            chunk_text = (
                f"[{len(chunks)+1}] {title}\n{text}"
                if title
                else f"[{len(chunks)+1}]\n{text}"
            )
            remaining = self.max_ctx_chars - used_chars
            if remaining <= 0:
                break
            if len(chunk_text) > remaining:
                chunk_text = chunk_text[:remaining].rstrip()
            if not chunk_text:
                break
            chunks.append(chunk_text)
            used_chars += len(chunk_text) + 2

        return "\n\n".join(chunks)

    # Build the final prompt with retrieved context
    def _build_prompt(self, retrieved_context: str, prompt: str) -> str:
        return (
            "You are given Wikipedia excerpts to help answer a multiple-choice question.\n"
            "Use the excerpts if relevant, otherwise rely on your general knowledge.\n\n"
            "Wikipedia excerpts:\n"
            f"{retrieved_context}\n\n---\n\n"
            f"{prompt}"
        )

    # Loglikelihood for a single (prompt, continuation) pair
    def _loglikelihood_one(self, prompt: str, continuation: str) -> Tuple[float, bool]:
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        continuation_ids = self.tokenizer(
            continuation, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        input_ids = torch.cat([prompt_ids, continuation_ids], dim=1).to(self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        prompt_len = prompt_ids.shape[1]
        cont_len = continuation_ids.shape[1]

        total_logprob = 0.0
        is_greedy = True
        for i in range(cont_len):
            tok_id = continuation_ids[0, i].item()
            lp = log_probs[0, prompt_len + i - 1, tok_id].item()
            total_logprob += lp
            if tok_id != int(torch.argmax(log_probs[0, prompt_len + i - 1]).item()):
                is_greedy = False

        return total_logprob, is_greedy

    # Lm-eval scoring path used by MMLU (multiple choice)
    def loglikelihood(self, requests):
        results = []
        for r in requests:
            context = r.args[0] if len(r.args) > 0 else ""
            continuation = r.args[1] if len(r.args) > 1 else ""

            rewritten_query = self._rewrite_query(str(context))
            retrieved_context = self._retrieve(rewritten_query)
            full_prompt = self._build_prompt(retrieved_context, str(context))

            if self.debug_count < self.debug_first_n:
                self.debug_count += 1
                print("\n=== RAG+QR DEBUG ===")
                print(f"rewrite_enabled={self.rewrite_enabled} query={rewritten_query}")
                print(
                    f"ctx_chars={len(retrieved_context)} top_k={self.top_k} "
                    f"max_ctx_chars={self.max_ctx_chars}"
                )
                print("CTX (truncated):")
                print(retrieved_context[:1500] if retrieved_context else "<empty>")
                print("PROMPT (truncated):")
                print(str(context)[:500])
                print("=== END DEBUG ===\n")

            results.append(self._loglikelihood_one(full_prompt, str(continuation)))

        return results

    # Generic generation path for tasks using generate_until
    def generate_until(self, requests):
        outputs = []
        for r in requests:
            prompt_text = r.args[0] if len(r.args) > 0 else ""
            stop_sequences = r.args[1] if len(r.args) > 1 else []
            stop = ensure_list(stop_sequences)

            rewritten_query = self._rewrite_query(str(prompt_text))
            retrieved_context = self._retrieve(rewritten_query)
            full_prompt = self._build_prompt(retrieved_context, str(prompt_text))
            input_ids = self.tokenizer(
                full_prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=32,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                out[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            for s in stop:
                if s and s in generated_text:
                    generated_text = generated_text.split(s, 1)[0]
            outputs.append(generated_text)

        return outputs

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--pretrained", default="swiss-ai/Apertus-8B-Instruct-2509")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--num_fewshot", type=int, default=5)
    ap.add_argument("--tasks", default="mmlu_stem,mmlu_humanities,mmlu_social_sciences,mmlu_other")
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--log_samples", action="store_true")

    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_ctx_chars", type=int, default=2000)
    ap.add_argument("--debug_first_n", type=int, default=0)
    ap.add_argument("--max_model_len", type=int, default=8192)

    ap.add_argument("--rewrite_enabled", action="store_true")
    ap.add_argument("--rewrite_max_new_tokens", type=int, default=32)
    ap.add_argument("--rewrite_temperature", type=float, default=0.0)
    ap.add_argument("--rewrite_top_p", type=float, default=1.0)

    args = ap.parse_args()

    model = RagQueryRewriteEvalModel(
        pretrained=args.pretrained,
        index_dir=args.index_dir,
        embed_model=args.embed_model,
        top_k=args.top_k,
        max_ctx_chars=args.max_ctx_chars,
        debug_first_n=args.debug_first_n,
        max_model_len=args.max_model_len,
        rewrite_enabled=args.rewrite_enabled,
        rewrite_max_new_tokens=args.rewrite_max_new_tokens,
        rewrite_temperature=args.rewrite_temperature,
        rewrite_top_p=args.rewrite_top_p,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    res = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=None if args.limit == 0 else args.limit,
        log_samples=args.log_samples,
    )

    out = {
        "results": res.get("results", {}),
        "group_results": res.get("group_results", {}),
        "config": {
            "pretrained": args.pretrained,
            "index_dir": args.index_dir,
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
            "tasks": tasks,
            "batch_size": args.batch_size,
            "embed_model": args.embed_model,
            "top_k": args.top_k,
            "max_ctx_chars": args.max_ctx_chars,
            "debug_first_n": args.debug_first_n,
            "max_model_len": args.max_model_len,
            "rewrite_enabled": args.rewrite_enabled,
            "rewrite_max_new_tokens": args.rewrite_max_new_tokens,
            "rewrite_temperature": args.rewrite_temperature,
            "rewrite_top_p": args.rewrite_top_p,
        },
    }

    outp = Path(args.output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(make_table(res, column="results"))
    if "group_results" in res:
        print(make_table(res, column="group_results"))

    print("Results written to:", outp)


if __name__ == "__main__":
    main()
