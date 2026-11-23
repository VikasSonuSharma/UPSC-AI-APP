#!/usr/bin/env python3
"""
ask.py - RAG query pipeline

    question -> embed -> semantic search in Chroma -> build augmented prompt -> LLM -> answer + sources

Usage
-----
    # interactive chat loop (Groq API key, default):
    python ask.py

    # single question:
    python ask.py -q "What is the return time limit?"

    # other providers: --provider openrouter | ollama | anthropic

Run ingest.py first. The embedding model here MUST be the same one used in ingest.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SYSTEM_PROMPT = """You are a helpful assistant answering questions using a set of document excerpts \
(the CONTEXT) retrieved from the user's own documents, plus your own general knowledge.

Rules:
1. Base your answer primarily on the CONTEXT. Cite the excerpts you use like [1], [2].
2. If you add anything from your own general knowledge that is not in the CONTEXT, put it in a \
separate final paragraph starting with "From general knowledge:" so the user can tell the difference.
3. If the CONTEXT does not contain the answer, say so plainly before giving any general-knowledge answer.
4. Never invent facts about the documents. Do not cite an excerpt that does not support the claim."""


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self, index_dir: Path, model_name: str):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.index_dir = index_dir
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.col = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection("docs")
        self._centroids = None

    def embed_query(self, query: str) -> np.ndarray:
        if "bge" in self.model_name.lower() and "m3" not in self.model_name.lower():
            query = BGE_QUERY_PREFIX + query
        v = self.model.encode([query], normalize_embeddings=True)[0]
        return np.asarray(v, dtype="float32")

    def _load_centroids(self) -> dict:
        """Mean vector of each (non-noise) cluster, from the saved embeddings + chunk labels."""
        if self._centroids is None:
            vecs = np.load(self.index_dir / "embeddings.npy")
            with open(self.index_dir / "chunks.jsonl", encoding="utf-8") as f:
                clusters = np.array([json.loads(line)["cluster"] for line in f])
            cents = {}
            for cid in set(clusters.tolist()) - {-1}:
                c = vecs[clusters == cid].mean(axis=0)
                cents[cid] = c / (np.linalg.norm(c) + 1e-9)
            self._centroids = cents
        return self._centroids

    def nearest_cluster(self, qvec: np.ndarray) -> int | None:
        cents = self._load_centroids()
        if not cents:
            return None
        return max(cents, key=lambda cid: float(np.dot(cents[cid], qvec)))

    def search(self, query: str, top_k: int, doc: str | None = None,
               cluster_filter: bool = False, max_distance: float | None = None) -> list[dict]:
        qvec = self.embed_query(query)
        conds = []
        if doc:
            conds.append({"doc": doc})
        if cluster_filter:
            cid = self.nearest_cluster(qvec)
            if cid is not None:
                conds.append({"cluster": int(cid)})
        where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})

        res = self.col.query(
            query_embeddings=[qvec.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if max_distance is not None and dist > max_distance:
                continue
            hits.append({"text": text, "meta": meta, "distance": float(dist)})
        return hits


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def format_source(meta: dict) -> str:
    p0, p1 = meta["page_start"], meta["page_end"]
    pages = f"p.{p0}" if p0 == p1 else f"pp.{p0}-{p1}"
    return f"{meta['path']} ({pages})"


def fit_hits(hits: list[dict], max_chars: int = 8000) -> list[dict]:
    """Keep the best chunks until a character budget is used up (trims the last one).
    Guarantees the prompt can't blow past free-tier tokens-per-minute limits."""
    out, used = [], 0
    for h in hits:
        room = max_chars - used
        if room < 300:
            break
        text = h["text"]
        if len(text) > room:
            text = text[:room].rsplit(" ", 1)[0] + " ..."
        out.append({**h, "text": text})
        used += len(text)
    return out


def build_prompt(question: str, hits: list[dict]) -> str:
    if not hits:
        context = "(No relevant excerpts were found in the documents.)"
    else:
        blocks = [f"[{i}] Source: {format_source(h['meta'])}\n{h['text']}"
                  for i, h in enumerate(hits, start=1)]
        context = "\n\n---\n\n".join(blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}"


# --------------------------------------------------------------------------- #
# LLM providers
# --------------------------------------------------------------------------- #
def call_ollama(prompt: str, model: str, host: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            text = json.load(r)["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not reach Ollama at {host} ({e}).\n"
            f"Install Ollama, start it, and run:  ollama pull {model}"
        )


# Free-tier providers that speak the OpenAI chat-completions format.
# To add another one (Mistral, Cerebras, ...), add an entry here and in --provider choices.
OPENAI_COMPATIBLE = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
        "get_key": "https://console.groq.com/keys",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "openrouter/free",  # router that picks an available free model
        "get_key": "https://openrouter.ai/keys",
    },
}


def call_openai_compatible(prompt: str, model: str, provider: str, api_key: str | None) -> str:
    cfg = OPENAI_COMPATIBLE[provider]
    api_key = api_key or os.environ.get(cfg["key_env"])
    if not api_key:
        raise SystemExit(
            f"No API key found for {provider}.\n"
            f"1) Get a free key at {cfg['get_key']}\n"
            f"2) Set it:  PowerShell:  $env:{cfg['key_env']}=\"your_key\"   "
            f"|  Command Prompt:  set {cfg['key_env']}=your_key\n"
            f"   (or pass it with --api-key your_key)"
        )
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    # Cap the answer length. Groq counts the *reserved* answer space against the per-minute token
    # limit, and its reasoning models use max_completion_tokens (plain max_tokens may be ignored).
    payload["max_completion_tokens" if provider == "groq" else "max_tokens"] = 1500
    if provider == "groq" and model.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"  # less hidden reasoning = fewer tokens, faster answers
    req = urllib.request.Request(
        cfg["url"],
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "rag-ask/1.0",  # some providers block the default Python user agent
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            text = json.load(r)["choices"][0]["message"].get("content") or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if not text:
                raise SystemExit("The model returned an empty answer. Ask again or pick another --llm-model.")
            return text
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        hints = {
            401: "The API key is invalid or missing.",
            403: "Access denied - check the key and that the service is available in your region.",
            404: f"Model '{model}' not found. Try another with --llm-model.",
            413: "Request too big for the free tokens-per-minute limit. Lower top_k (e.g. 3) or wait a minute.",
            402: "This model needs credits. Use a free model (id ends in :free) or openrouter/free.",
            429: "Rate limit hit (free tiers allow few requests per minute/day). Wait and retry.",
        }
        raise SystemExit(f"{provider} API error {e.code}. {hints.get(e.code, '')}\n{body}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach {provider} ({e}). Check your internet connection.")


def call_anthropic(prompt: str, model: str) -> str:
    import anthropic  # pip install anthropic ; needs ANTHROPIC_API_KEY

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def generate(prompt: str, args: argparse.Namespace) -> str:
    if args.provider in OPENAI_COMPATIBLE:
        model = args.llm_model or OPENAI_COMPATIBLE[args.provider]["default_model"]
        return call_openai_compatible(prompt, model, args.provider, args.api_key)
    if args.provider == "anthropic":
        return call_anthropic(prompt, args.llm_model or "claude-sonnet-5-5")
    return call_ollama(prompt, args.llm_model or "qwen3:4b", args.ollama_host)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def answer(question: str, retriever: Retriever, args: argparse.Namespace) -> None:
    hits = retriever.search(
        question,
        top_k=args.top_k,
        doc=args.doc,
        cluster_filter=args.cluster_filter,
        max_distance=args.max_distance,
    )
    hits = fit_hits(hits, args.max_context_chars)
    if args.show_chunks:
        for i, h in enumerate(hits, start=1):
            print(f"\n--- chunk [{i}] distance={h['distance']:.3f} | {format_source(h['meta'])}")
            print(h["text"])
    print("\nThinking...\n")
    print(generate(build_prompt(question, hits), args))
    print("\nSources:")
    if not hits:
        print("  (nothing retrieved above the relevance threshold)")
    for i, h in enumerate(hits, start=1):
        print(f"  [{i}] {h['meta']['doc']} | {format_source(h['meta'])} | distance {h['distance']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask questions over your ingested documents")
    ap.add_argument("-q", "--question", help="single question (omit for interactive mode)")
    ap.add_argument("--index-dir", type=Path, default=Path("index"))
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5",
                    help="embedding model - must match ingest.py")
    ap.add_argument("--top-k", type=int, default=5, help="chunks to retrieve")
    ap.add_argument("--max-distance", type=float, default=0.8,
                    help="drop chunks farther than this cosine distance (lower = stricter)")
    ap.add_argument("--max-context-chars", type=int, default=8000,
                    help="max characters of retrieved text sent to the LLM")
    ap.add_argument("--doc", help="only search one document (its file name without .pdf)")
    ap.add_argument("--cluster-filter", action="store_true",
                    help="search only inside the cluster nearest to the question")
    ap.add_argument("--provider", choices=["groq", "openrouter", "ollama", "anthropic"], default="groq")
    ap.add_argument("--api-key", help="API key (or set GROQ_API_KEY / OPENROUTER_API_KEY)")
    ap.add_argument("--llm-model", help="override the provider's default model")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--show-chunks", action="store_true", help="print retrieved chunks (for debugging)")
    args = ap.parse_args()

    print("Loading embedding model and index ...")
    retriever = Retriever(args.index_dir, args.model)
    print(f"Index ready: {retriever.col.count()} chunks.")

    if args.question:
        answer(args.question, retriever, args)
        return
    print("Ask a question (type 'exit' to quit).")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", ""}:
            break
        answer(q, retriever, args)


if __name__ == "__main__":
    main()
