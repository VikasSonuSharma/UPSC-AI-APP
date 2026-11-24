#!/usr/bin/env python3
"""
ingest.py - RAG ingestion pipeline

    PDFs -> heading-aware chunks -> embeddings -> clusters -> stored index

Usage
-----
    # 1) Validate chunking first (no models needed beyond PyMuPDF):
    python ingest.py --pdf-dir ./pdfs --out-dir ./index --chunks-only

    # 2) Full pipeline:
    python ingest.py --pdf-dir ./pdfs --out-dir ./index

Outputs (in --out-dir)
----------------------
    outline.txt      heading tree of every PDF (check this against the real documents)
    chunks.jsonl     one chunk per line: text, heading path, pages, cluster id
    embeddings.npy   one normalized vector per chunk (same order as chunks.jsonl)
    clusters.json    cluster id -> size + top keywords
    chroma/          persistent Chroma collection "docs" (skip with --no-chroma)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    text: str
    size: float
    bold: bool
    page: int
    y0: float


@dataclass
class Section:
    doc: str
    path: list  # [doc_name, heading, subheading, ...]
    paras: list  # list of [text, page]


@dataclass
class Chunk:
    id: str
    doc: str
    path: list
    text: str
    page_start: int
    page_end: int
    n_tokens: int
    cluster: int = -1

    @property
    def embed_text(self) -> str:
        # Heading path is prepended so the vector carries the chunk's context.
        return " > ".join(self.path) + "\n" + self.text


# --------------------------------------------------------------------------- #
# 1. PDF extraction
# --------------------------------------------------------------------------- #
PAGE_NUM_RE = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)


def extract_lines(pdf_path: Path) -> list[Line]:
    """Read every text line with its font size / boldness / position."""
    import fitz  # PyMuPDF

    lines: list[Line] = []
    with fitz.open(pdf_path) as doc:
        n_pages = len(doc)
        for pno, page in enumerate(doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:  # skip images
                    continue
                for ln in block["lines"]:
                    spans = [s for s in ln["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = re.sub(r"\s+", " ", "".join(s["text"] for s in spans)).strip()
                    if not text or PAGE_NUM_RE.match(text):
                        continue
                    main = max(spans, key=lambda s: len(s["text"]))
                    bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
                    lines.append(
                        Line(text, round(main["size"] * 2) / 2, bool(bold), pno, ln["bbox"][1])
                    )
    return remove_repeated(lines, n_pages)


def remove_repeated(lines: list[Line], n_pages: int, frac: float = 0.5) -> list[Line]:
    """Drop running headers/footers: lines (digits ignored) that appear on >= half the pages."""
    if n_pages < 4:
        return lines
    norm = lambda t: re.sub(r"\d+", "#", t.lower())
    pages_seen: dict[str, set] = defaultdict(set)
    for l in lines:
        pages_seen[norm(l.text)].add(l.page)
    return [l for l in lines if len(pages_seen[norm(l.text)]) / n_pages < frac]


# --------------------------------------------------------------------------- #
# 2. Heading detection -> section tree
# --------------------------------------------------------------------------- #
def body_size(lines: list[Line]) -> float:
    c: Counter = Counter()
    for l in lines:
        c[l.size] += len(l.text)
    return c.most_common(1)[0][0]


def is_heading(l: Line, body: float) -> bool:
    words = len(l.text.split())
    if words > 15 or len(l.text) < 2:
        return False
    if l.size >= body + 1.0 or l.size >= body * 1.12:  # clearly larger than body text
        return True
    # bold, body-sized, short, and not a sentence fragment
    return l.bold and l.size >= body and not l.text.endswith((".", ",", ";"))


BULLET_RE = re.compile(r"^([\u2022\-*]|\d+[.)])\s")


def add_body(sec: Section, l: Line, prev: Line | None) -> None:
    if not sec.paras or prev is None:
        sec.paras.append([l.text, l.page])
        return
    new_para = (l.page == prev.page and l.y0 - prev.y0 > 1.6 * max(l.size, 1)) or bool(
        BULLET_RE.match(l.text)
    )
    if new_para:
        sec.paras.append([l.text, l.page])
        return
    last = sec.paras[-1][0]
    if last.endswith("-") and l.text[:1].islower():  # de-hyphenate line wraps
        sec.paras[-1][0] = last[:-1] + l.text
    else:
        sec.paras[-1][0] = last + " " + l.text


def build_sections(lines: list[Line], doc_name: str) -> list[Section]:
    if not lines:
        return []
    body = body_size(lines)
    heading_sizes = sorted({l.size for l in lines if is_heading(l, body)}, reverse=True)
    level_of = {s: i for i, s in enumerate(heading_sizes)}

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    cur = Section(doc_name, [doc_name], [])
    prev: Line | None = None
    i = 0
    while i < len(lines):
        l = lines[i]
        if is_heading(l, body):
            title, j = l.text, i + 1
            # join headings that wrap onto a second line
            while (
                j < len(lines)
                and is_heading(lines[j], body)
                and lines[j].size == l.size
                and lines[j].page == l.page
                and 0 < lines[j].y0 - lines[j - 1].y0 < 2.0 * l.size
            ):
                title += " " + lines[j].text
                j += 1
            if cur.paras:
                sections.append(cur)
            level = level_of[l.size]
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur = Section(doc_name, [doc_name] + [t for _, t in stack], [])
            prev = None
            i = j
            continue
        add_body(cur, l, prev)
        prev = l
        i += 1
    if cur.paras:
        sections.append(cur)
    return sections


# --------------------------------------------------------------------------- #
# 3. Chunking
# --------------------------------------------------------------------------- #
def n_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens per word). Good enough for sizing."""
    return int(len(text.split()) * 1.3)


def pack(items: list, size, max_tok: int) -> list[list]:
    out, cur, cur_t = [], [], 0
    for it in items:
        t = size(it)
        if cur and cur_t + t > max_tok:
            out.append(cur)
            cur, cur_t = [], 0
        cur.append(it)
        cur_t += t
    if cur:
        out.append(cur)
    return out


def split_text(paras: list, max_tok: int) -> list[tuple[str, int, int]]:
    """Pack paragraphs into <= max_tok pieces; oversized paragraphs split by sentence."""
    units: list[tuple[str, int]] = []
    for text, page in paras:
        if n_tokens(text) > max_tok:
            sents = re.split(r"(?<=[.!?])\s+", text)
            for g in pack(sents, n_tokens, max_tok):
                units.append((" ".join(g), page))
        else:
            units.append((text, page))
    out = []
    for g in pack(units, lambda u: n_tokens(u[0]), max_tok):
        out.append(("\n\n".join(t for t, _ in g), min(p for _, p in g), max(p for _, p in g)))
    return out


def common_prefix(a: list, b: list) -> list:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def merge_small(raw: list, max_tok: int, min_tok: int) -> list:
    """Merge tiny neighbouring chunks that share a parent heading so none are context-free."""
    out: list = []
    for path, text, ps, pe in raw:
        if out:
            pp, pt, pps, ppe = out[-1]
            related = pp == path or pp[:-1] == path[:-1] or pp == path[:-1] or path == pp[:-1]
            small = n_tokens(pt) < min_tok or n_tokens(text) < min_tok
            if (
                pp[0] == path[0]
                and related
                and small
                and n_tokens(pt) + n_tokens(text) <= max_tok
            ):
                newpath = common_prefix(pp, path)
                a = pt if pp == newpath else pp[-1] + "\n" + pt
                b = text if path == newpath else path[-1] + "\n" + text
                out[-1] = [newpath, a + "\n\n" + b, min(pps, ps), max(ppe, pe)]
                continue
        out.append([path, text, ps, pe])
    return out


def sections_to_chunks(sections: list[Section], max_tok: int, min_tok: int) -> list[Chunk]:
    raw = []
    for s in sections:
        for text, ps, pe in split_text(s.paras, max_tok):
            raw.append([s.path, text, ps, pe])
    merged = merge_small(raw, max_tok, min_tok)
    chunks = []
    counters: Counter = Counter()
    for path, text, ps, pe in merged:
        doc = path[0]
        chunks.append(
            Chunk(f"{doc}::{counters[doc]:04d}", doc, path, text, ps, pe, n_tokens(text))
        )
        counters[doc] += 1
    return chunks


def write_outline(all_sections: dict, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for doc, secs in all_sections.items():
            f.write(f"{doc}\n")
            seen = set()
            for s in secs:
                for depth in range(1, len(s.path)):
                    key = tuple(s.path[: depth + 1])
                    if key not in seen:
                        seen.add(key)
                        f.write("  " * depth + s.path[depth] + "\n")
            f.write("\n")


# --------------------------------------------------------------------------- #
# 4. Embeddings
# --------------------------------------------------------------------------- #
def embed(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(vecs, dtype="float32")


# --------------------------------------------------------------------------- #
# 5. Clustering
# --------------------------------------------------------------------------- #
def cluster(vecs: np.ndarray, min_cluster_size: int) -> tuple[np.ndarray, str]:
    """UMAP + HDBSCAN for real corpora; KMeans for very small ones or as a fallback."""
    from sklearn.cluster import KMeans

    n = len(vecs)
    if n < 4:
        return np.zeros(n, dtype=int), "single-cluster"

    def kmeans(k: int) -> np.ndarray:
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vecs)

    k_default = max(2, min(8, n // 5))
    if n < 40:
        return kmeans(k_default), "kmeans (too few chunks for UMAP/HDBSCAN)"

    from sklearn.cluster import HDBSCAN
    import umap

    reduced = umap.UMAP(
        n_components=min(10, n - 2),
        n_neighbors=min(15, n - 1),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(vecs)
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)
    if len(set(labels) - {-1}) < 2:
        return kmeans(k_default), "kmeans (HDBSCAN found <2 clusters)"
    return labels, "umap + hdbscan"


def cluster_keywords(chunks: list[Chunk], labels: np.ndarray, top: int = 6) -> dict:
    """Describe each cluster by its most distinctive terms (TF-IDF across clusters)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    ids = sorted(set(int(l) for l in labels))
    docs = [" ".join(c.text for c, l in zip(chunks, labels) if l == cid) for cid in ids]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for row, cid in enumerate(ids):
        weights = X[row].toarray().ravel()
        out[cid] = {
            "size": int(sum(1 for l in labels if l == cid)),
            "keywords": terms[weights.argsort()[::-1][:top]].tolist(),
        }
    return out


# --------------------------------------------------------------------------- #
# 6. Storage
# --------------------------------------------------------------------------- #
def store_chroma(out_dir: Path, chunks: list[Chunk], vecs: np.ndarray) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(out_dir / "chroma"))
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.create_collection("docs", metadata={"hnsw:space": "cosine"})
    B = 500
    for i in range(0, len(chunks), B):
        part = chunks[i : i + B]
        col.add(
            ids=[c.id for c in part],
            embeddings=vecs[i : i + B].tolist(),
            documents=[c.text for c in part],
            metadatas=[
                {
                    "doc": c.doc,
                    "path": " > ".join(c.path),
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "cluster": int(c.cluster),
                }
                for c in part
            ],
        )


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def print_chunk_stats(chunks: list[Chunk]) -> None:
    toks = [c.n_tokens for c in chunks]
    print(f"\n{len(chunks)} chunks | tokens min/median/max = "
          f"{min(toks)}/{int(statistics.median(toks))}/{max(toks)}")
    per_doc = Counter(c.doc for c in chunks)
    for doc, n in per_doc.items():
        print(f"  {doc}: {n} chunks")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="RAG ingestion: chunk -> embed -> cluster -> store")
    ap.add_argument("--pdf-dir", default=Path(r"D:\myrag\PDFs"), type=Path)
    ap.add_argument("--out-dir", default=Path("index"), type=Path)
    ap.add_argument("--max-tokens", type=int, default=600, help="max chunk size (approx tokens)")
    ap.add_argument("--min-tokens", type=int, default=80, help="chunks smaller than this get merged")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="sentence-transformers model")
    ap.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min cluster size")
    ap.add_argument("--chunks-only", action="store_true", help="stop after chunking (to inspect it)")
    ap.add_argument("--no-chroma", action="store_true", help="skip writing the Chroma collection")
    args = ap.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.pdf_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- extract + chunk ---------------------------------------------------- #
    all_sections: dict[str, list[Section]] = {}
    chunks: list[Chunk] = []
    for pdf in pdfs:
        print(f"Reading {pdf.name} ...")
        secs = build_sections(extract_lines(pdf), pdf.stem)
        all_sections[pdf.stem] = secs
        chunks.extend(sections_to_chunks(secs, args.max_tokens, args.min_tokens))

    write_outline(all_sections, args.out_dir / "outline.txt")
    print_chunk_stats(chunks)
    if args.chunks_only:
        write_chunks(chunks, args.out_dir / "chunks.jsonl")
        print(f"\nWrote outline.txt and chunks.jsonl to {args.out_dir}. "
              "Check the outline against your PDFs before embedding.")
        return

    # --- embed -------------------------------------------------------------- #
    print(f"\nEmbedding with {args.model} ...")
    vecs = embed([c.embed_text for c in chunks], args.model)

    # --- cluster ------------------------------------------------------------ #
    labels, method = cluster(vecs, args.min_cluster_size)
    for c, l in zip(chunks, labels):
        c.cluster = int(l)
    info = cluster_keywords(chunks, labels)
    print(f"\nClustering: {method}")
    for cid, d in info.items():
        name = "noise" if cid == -1 else f"cluster {cid}"
        print(f"  {name}: {d['size']} chunks | {', '.join(d['keywords'])}")

    # --- store -------------------------------------------------------------- #
    write_chunks(chunks, args.out_dir / "chunks.jsonl")
    np.save(args.out_dir / "embeddings.npy", vecs)
    with open(args.out_dir / "clusters.json", "w", encoding="utf-8") as f:
        json.dump({"method": method, "clusters": info}, f, indent=2)
    if not args.no_chroma:
        store_chroma(args.out_dir, chunks, vecs)
    print(f"\nDone. Index written to {args.out_dir}")


if __name__ == "__main__":
    main()
