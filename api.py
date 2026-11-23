"""
api.py - FastAPI app for the RAG pipeline (uses ask.py, so keep both files in the same folder)

Run:
    pip install fastapi uvicorn
    uvicorn api:app --reload

Then open:
    http://127.0.0.1:8000        simple chat page
    http://127.0.0.1:8000/docs   interactive API docs (Swagger)

Settings come from environment variables (all optional except the API key for your provider):
    GROQ_API_KEY       your Groq key (or OPENROUTER_API_KEY for openrouter)
    LLM_PROVIDER       groq (default) | openrouter | ollama | anthropic
    LLM_MODEL          overrides the provider's default model
    INDEX_DIR          folder written by ingest.py (default: index)
    EMBED_MODEL        must match ingest.py (default: BAAI/bge-small-en-v1.5)
    OLLAMA_HOST        default http://localhost:11434
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import ask  # reuses Retriever, build_prompt, generate from ask.py

INDEX_DIR = Path(os.environ.get("INDEX_DIR", "index"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "openrouter": "openrouter/free",
    "ollama": "qwen3:4b",
    "anthropic": "claude-sonnet-5-5",
}
# Decided here, so an old default inside ask.py can never be used by the web app.
LLM_MODEL = os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(LLM_PROVIDER)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "8000"))

state: dict = {}
search_lock = threading.Lock()  # one search at a time keeps the embedding model / Chroma safe


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model and open Chroma once, at startup (not on every request).
    state["retriever"] = ask.Retriever(INDEX_DIR, EMBED_MODEL)
    print(f"RAG app ready | provider={LLM_PROVIDER} | model={LLM_MODEL} | api.py v2", flush=True)
    yield
    state.clear()


app = FastAPI(title="Document RAG API", version="1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(4, ge=1, le=20, description="how many chunks to retrieve")
    doc: str | None = Field(None, description="restrict search to one document (file name without .pdf)")
    max_distance: float | None = Field(0.8, ge=0, le=2, description="drop chunks farther than this")
    cluster_filter: bool = Field(False, description="search only the cluster nearest to the question")
    llm_model: str | None = Field(None, description="override the LLM model for this request")


class Source(BaseModel):
    ref: int
    doc: str
    section: str
    pages: str
    distance: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    r = state["retriever"]
    return {"status": "ok", "chunks": r.col.count(), "provider": LLM_PROVIDER, "model": LLM_MODEL}


@app.get("/documents")
def documents() -> dict:
    """List the documents that were ingested."""
    metas = state["retriever"].col.get(include=["metadatas"])["metadatas"]
    return {"documents": sorted({m["doc"] for m in metas})}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest) -> AskResponse:
    retriever = state["retriever"]
    try:
        with search_lock:
            hits = retriever.search(
                req.question,
                top_k=req.top_k,
                doc=req.doc,
                cluster_filter=req.cluster_filter,
                max_distance=req.max_distance,
            )
    except Exception as e:  # e.g. unknown filter value
        raise HTTPException(status_code=400, detail=f"Search failed: {e}")

    hits = ask.fit_hits(hits, MAX_CONTEXT_CHARS)  # keep the prompt under free-tier token limits
    print(f"[ask] chunks sent={len(hits)} chars={sum(len(h['text']) for h in hits)}", flush=True)

    llm_args = SimpleNamespace(
        provider=LLM_PROVIDER,
        llm_model=req.llm_model or LLM_MODEL,
        api_key=None,  # read from the environment (GROQ_API_KEY etc.)
        ollama_host=OLLAMA_HOST,
    )
    try:
        answer = ask.generate(ask.build_prompt(req.question, hits), llm_args)
    except SystemExit as e:  # ask.py reports LLM problems via SystemExit; don't let it kill the server
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    sources = []
    for i, h in enumerate(hits, start=1):
        m = h["meta"]
        pages = str(m["page_start"]) if m["page_start"] == m["page_end"] else f"{m['page_start']}-{m['page_end']}"
        sources.append(Source(ref=i, doc=m["doc"], section=m["path"], pages=pages,
                              distance=round(h["distance"], 3), text=h["text"]))
    return AskResponse(answer=answer, sources=sources)


# --------------------------------------------------------------------------- #
# Minimal chat page
# --------------------------------------------------------------------------- #
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Document Q&amp;A</title>
<style>
  body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1b1b1b;background:#fafafa}
  h1{font-size:1.4rem} textarea{width:100%;box-sizing:border-box;padding:.7rem;font:inherit;border:1px solid #bbb;border-radius:8px;resize:vertical}
  button{margin-top:.6rem;padding:.6rem 1.2rem;border:0;border-radius:8px;background:#2b5fd9;color:#fff;font:inherit;cursor:pointer}
  button:disabled{opacity:.6;cursor:wait} .card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:1rem;margin-top:1rem}
  .answer{white-space:pre-wrap;line-height:1.5} .src{font-size:.9rem;margin-top:.6rem} .src summary{cursor:pointer}
  .src pre{white-space:pre-wrap;background:#f3f3f3;padding:.6rem;border-radius:6px;font-size:.85rem}
  .err{color:#b00020} .row{display:flex;gap:.6rem;align-items:center;margin-top:.5rem;flex-wrap:wrap}
  select{padding:.4rem;border-radius:6px;border:1px solid #bbb;font:inherit}
  @media (prefers-color-scheme: dark){body{background:#141414;color:#eee}.card{background:#1e1e1e;border-color:#333}
    textarea,select{background:#1e1e1e;color:#eee;border-color:#444}.src pre{background:#2a2a2a}}
</style></head><body>
<h1>Ask your documents</h1>
<textarea id="q" rows="3" placeholder="Type your question..."></textarea>
<div class="row"><label>Search in <select id="doc"><option value="">all documents</option></select></label>
<button id="go">Ask</button></div>
<div id="out"></div>
<script>
const out=document.getElementById('out'),btn=document.getElementById('go'),q=document.getElementById('q'),sel=document.getElementById('doc');
fetch('/documents').then(r=>r.json()).then(d=>d.documents.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;sel.appendChild(o);}));
async function ask(){
  const question=q.value.trim(); if(question.length<3) return;
  btn.disabled=true; btn.textContent='Thinking...'; out.replaceChildren();
  try{
    const res=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question,doc:sel.value||null})});
    const data=await res.json();
    const card=document.createElement('div'); card.className='card';
    if(!res.ok){card.className+=' err'; card.textContent=data.detail||'Something went wrong.'; out.appendChild(card); return;}
    const a=document.createElement('div'); a.className='answer'; a.textContent=data.answer; card.appendChild(a);
    if(data.sources.length){
      const h=document.createElement('div'); h.className='src'; h.innerHTML='<strong>Sources</strong>'; card.appendChild(h);
      data.sources.forEach(s=>{
        const d=document.createElement('details'); d.className='src';
        const sm=document.createElement('summary'); sm.textContent='['+s.ref+'] '+s.section+' (p.'+s.pages+', distance '+s.distance+')';
        const p=document.createElement('pre'); p.textContent=s.text; d.append(sm,p); card.appendChild(d);
      });
    }
    out.appendChild(card);
  }catch(e){const c=document.createElement('div'); c.className='card err'; c.textContent='Request failed: '+e; out.appendChild(c);}
  finally{btn.disabled=false; btn.textContent='Ask';}
}
btn.addEventListener('click',ask);
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)) ask();});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


if __name__ == "__main__":
    # Lets you start the server with:  python api.py
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
