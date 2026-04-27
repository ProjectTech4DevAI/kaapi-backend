"""
Simple persistent RAG over every .json / .pdf file in this directory.

- Embeddings: gemini-embedding-2 (3072 dims, task-instruction prompts, per-item calls)
- Store: ChromaDB persistent client at ./chroma_store
  First run embeds + indexes; subsequent runs load from disk unless a source
  file changed (tracked via size+mtime in chroma_store/manifest.json) or
  --rebuild is passed.
- Citations:
    * PDF hits: page + paragraph (raw text)
    * JSON hits: url
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if (Path(_SCRIPT_DIR) / "google.py").exists():
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_SCRIPT_DIR)]

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import chromadb
import numpy as np
import pymupdf4llm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 3072
COLLECTION = "rag_corpus"
EMBED_WORKERS = 8
EMBED_MAX_RETRIES = 5
PARAGRAPH_PREVIEW_CHARS = 600

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_store"
MANIFEST_FILE = CHROMA_DIR / "manifest.json"

PDF_CHUNK_CHARS = 1200
PDF_CHUNK_OVERLAP = 150

_thread_local = threading.local()


def _is_pipeline_output_json(name: str) -> bool:
    stem = Path(name).stem
    return stem.endswith("_pipeline_output") or "_pipeline_output" in stem


def _client() -> genai.Client:
    # google-genai wraps a single httpx.Client that isn't safe across threads;
    # give each worker thread its own Client so concurrent calls don't collide.
    client = getattr(_thread_local, "client", None)
    if client is None:
        if not GEMINI_API_KEY:
            sys.exit("GEMINI_API_KEY not set")
        client = genai.Client(api_key=GEMINI_API_KEY)
        _thread_local.client = client
    return client


# ── Loaders ────────────────────────────────────────────────────────────

def _innovation_text(inn: dict) -> str:
    parts = []
    if inn.get("authors"):
        parts.append(f"Innovator(s): {inn['authors']}.")
    if inn.get("location"):
        parts.append(f"Location: {inn['location']}.")
    if inn.get("award_function"):
        parts.append(f"Award: {inn['award_function']}.")
    if inn.get("description"):
        parts.append(inn["description"])
    return " ".join(parts).strip()


def _coerce_json_record(entry, i: int, source: str) -> dict | None:
    if isinstance(entry, dict):
        title = (entry.get("title") or "").strip()
        body = _innovation_text(entry) or title
        if not body and not title:
            # Fallback: concatenate string-valued fields.
            body = " ".join(
                f"{k}: {v}" for k, v in entry.items() if isinstance(v, str) and v.strip()
            ).strip()
        if not body:
            return None
        return {
            "id": f"{source}_{i}",
            "title": title or "none",
            "text": body,
            "metadata": {
                "source": source,
                "type": "json",
                "file": f"{source}.json",
                "title": title,
                "authors": entry.get("authors", "") or "",
                "location": entry.get("location", "") or "",
                "award_function": entry.get("award_function", "") or "",
                "url": entry.get("url", "") or "",
                "image_url": entry.get("image_url", "") or "",
            },
        }
    if isinstance(entry, str) and entry.strip():
        return {
            "id": f"{source}_{i}",
            "title": "none",
            "text": entry.strip(),
            "metadata": {"source": source, "type": "json", "file": f"{source}.json", "url": ""},
        }
    return None


def load_json_file(path: Path) -> list[dict]:
    source = path.stem
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] {path.name}: failed to parse ({e}), skipping")
        return []

    entries = data if isinstance(data, list) else [data]
    items: list[dict] = []
    for i, entry in enumerate(entries):
        rec = _coerce_json_record(entry, i, source)
        if rec is not None:
            items.append(rec)
    print(f"[*] {path.name}: {len(items)} records")
    return items


_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,
)
_RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=PDF_CHUNK_CHARS,
    chunk_overlap=PDF_CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
)


def _clean_pdf_md(md: str) -> str:
    md = md.replace("==> picture", "").replace("intentionally omitted <==", "")
    md = re.sub(r"\*\*\s*\[\s*\d+\s*x\s*\d+\s*\]\s*\*\*", "", md)
    md = re.sub(r"-----\s*(Start|End) of picture text\s*-----", "", md, flags=re.IGNORECASE)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _strip_md(text: str) -> str:
    # Light markdown stripping for citation previews (preserve the words themselves).
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[!] {path.name} not found, skipping")
        return []

    source = path.stem
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    items: list[dict] = []
    global_idx = 0
    for page in pages:
        page_num = page.get("metadata", {}).get("page_number") or 0
        md = _clean_pdf_md(page.get("text", ""))
        if not md:
            continue

        header_docs = _HEADER_SPLITTER.split_text(md)
        if not header_docs:
            header_docs = [type("D", (), {"page_content": md, "metadata": {}})()]

        for hd in header_docs:
            content = hd.page_content if hasattr(hd, "page_content") else hd["page_content"]
            hmeta = hd.metadata if hasattr(hd, "metadata") else hd.get("metadata", {})
            for piece in _RECURSIVE_SPLITTER.split_text(content):
                piece = piece.strip()
                if not piece:
                    continue
                heading_path = " › ".join(
                    v for k in ("h1", "h2", "h3") if (v := hmeta.get(k))
                ) if isinstance(hmeta, dict) else ""
                clean_heading = _strip_md(heading_path) if heading_path else ""
                clean_paragraph = _strip_md(piece)
                preview = clean_paragraph[:PARAGRAPH_PREVIEW_CHARS]
                if len(clean_paragraph) > PARAGRAPH_PREVIEW_CHARS:
                    preview += "…"
                items.append({
                    "id": f"{source}_{global_idx:04d}",
                    "title": clean_heading or "none",
                    "text": piece,
                    "metadata": {
                        "source": source,
                        "type": "pdf",
                        "page": page_num,
                        "heading": clean_heading,
                        "file": path.name,
                        "paragraph": preview,
                    },
                })
                global_idx += 1

    print(f"[*] {path.name}: {len(items)} chunks (pymupdf4llm + MarkdownHeader+Recursive)")
    return items


# ── Embedding (gemini-embedding-2, per-item, concurrent) ───────────────

def _format_doc(title: str, content: str) -> str:
    t = (title or "none").strip() or "none"
    return f"title: {t} | text: {content.strip()}"


def _format_query(content: str) -> str:
    return f"task: search result | query: {content.strip()}"


def _normalize(vec):
    arr = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n > 0:
        arr = arr / n
    return arr.tolist()


def _embed_one(content: str) -> list[float]:
    delay = 1.0
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            result = _client().models.embed_content(
                model=EMBEDDING_MODEL,
                contents=content,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
            )
            [emb] = result.embeddings
            return _normalize(emb.values)
        except Exception as e:
            msg = str(e).lower()
            transient = any(s in msg for s in ("429", "rate", "503", "unavailable", "timeout", "temporarily"))
            if attempt == EMBED_MAX_RETRIES - 1 or not transient:
                raise
            sleep_for = delay + random.random() * 0.5
            time.sleep(sleep_for)
            delay = min(delay * 2, 30.0)


def embed_documents(records: list[dict]) -> list[list[float]]:
    inputs = [_format_doc(r.get("title"), r["text"]) for r in records]
    results: list[list[float] | None] = [None] * len(inputs)
    done = 0
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = {pool.submit(_embed_one, inp): idx for idx, inp in enumerate(inputs)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(inputs):
                print(f"    embedded {done}/{len(inputs)}", flush=True)
    return [r for r in results if r is not None]


def embed_query(query: str) -> list[float]:
    return _embed_one(_format_query(query))


# ── Vector Store (in-memory) ───────────────────────────────────────────

def _chroma_client() -> chromadb.api.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _manifest_for(files: list[Path]) -> dict:
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "files": {
            p.name: {"size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}
            for p in sorted(files)
        },
    }


def _normalized_manifest(manifest: dict | None) -> dict | None:
    if not isinstance(manifest, dict):
        return None
    files = manifest.get("files")
    if not isinstance(files, dict):
        return manifest
    clean_files = {
        name: meta for name, meta in files.items()
        if not _is_pipeline_output_json(name)
    }
    return {
        "embedding_model": manifest.get("embedding_model"),
        "embedding_dim": manifest.get("embedding_dim"),
        "files": clean_files,
    }


def _manifest_diff_msg(want: dict, have: dict | None) -> str:
    if have is None:
        return "manifest missing/unreadable"
    if want.get("embedding_model") != have.get("embedding_model"):
        return (
            f"embedding_model changed "
            f"({have.get('embedding_model')} -> {want.get('embedding_model')})"
        )
    if want.get("embedding_dim") != have.get("embedding_dim"):
        return f"embedding_dim changed ({have.get('embedding_dim')} -> {want.get('embedding_dim')})"

    wf = want.get("files", {})
    hf = have.get("files", {})
    missing = sorted(set(wf) - set(hf))
    extra = sorted(set(hf) - set(wf))
    changed = sorted(k for k in set(wf).intersection(hf) if wf[k] != hf[k])

    parts = []
    if missing:
        parts.append(f"missing_in_manifest={missing[:3]}")
    if extra:
        parts.append(f"extra_in_manifest={extra[:3]}")
    if changed:
        parts.append(f"changed={changed[:3]}")
    return "; ".join(parts) if parts else "unknown manifest mismatch"


def _load_manifest() -> dict | None:
    if not MANIFEST_FILE.exists():
        return None
    try:
        with MANIFEST_FILE.open() as f:
            return json.load(f)
    except Exception:
        return None


def _save_manifest(manifest: dict) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_FILE.open("w") as f:
        json.dump(manifest, f, indent=2)


def _build_fresh(records: list[dict]) -> chromadb.Collection:
    client = _chroma_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[*] Embedding {len(records)} documents with {EMBEDDING_MODEL} ({EMBEDDING_DIM}d)...")
    embeddings = embed_documents(records)

    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    metas = [r["metadata"] for r in records]
    for i in range(0, len(records), 200):
        j = min(i + 200, len(records))
        collection.add(
            ids=ids[i:j],
            embeddings=embeddings[i:j],
            documents=texts[i:j],
            metadatas=metas[i:j],
        )
    print(f"[*] Vector store ready: {collection.count()} documents indexed at {CHROMA_DIR}")
    return collection


def get_or_build_store(
    source_files: list[Path],
    records_fn,
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """Load the persistent store if sources are unchanged, otherwise rebuild."""
    want = _manifest_for(source_files)
    have = _normalized_manifest(_load_manifest())
    needs_rebuild = force_rebuild or want != have
    if needs_rebuild:
        reason = "force_rebuild=true" if force_rebuild else _manifest_diff_msg(want, have)
        print(f"[*] Rebuild required: {reason}")

    if not needs_rebuild:
        try:
            client = _chroma_client()
            collection = client.get_collection(name=COLLECTION, embedding_function=None)
            if collection.count() == 0:
                print("[!] Existing collection is empty despite matching manifest; rebuilding.")
                needs_rebuild = True
            else:
                print(
                    f"[*] Loaded persistent store from {CHROMA_DIR} "
                    f"({collection.count()} documents, sources unchanged)."
                )
                return collection
        except Exception as e:
            print(f"[!] Existing store could not be loaded ({e}); rebuilding.")

    records = records_fn()
    if not records:
        sys.exit("No documents loaded.")
    collection = _build_fresh(records)
    _save_manifest(want)
    return collection


def query_store(collection: chromadb.Collection, query: str, top_k: int = 5):
    qemb = embed_query(query)
    res = collection.query(
        query_embeddings=[qemb],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "similarity": round(1.0 - dist, 4),
            "document": doc,
            "metadata": meta,
        })
    return hits


# ── CLI ────────────────────────────────────────────────────────────────

def _format_hit(i: int, hit: dict) -> str:
    meta = hit["metadata"]
    src = meta.get("source", "?")
    lines = [f"── [{i}] similarity={hit['similarity']}  source={src}"]
    if meta.get("type") == "pdf":
        page = meta.get("page", "?")
        heading = meta.get("heading") or "(no heading)"
        lines.append(f"   Citation: {meta.get('file', '')} — p.{page}  §{heading}")
        paragraph = meta.get("paragraph") or _strip_md(hit["document"])[:PARAGRAPH_PREVIEW_CHARS]
        lines.append(f"   Paragraph: {paragraph}")
    else:
        title = meta.get("title", "") or "(untitled)"
        lines.append(f"   Citation: {title}")
        if meta.get("authors"):
            lines.append(f"   Authors : {meta['authors']}")
        if meta.get("location"):
            lines.append(f"   Location: {meta['location']}")
        if meta.get("award_function"):
            lines.append(f"   Award   : {meta['award_function']}")
        if meta.get("url"):
            lines.append(f"   URL     : {meta['url']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Simple persistent RAG (Gemini embeddings)")
    parser.add_argument("query", nargs="?", help="Query text. If omitted, enters interactive mode.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-chunking and re-embedding even if source files are unchanged.",
    )
    args = parser.parse_args()

    json_files = sorted(
        p for p in BASE_DIR.glob("*.json")
        if not _is_pipeline_output_json(p.name)
    )
    pdf_files = sorted(BASE_DIR.glob("*.pdf"))
    print(f"[*] Discovered {len(json_files)} JSON file(s) and {len(pdf_files)} PDF file(s) in {BASE_DIR}")
    source_files = list(json_files) + list(pdf_files)
    if not source_files:
        sys.exit("No .json or .pdf files found.")

    def build_records() -> list[dict]:
        if not GEMINI_API_KEY:
            sys.exit("GEMINI_API_KEY not set")
        records: list[dict] = []
        for p in json_files:
            records.extend(load_json_file(p))
        for p in pdf_files:
            records.extend(load_pdf(p))
        return records

    collection = get_or_build_store(source_files, build_records, force_rebuild=args.rebuild)

    if not GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY not set (needed for querying)")

    if args.query:
        queries = [args.query]
    else:
        print("\nEnter query (empty line to quit):")
        queries = []
        try:
            while True:
                q = input("> ").strip()
                if not q:
                    break
                queries.append(q)
                hits = query_store(collection, q, top_k=args.top_k)
                print()
                for i, h in enumerate(hits, 1):
                    print(_format_hit(i, h))
                    print()
        except (EOFError, KeyboardInterrupt):
            print()
        return

    for q in queries:
        print(f"\n=== Query: {q}")
        hits = query_store(collection, q, top_k=args.top_k)
        for i, h in enumerate(hits, 1):
            print(_format_hit(i, h))
            print()


if __name__ == "__main__":
    main()
