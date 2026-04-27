"""
Inquilab golden-dataset triage pipeline.

Runs three checks over every row of the 200-row golden dataset and writes a
single Excel with input columns + pipeline verdict columns:

  1. Duplicate detection   — cosine similarity vs. persistent chroma_store
                             corpus from rag.py. verdict = "duplicate" if
                             top-1 similarity >= DUP_THRESHOLD, else "unique".
  2. GPT-pasted detection  — indic_ai_detector on Problem and Solution
                             independently (never rejects, pass-through).
  3. Irrelevant PSI        — LLM call with the topic-relevance prompt. Emits
                             independent bools for theme alignment and
                             cross-field coherence.

CLI:
  python pipeline.py                           # defaults: gemini-2.5-flash
  python pipeline.py --limit 5                 # smoke test
  python pipeline.py --provider openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "gpt_content_detection"))

load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env")

import rag
import llm_client
from prompt import (
    get_evaluation_method,
    get_evaluation_objective,
    get_feedback_instructions,
    get_few_shot_examples,
    get_scoring_criteria,
    get_system_role,
    get_topic_relevance_system_prompt,
    get_topic_relevance_user_prompt,
    get_topic_relevance_output_schema,
)
from indic_ai_detector import IndicAIDetector

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DUP_THRESHOLD = 0.90
MAX_DOC_IMAGES = 3
METRICS = ["Novelty", "Usefulness", "Feasibility", "Scalability", "Sustainability"]


# ── Helpers ────────────────────────────────────────────────────────────────

def _to_direct_url(url: str) -> str:
    """Convert a Google Drive share link to a direct-content URL."""
    url = url.strip()
    file_id = None

    m = re.match(r"https://drive\.google\.com/file/d/([^/]+)", url)
    if m:
        file_id = m.group(1)

    if not file_id:
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
        if m and ("drive.google.com" in url or "drive.usercontent.google.com" in url):
            file_id = m.group(1)

    if file_id:
        return f"https://lh3.googleusercontent.com/d/{file_id}"
    return url


def _parse_documents(val: Any) -> list[str]:
    """Split the Documents cell into direct URLs. Handles NaN, commas, stray whitespace."""
    if pd.isna(val):
        return []
    parts = [p.strip() for p in str(val).split(",")]
    return [_to_direct_url(p) for p in parts if p]


def _clean_text(val: Any) -> str:
    return "" if pd.isna(val) else str(val).strip()


def _safe_str(val: Any) -> str:
    return "" if pd.isna(val) else str(val)


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def _safe_dir(name: str) -> str:
    """Sanitize a string for use as a single directory segment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    return cleaned or "unknown"


# ── Corpus loader (reuse rag.py) ────────────────────────────────────────────

def _load_duplicate_corpus(output_file: str | None = None):
    """Load or build the persistent chroma_store corpus via rag.py.

    rag.py discovers .json / .pdf files in inquilab/, chunks + embeds, and
    caches to chroma_store/ keyed by source-file manifest. Subsequent runs
    load from disk unless sources change.
    """
    # Exclude pipeline-generated JSON outputs from the duplicate corpus.
    excluded_json_names = set()
    if output_file:
        excluded_json_names.add(Path(output_file).with_suffix(".json").name)

    json_files = sorted(
        p for p in BASE_DIR.glob("*.json")
        if p.name not in excluded_json_names
        and "_pipeline_output" not in p.stem
    )
    pdf_files = sorted(BASE_DIR.glob("*.pdf"))
    source_files = list(json_files) + list(pdf_files)
    if not source_files:
        sys.exit(
            "[pipeline] No .json or .pdf files in inquilab/ — "
            "drop your innovation corpus files there before running."
        )

    def build_records() -> list[dict]:
        records: list[dict] = []
        for p in json_files:
            records.extend(rag.load_json_file(p))
        for p in pdf_files:
            records.extend(rag.load_pdf(p))
        return records

    return rag.get_or_build_store(source_files, build_records, force_rebuild=False)


# ── Per-row checks ──────────────────────────────────────────────────────────

def _duplicate_check(title: str, problem: str, solution: str, collection) -> dict:
    query_text = f"Title: {title}. Problem: {problem}. Solution: {solution}".strip()
    if not query_text or query_text == "Title: . Problem: . Solution:":
        return {
            "dup_verdict": "unique",
            "dup_similarity": None,
            "dup_match": "",
        }
    try:
        hits = rag.query_store(collection, query_text, top_k=1)
    except Exception as e:
        logger.warning(f"[_duplicate_check] failed: {e}")
        return {
            "dup_verdict": "error",
            "dup_similarity": None,
            "dup_match": "",
        }

    if not hits:
        return {
            "dup_verdict": "unique",
            "dup_similarity": None,
            "dup_match": "",
        }

    top = hits[0]
    sim = float(top["similarity"])
    meta = top.get("metadata") or {}
    source_type = (meta.get("type") or "").lower()
    if source_type == "pdf":
        citation = f"{meta.get('file', '')} — p.{meta.get('page', '?')}"
        heading = meta.get("heading")
        if heading:
            citation = f"{citation}  §{heading}"
        match_obj = {
            "source": "pdf",
            "citation": citation,
        }
    else:
        match_obj = {
            "source": "json",
            "citation": meta.get("title", "") or "",
            "url": meta.get("url", "") or "",
        }

    return {
        "dup_verdict": "duplicate" if sim >= DUP_THRESHOLD else "unique",
        "dup_similarity": round(sim, 4),
        "dup_match": _json_text(match_obj),
    }


def _gpt_detect(text: str, detector: IndicAIDetector, lock: threading.Lock, field: str) -> dict:
    """Run IndicAIDetector on a single text. Returns verdict / confidence / p_ai."""
    if not text.strip():
        return {
            f"{field}_ai_verdict": "uncertain",
            f"{field}_ai_confidence": "low",
            f"{field}_ai_prob": None,
        }
    with lock:
        result = detector.detect(text)
    p_ai = result.class_probs.get("ai", result.class_probs.get("gpt", None))
    return {
        f"{field}_ai_verdict": result.verdict,
        f"{field}_ai_confidence": result.confidence,
        f"{field}_ai_prob": round(float(p_ai), 4) if p_ai is not None else None,
    }


@lru_cache(maxsize=1)
def _evaluate_system_prompt() -> str:
    """Single combined system prompt: topic-relevance gate + scoring + feedback."""
    schema = {
        "task": {
            "role": get_system_role(),
            "objective": get_evaluation_objective(),
            "method": get_evaluation_method(),
        },
        "schema": {
            "name": "Evaluation Schema",
            "parameters": get_scoring_criteria(),
        },
        "output_rules": {
            "format": "Return ONLY valid JSON object",
            "required_keys": [
                "problem_aligned", "solution_aligned", "document_aligned",
                "pipeline_verdict", "reason",
                *METRICS, "Attachment_Summary", "Idea_Feedback",
            ],
            "structure": {
                m: {"score": "1-10", "reason": "string (one line why this score)"}
                for m in METRICS
            },
        },
    }
    examples = {"name": "Example Dataset", "list": get_few_shot_examples()}
    return (
        "# Topic Relevance Gate (judge first)\n"
        f"{get_topic_relevance_system_prompt()}\n\n"
        "# Evaluation System Schema\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "# Few-Shot Examples\n"
        f"{json.dumps(examples, indent=2, ensure_ascii=False)}\n\n"
        f"{get_feedback_instructions()}\n\n"
        "# Combined Output Rules\n"
        "Return a SINGLE JSON object containing BOTH sections in the same response:\n"
        "  (1) Topic relevance fields: problem_aligned, solution_aligned, document_aligned, pipeline_verdict, reason.\n"
        "  (2) Scoring + feedback fields: Novelty/Usefulness/Feasibility/Scalability/Sustainability "
        "(each {score, reason}), Attachment_Summary, Idea_Feedback.\n"
        "Always produce all fields. If pipeline_verdict = REJECT, still emit the rubric scores and feedback — "
        "downstream code uses pipeline_verdict to decide whether to publish them."
    )


def _evaluate_user_prompt(
    theme: str,
    problem: str,
    solution: str,
    title: str | None,
    has_documents: bool,
) -> str:
    relevance_text = get_topic_relevance_user_prompt(
        theme, problem, solution, title=title, has_documents=has_documents,
    )
    return (
        f"{relevance_text}\n\n"
        "After judging the topic-relevance fields above, also score this submission across "
        "Novelty, Usefulness, Feasibility, Scalability, and Sustainability, and generate Idea_Feedback. "
        "IMPORTANT: Idea_Feedback MUST be written in the SAME language as the student's problem/solution text "
        "(Hindi → Hindi, Telugu → Telugu, English → English; mixed → dominant language of the solution). "
        "If attachments are present, summarize what they show in Attachment_Summary. "
        "Respond in the required JSON format."
    )


@lru_cache(maxsize=1)
def _evaluate_schema() -> dict:
    relevance_props = get_topic_relevance_output_schema()["properties"]
    score_props = {
        m: {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        }
        for m in METRICS
    }
    return {
        "type": "object",
        "properties": {
            **relevance_props,
            **score_props,
            "Attachment_Summary": {"type": "string"},
            "Idea_Feedback": {"type": "string"},
        },
        "required": [
            "problem_aligned", "solution_aligned", "document_aligned",
            "pipeline_verdict", "reason",
            *METRICS, "Attachment_Summary", "Idea_Feedback",
        ],
        "additionalProperties": False,
    }


def _evaluate(
    theme: str,
    problem: str,
    solution: str,
    doc_urls: list[str],
    provider: str,
    model: str,
    api_key: str,
    temperature: float,
    title: str | None = None,
) -> dict:
    """Single LLM call: topic relevance + scoring + feedback in one response."""
    has_documents = bool(doc_urls)
    image_urls = doc_urls[:MAX_DOC_IMAGES] if doc_urls else None
    call_fn = llm_client.call_openai if provider == "openai" else llm_client.call_gemini

    try:
        raw = call_fn(
            api_key=api_key,
            model=model,
            system_prompt=_evaluate_system_prompt(),
            user_text=_evaluate_user_prompt(theme, problem, solution, title, has_documents),
            image_urls=image_urls,
            output_schema=_evaluate_schema(),
            temperature=temperature,
        )
    except Exception as e:
        logger.warning(f"[_evaluate] LLM call failed: {e}")
        raw = {}

    def _score_and_reason(metric: str) -> tuple[Any, str]:
        obj = raw.get(metric)
        if isinstance(obj, dict):
            return obj.get("score"), obj.get("reason", "")
        return None, ""

    novelty_score, novelty_reason = _score_and_reason("Novelty")
    usefulness_score, usefulness_reason = _score_and_reason("Usefulness")
    feasibility_score, feasibility_reason = _score_and_reason("Feasibility")
    scalability_score, scalability_reason = _score_and_reason("Scalability")
    sustainability_score, sustainability_reason = _score_and_reason("Sustainability")

    return {
        "problem_aligned": raw.get("problem_aligned"),
        "solution_aligned": raw.get("solution_aligned"),
        "document_aligned": raw.get("document_aligned"),
        "relevance_verdict": raw.get("pipeline_verdict"),
        "relevance_reason": raw.get("reason", ""),
        "Novelty": novelty_score,
        "Novelty Reason": novelty_reason,
        "Usefulness": usefulness_score,
        "Usefulness Reason": usefulness_reason,
        "Feasibility": feasibility_score,
        "Feasibility Reason": feasibility_reason,
        "Scalability": scalability_score,
        "Scalability Reason": scalability_reason,
        "Sustainability": sustainability_score,
        "Sustainability Reason": sustainability_reason,
        "Attachment Summary": raw.get("Attachment_Summary", ""),
        "Idea Feedback": raw.get("Idea_Feedback", ""),
    }


def _final_verdict(dup: dict, evaluation: dict) -> dict:
    """Combine gating signals into an overall pipeline verdict + reason.

    Duplicate hit overrides everything. Otherwise we trust the LLM's relevance
    verdict directly and surface its reason. GPT-detection is pass-through
    (metadata only) and never blocks.
    """
    if dup["dup_verdict"] == "duplicate":
        return {"pipeline_verdict": "REJECT", "pipeline_reason": "duplicate of existing submission"}
    verdict = evaluation.get("relevance_verdict")
    reason = evaluation.get("relevance_reason", "") or ""
    if verdict in ("ACCEPT", "REJECT", "REVIEW"):
        return {"pipeline_verdict": verdict, "pipeline_reason": reason}
    return {"pipeline_verdict": "REVIEW", "pipeline_reason": reason or "relevance check failed"}


# ── Row processor ──────────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "dup_verdict", "dup_similarity", "dup_match",
    "problem_ai_verdict", "problem_ai_prob",
    "solution_ai_verdict", "solution_ai_prob",
    "problem_aligned", "solution_aligned", "document_aligned",
    "pipeline_verdict", "pipeline_reason",
    "Novelty", "Novelty Reason",
    "Usefulness", "Usefulness Reason",
    "Feasibility", "Feasibility Reason",
    "Scalability", "Scalability Reason",
    "Sustainability", "Sustainability Reason",
    "Attachment Summary", "Idea Feedback",
]
LEGACY_OUTPUT_COLUMNS = [
    "dup_top_match_title", "dup_top_match_url",
    "ai_detection", "relevance",
    "problem_ai_confidence", "solution_ai_confidence",
    "problem_on_theme", "solution_on_theme", "documents_on_theme",
    "problem_solution_aligned", "problem_documents_aligned", "solution_documents_aligned",
    "overall_relevant", "relevance_reason", "relevance_verdict",
]


def _process_row(
    idx: int,
    row: pd.Series,
    collection,
    detector: IndicAIDetector,
    detector_lock: threading.Lock,
    provider: str,
    model: str,
    api_key: str,
    temperature: float,
) -> dict:
    theme = _clean_text(row.get("Theme"))
    title = _clean_text(row.get("Title"))
    problem = _clean_text(row.get("Problem"))
    solution = _clean_text(row.get("Solution"))
    doc_urls = _parse_documents(row.get("Documents"))

    dup = _duplicate_check(title, problem, solution, collection)
    p_ai = _gpt_detect(problem, detector, detector_lock, "problem")
    s_ai = _gpt_detect(solution, detector, detector_lock, "solution")
    eval_result = _evaluate(
        theme, problem, solution, doc_urls,
        provider, model, api_key, temperature, title=title,
    )
    final = _final_verdict(dup, eval_result)

    out: dict[str, Any] = {}
    out.update(dup)
    out.update(p_ai)
    out.update(s_ai)
    out.update(eval_result)
    out.update(final)
    return {"idx": idx, **out}


# ── Main ───────────────────────────────────────────────────────────────────

def run(
    input_file: str,
    output_file: str,
    provider: str,
    model: str,
    temperature: float,
    limit: int | None,
    concurrency: int,
) -> None:
    input_path = BASE_DIR / input_file
    if not input_path.exists():
        sys.exit(f"[pipeline] Input not found: {input_path}")

    api_key_env = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    api_key = os.getenv(api_key_env)
    if not api_key:
        sys.exit(f"[pipeline] {api_key_env} not set")

    print(f"[pipeline] Loading duplicate corpus via rag.get_or_build_store ...")
    collection = _load_duplicate_corpus(output_file=output_file)
    print(f"[pipeline] Corpus ready: {collection.count()} docs")

    print(f"[pipeline] Loading IndicAIDetector ...")
    detector = IndicAIDetector()
    detector_lock = threading.Lock()

    df = pd.read_excel(input_path)
    df = df.dropna(how="all")
    # If a previously generated pipeline output is used as input, avoid
    # re-appending old/new pipeline columns and keep the output schema clean.
    drop_cols = [c for c in (OUTPUT_COLUMNS + LEGACY_OUTPUT_COLUMNS) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if limit is not None:
        df = df.head(limit)
    print(f"[pipeline] Processing {len(df)} rows (provider={provider}, model={model}, workers={concurrency})")

    results_by_idx: dict[int, dict] = {}
    workers = min(concurrency, max(1, len(df)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_row, idx, row, collection, detector, detector_lock,
                provider, model, api_key, temperature,
            ): idx
            for idx, row in df.iterrows()
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
                results_by_idx[idx] = res
            except Exception as e:
                logger.exception(f"[pipeline] row {idx} crashed: {e}")
                results_by_idx[idx] = {"idx": idx, "pipeline_verdict": "ERROR", "pipeline_reason": str(e)[:200]}
            done += 1
            cid = _safe_str(df.loc[idx].get("CID", "?"))
            if done % 10 == 0 or done == len(df):
                print(f"  [{done}/{len(df)}] last CID={cid}", flush=True)

    input_columns = list(df.columns)
    rows_out: list[dict] = []
    for idx in df.index:
        row = df.loc[idx]
        res = results_by_idx.get(idx, {})
        merged = {col: _safe_str(row[col]) for col in input_columns}
        for col in OUTPUT_COLUMNS:
            merged[col] = res.get(col, "")
        rows_out.append(merged)

    # Outputs go under inquilab/<provider>/<model>/ so different runs don't overwrite each other.
    output_dir = BASE_DIR / _safe_dir(provider) / _safe_dir(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_file
    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows_out, f, indent=2, ensure_ascii=False)
    print(f"[pipeline] JSON backup: {json_path}")

    pd.DataFrame(rows_out, columns=input_columns + OUTPUT_COLUMNS).to_excel(output_path, index=False)
    print(f"[pipeline] Excel written: {output_path}")

    accept = sum(1 for r in rows_out if r.get("pipeline_verdict") == "ACCEPT")
    reject = sum(1 for r in rows_out if r.get("pipeline_verdict") == "REJECT")
    review = sum(1 for r in rows_out if r.get("pipeline_verdict") == "REVIEW")
    error = sum(1 for r in rows_out if r.get("pipeline_verdict") == "ERROR")
    print(f"[pipeline] Verdict summary: ACCEPT={accept} REJECT={reject} REVIEW={review} ERROR={error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="200 Golden_dataset_2.O-3.xlsx")
    parser.add_argument("--output", default="200_Golden_dataset_pipeline_output.xlsx")
    parser.add_argument("--provider", default="google", choices=["openai", "google"])
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows (smoke test)")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    run(
        input_file=args.input,
        output_file=args.output,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        limit=args.limit,
        concurrency=args.concurrency,
    )
