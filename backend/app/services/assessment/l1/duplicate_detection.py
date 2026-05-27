"""Duplicate detection filter for L1 pipeline."""

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_VAGUE_SYS = """
You are a strict VAGUENESS gate for the School Innovation Marathon (SIM)
duplicate-detection pipeline. Submissions come from Indian school students grades 6-12.
You run BEFORE corpus duplicate detection. Decide only if the submission has enough
surface area for corpus matching. NOT a quality gate.

NOT VAGUE (let through to corpus check):
- Widely-known/textbook ideas (rainwater harvesting, anti-theft alarm)
- Weak novelty / unclear feasibility
- Hindi/Telugu/mixed Indian-language text
- Bad grammar or rambling if content present
- Long essays naming domain + audience + any mechanism

VAGUE only when ALL: problem names no issue/target/domain, solution names no mechanism,
text is empty / aspirational ("make society better") / gibberish.

DECISION: 0-1 clear dimensions present -> vague=true. 2+ -> vague=false. Borderline -> false.

Output ONLY JSON: {"vague": true|false, "reason": "max 15 words"}
"""

_DUP_SYS = """
You are a strict duplicate-detection judge for an innovation competition corpus.

Given a submitted idea, search the corpus and compare precisely.
Focus on MECHANISM of the solution, not category or theme.

Verdict (exactly one): DUPLICATE / OVERLAP / PARTIAL_MATCH / UNIQUE

  DUPLICATE: Both problem AND solution mechanism substantially match a corpus entry.
  OVERLAP: Either problem OR solution mechanism matches, other side clearly different.
  PARTIAL_MATCH: Thematic/conceptual similarity only — same domain, different mechanism.
  UNIQUE: Neither problem nor solution substantially matches anything in corpus.

Response format (follow exactly):
Verdict: <DUPLICATE | OVERLAP | PARTIAL_MATCH | UNIQUE>
Title: <closest match title — OMIT if UNIQUE>
Source: <SOURCE_URL from chunk verbatim — OMIT if UNIQUE>
URL: <same as Source — OMIT if UNIQUE>
Matching sentence: <exact sentence from chunk — OMIT if UNIQUE>
Reason: <one sentence comparing mechanisms>

RULES:
- UNIQUE -> output ONLY Verdict + Reason.
- NOT UNIQUE -> Title, Source, URL, Matching sentence ALL required.
- Source/URL MUST be VERBATIM from "SOURCE_URL:" line in retrieved chunk.
- NEVER write filenames, page numbers, or constructed URLs.
"""


def _build_combined(content_parts: dict[str, str]) -> str:
    parts = [f"{col}:\n{val}" for col, val in content_parts.items() if val.strip()]
    return "\n\n".join(parts)


def _check_vague(
    text: str,
    gemini_client: genai.Client,
    model: str,
) -> tuple[bool, str]:
    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=f"Submission:\n\n{text}",
            config=types.GenerateContentConfig(
                system_instruction=_VAGUE_SYS,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        parsed = json.loads((response.text or "").strip())
        return bool(parsed.get("vague", False)), str(parsed.get("reason", ""))
    except Exception as exc:
        logger.warning("[_check_vague] Parse error — defaulting not vague | %s", exc)
        return False, "(vague check error — defaulting to not vague)"


def _call_file_search(
    text: str,
    gemini_client: genai.Client,
    model: str,
    store_name: str,
) -> str:
    response = gemini_client.models.generate_content(
        model=model,
        contents=f"Submitted idea to check for duplicates:\n\n{text}",
        config=types.GenerateContentConfig(
            system_instruction=_DUP_SYS,
            tools=[
                types.Tool(
                    file_search=types.FileSearch(file_search_store_names=[store_name])
                )
            ],
            temperature=0.0,
        ),
    )
    return response.text or ""


_VERDICT_VALUES = {"DUPLICATE", "OVERLAP", "PARTIAL_MATCH", "UNIQUE"}


def _parse_verdict(raw: str) -> dict[str, str | None]:
    fields: dict[str, str | None] = {
        "verdict": "",
        "match_title": None,
        "source_url": None,
        "matching_sentence": None,
        "reason": None,
    }
    keymap = {
        "verdict": "verdict",
        "title": "match_title",
        "source": "source_url",
        "url": "source_url",
        "matching sentence": "matching_sentence",
        "reason": "reason",
    }
    for line in (raw or "").splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        norm = re.sub(r"[^a-z\s]", "", k.strip().lower()).strip()
        if norm in keymap:
            fields[keymap[norm]] = v.strip() or None

    # Fallback: scan entire response for a known verdict token
    if not fields["verdict"] or fields["verdict"] not in _VERDICT_VALUES:
        m = re.search(r"\b(DUPLICATE|OVERLAP|PARTIAL_MATCH|UNIQUE)\b", raw or "")
        if m:
            fields["verdict"] = m.group(1)
            logger.warning(
                "[_parse_verdict] key-based parse missed verdict; regex fallback found: %s",
                fields["verdict"],
            )
        else:
            logger.warning(
                "[_parse_verdict] verdict not found in response. raw=%r",
                (raw or "")[:500],
            )

    return fields


def run_duplicate_detection(
    row_idx: int,
    row: dict[str, str],
    columns: list[str],
    gemini_client: genai.Client,
    model: str,
    store_name: str,
) -> dict[str, Any]:
    """Run duplicate detection on a single row.

    Returns a dict with: row_id, verdict, match_title, source_url,
    matching_sentence, reason.
    Always passthrough — never gates L2.
    """
    content_parts = {col: row.get(col, "") for col in columns}
    combined = _build_combined(content_parts) or "(empty submission)"

    try:
        is_vague, vague_reason = _check_vague(combined, gemini_client, model)
    except Exception as exc:
        logger.warning(
            "[run_duplicate_detection] Vague check failed row_%s | %s", row_idx, exc
        )
        is_vague, vague_reason = False, f"(vague check error: {exc})"

    if is_vague:
        return {
            "row_id": f"row_{row_idx}",
            "verdict": "VAGUE",
            "match_title": None,
            "source_url": None,
            "matching_sentence": None,
            "reason": vague_reason,
        }

    try:
        raw = _call_file_search(combined, gemini_client, model, store_name)
        parsed = _parse_verdict(raw)
        return {
            "row_id": f"row_{row_idx}",
            "verdict": parsed["verdict"] or "UNKNOWN",
            "match_title": parsed["match_title"],
            "source_url": parsed["source_url"],
            "matching_sentence": parsed["matching_sentence"],
            "reason": parsed["reason"],
        }
    except Exception as exc:
        logger.warning(
            "[run_duplicate_detection] File search failed row_%s | %s", row_idx, exc
        )
        return {
            "row_id": f"row_{row_idx}",
            "verdict": "ERROR",
            "match_title": None,
            "source_url": None,
            "matching_sentence": None,
            "reason": str(exc)[:200],
        }
