"""Turn the recorded Gemini TTS token counts into a cost table.

Reads the CSV written by gemini_token_usage.py. Rates come from the Gemini
Developer API pricing page, which is the API these calls go to — Google Cloud
Text-to-Speech is a separate product billed per character, and its rates do not
apply here.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "audio_output"

TOKENS_PER_UNIT = 1_000_000
PROMO_NOTE = "through 31 Dec 2026"

# Source: https://ai.google.dev/gemini-api/docs/pricing (paid tier, standard).
# Input is listed as a text rate only; the audio input tokens every call reports
# have no published rate, so they are priced at the text rate as the cautious
# reading and reported separately below.
PRICING: dict[str, dict[str, float]] = {
    "gemini-3.1-flash-tts-preview": {"input": 1.00, "output": 20.00},
    "gemini-3.8-flash-tts": {"input": 0.50, "output": 9.00},
    "gemini-3.8-flash-lite-tts": {"input": 0.50, "output": 6.00},
}

# The promotional rates on the 3.8 models double on 1 Jan 2027; 3.1 Preview has
# no promotional rate, so its price is unchanged.
PRICING_FROM_2027: dict[str, dict[str, float]] = {
    "gemini-3.1-flash-tts-preview": {"input": 1.00, "output": 20.00},
    "gemini-3.8-flash-tts": {"input": 1.00, "output": 18.00},
    "gemini-3.8-flash-lite-tts": {"input": 1.00, "output": 12.00},
}

PROJECTION_REQUESTS = 1_000

INR_PER_USD = 96.5
FX_NOTE = "USD converted at ₹96.5"

DATASET_TITLES = {
    "setu_law_golden_qna": "Setu Law",
    "atree_golden_qna": "ATREE (Nepali)",
}

DISPLAY_NAMES = {
    "gemini-3.1-flash-tts-preview": "Gemini 3.1 Flash TTS Preview",
    "gemini-3.8-flash-tts": "Gemini 3.8 Flash TTS",
    "gemini-3.8-flash-lite-tts": "Gemini 3.8 Flash Lite TTS",
}

# Sarvam and ElevenLabs bill on their own dashboards rather than exposing token
# counts, so these come from the provider consoles for a median-length answer —
# a different method from the measured Gemini figures, not a weaker one. Keyed by
# dataset, because an answer's cost depends on how long that dataset's answers are.
DASHBOARD_COST_INR: dict[str, dict[str, float]] = {
    "setu_law_golden_qna": {
        "Sarvam Bulbul v3": 2.07,
        "ElevenLabs v4": 1.66,
    },
}

# ElevenLabs bills per character, so a per-character rate carries exactly from one
# dataset to another. The multiplier is the MEAN answer length, not the median:
# the column projects the cost of 1,000 answers, and only the mean multiplies out
# to a correct total. The published Setu figure is quoted for a median answer, so
# the rate is recovered from it before being applied.
ELEVENLABS_INR_PER_CHAR = 1.66 / 691
MEAN_ANSWER_CHARS = {"atree_golden_qna": 250.5}

DERIVED_COST_INR: dict[str, dict[str, float]] = {
    dataset: {"ElevenLabs v4": ELEVENLABS_INR_PER_CHAR * chars}
    for dataset, chars in MEAN_ANSWER_CHARS.items()
}


def cost_of(input_tokens: int, output_tokens: int, rates: dict[str, float]) -> float:
    return (
        input_tokens * rates["input"] + output_tokens * rates["output"]
    ) / TOKENS_PER_UNIT


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "success"]
    for row in rows:
        for field, value in row.items():
            if field not in {"model", "status", "error"}:
                row[field] = float(value) if "." in value else int(value)
    return rows


def summarise(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in sorted({r["model"] for r in records}):
        rows = [r for r in records if r["model"] == model]
        if model not in PRICING:
            logger.warning(f"[summarise] No published rate for {model}, skipping")
            continue
        input_tokens = sum(r["total_input_tokens"] for r in rows)
        audio_input_tokens = sum(r["input_audio_tokens"] for r in rows)
        output_tokens = sum(r["total_output_tokens"] for r in rows)
        chars = sum(r["char_count"] for r in rows)
        now = cost_of(input_tokens, output_tokens, PRICING[model])
        later = cost_of(input_tokens, output_tokens, PRICING_FROM_2027[model])
        out.append(
            {
                "model": model,
                "requests": len(rows),
                "chars": chars,
                "input_tokens": input_tokens,
                "audio_input_tokens": audio_input_tokens,
                "output_tokens": output_tokens,
                "cost_now": now,
                "cost_2027": later,
                "cost_per_request": now / len(rows),
                "cost_per_1k": now / len(rows) * PROJECTION_REQUESTS,
                "cost_2027_per_1k": later / len(rows) * PROJECTION_REQUESTS,
            }
        )
    return sorted(out, key=lambda s: s["cost_per_request"])


def render(summary: list[dict[str, Any]], dataset: str, answers: int) -> str:
    dashboard = dict(DASHBOARD_COST_INR.get(dataset, {}))
    derived = DERIVED_COST_INR.get(dataset, {})
    dashboard.update(derived)
    lines = [
        f"# TTS cost — {answers} {DATASET_TITLES.get(dataset, dataset)} answers",
        "",
        "Token counts measured from the API response of every request "
        f"(`audio_output/{dataset}_token_usage.csv`). Rates are Gemini Developer API "
        f"paid-tier, standard, {PROMO_NOTE} where promotional.",
        "",
        "## Measured run",
        "",
        "| Model | Requests | Input tokens | Output audio tokens | Cost | Per request |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary:
        lines.append(
            f"| {s['model']} | {s['requests']} | {s['input_tokens']:,} | "
            f"{s['output_tokens']:,} | ${s['cost_now']:.4f} | ${s['cost_per_request']:.5f} |"
        )

    lines += [
        "",
        "## Projected",
        "",
        f"| Model | Per {PROJECTION_REQUESTS:,} answers | Same, from 1 Jan 2027 |",
        "| --- | ---: | ---: |",
    ]
    for s in summary:
        lines.append(
            f"| {s['model']} | ${s['cost_per_1k']:.2f} | ${s['cost_2027_per_1k']:.2f} |"
        )

    lines += [
        "",
        "## All providers",
        "",
        f"Every service reading the same {answers} answers. {FX_NOTE}.",
        "",
        "| Service | ₹ / answer | $ / answer | ₹ / 1,000 answers | $ / 1,000 answers "
        "| ₹ / 1,000 answers from 1 Jan 2027 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    combined: list[tuple[str, float, float | None]] = [
        (
            DISPLAY_NAMES.get(s["model"], s["model"]),
            s["cost_per_request"] * INR_PER_USD,
            s["cost_2027"] / s["requests"] * INR_PER_USD,
        )
        for s in summary
    ]
    # Neither provider has announced a 2027 rate, so the column is left empty for
    # them rather than carrying today's price forward as though it were known.
    combined += [(name, inr, None) for name, inr in dashboard.items()]
    for name, inr, inr_2027 in sorted(combined, key=lambda c: c[1]):
        later = (
            f"{inr_2027 * PROJECTION_REQUESTS:,.0f}"
            if inr_2027 is not None
            else "not announced"
        )
        lines.append(
            f"| {name} | {inr:.2f} | {inr / INR_PER_USD:.5f} | "
            f"{inr * PROJECTION_REQUESTS:,.0f} | "
            f"{inr / INR_PER_USD * PROJECTION_REQUESTS:,.2f} | {later} |"
        )

    lines += [
        "",
        "## Notes",
        "",
    ]
    measured = [n for n in dashboard if n not in derived]
    if measured:
        lines.append(
            "- "
            + " and ".join(measured)
            + " figures come from the provider dashboards "
            "for a median answer; the Gemini figures are measured token counts averaged "
            "over every answer. The two are different methods, and answer length differs "
            "slightly between them."
        )
    if derived:
        lines.append(
            "- "
            + " and ".join(derived)
            + " bills per character, so its figure here is "
            f"its rate of ₹{ELEVENLABS_INR_PER_CHAR * 1e6:,.0f} per million characters "
            "applied to this dataset's mean answer length."
        )
    lines += [
        "- Output audio tokens dominate the bill; input is a rounding error at these rates.",
        "- Every request carries fixed audio input tokens that do not scale with the text "
        "(see `input_audio_tokens`). They are priced here at the published text input rate, "
        "because no separate audio input rate is published for TTS models.",
        "- A free tier exists for all three models; these figures are the paid tier.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="setu_law_golden_qna")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input or OUTPUT_DIR / f"{args.dataset}_token_usage.csv"
    report = args.output or OUTPUT_DIR / f"{args.dataset}_tts_cost.md"
    records = load_records(source)
    summary = summarise(records)
    answers = max((s["requests"] for s in summary), default=0)
    report.write_text(render(summary, args.dataset, answers), encoding="utf-8")
    for s in summary:
        logger.info(
            f"[main] {s['model']} | {s['requests']} requests | ${s['cost_now']:.4f} | "
            f"${s['cost_per_request']:.5f}/request | ${s['cost_per_1k']:.2f}/1k"
        )
    logger.info(f"[main] Wrote {report}")


if __name__ == "__main__":
    main()
