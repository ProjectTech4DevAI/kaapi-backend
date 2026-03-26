from dotenv import load_dotenv
import os
import random
import json
import string
import pandas as pd
from pathlib import Path
import re
import llm_client
from typing import List
from prompt import(
    get_scoring_criteria,
    get_evaluation_method,
    get_evaluation_objective,
    get_few_shot_examples,
    get_system_role
)

load_dotenv()

METRICS = ["Novelty", "Usefulness", "Feasibility", "Scalability", "Sustainability"]
BASE_DIR = Path(__file__).parent


# =====================================================================
#  PROMPTS — Edit these functions to change AI behavior
# =====================================================================


def get_user_prompt(problem: str, solution: str) -> str:
    """
    USER PROMPT — This is sent for each submission.
    Edit this to change what the AI sees for each student's work.
    """
    return (
        f"# Problem Statement\n:{problem}\n\n"
        f"# Solution to the problem statement:\n{solution}\n\n"
        "Score this submission across Novelty, Usefulness, Feasibility, Scalability, and Sustainability. "
        "If attachments are present, summarize what they show in Image_contains_summary. "
        "Respond in the required JSON format."
    )


# =====================================================================
#  MAIN CODE LOGIC
# =====================================================================


def _build_system_instructions() -> str:
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
            "required_keys": METRICS,
            "structure": {
                m: {"score": "1-10", "reason": "string (one line why this score)"}
                for m in METRICS
            },
        },
    }

    examples = {"name": "Example Dataset", "list": get_few_shot_examples()}

    return (
        "# Evaluation System Schema\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "# Few-Shot Examples\n"
        f"{json.dumps(examples, indent=2, ensure_ascii=False)}\n\n"
        "# You must output a single JSON object with scores for each metric (score + reason). "
        "If images are provided, also include an Image_contains_summary field."
    )


def _to_direct_url(url: str) -> str:
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


def _generate_id(length: int = 5) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _write_excel(results: List[dict], output_path: str, use_images: bool) -> None:
    from openpyxl import Workbook

    criteria = ["Novelty", "Usefulness", "Feasibility", "Scalability", "Sustainability"]

    score_headers: List[str] = []
    for c in criteria:
        score_headers.extend([c, "Reason"])
    score_headers.append("Image Summary")

    score_cols = len(score_headers)
    idea_details_cols = 4

    new_by_cid = {str(e.get("CID", "")): e for e in results}

    if use_images:
        other_json = output_path.replace(".xlsx", "_wo_images.json")
    else:
        other_json = output_path.replace(".xlsx", "_with_images.json")

    other_by_cid: dict = {}
    if os.path.exists(other_json):
        with open(other_json, "r", encoding="utf-8") as f:
            other_data = json.load(f)
        other_by_cid = {str(e.get("CID", "")): e for e in other_data}

    if use_images:
        with_by_cid = new_by_cid
        wo_by_cid = other_by_cid
    else:
        with_by_cid = other_by_cid
        wo_by_cid = new_by_cid

    all_cids = list(with_by_cid.keys())
    for cid in wo_by_cid:
        if cid not in with_by_cid:
            all_cids.append(cid)

    has_both = bool(with_by_cid and wo_by_cid)

    wb = Workbook()
    ws = wb.active

    group_row = ["Idea Details"] + [""] * (idea_details_cols - 1)
    if has_both:
        group_row += ["Without Attachments"] + [""] * (score_cols - 1)
        group_row += ["With Attachments"] + [""] * (score_cols - 1)
    elif wo_by_cid:
        group_row += ["Without Attachments"] + [""] * (score_cols - 1)
    else:
        group_row += ["With Attachments"] + [""] * (score_cols - 1)
    ws.append(group_row)

    header = ["CID", "Problem", "Solution", "Attachments url"]
    if has_both:
        header += score_headers + score_headers
    else:
        header += score_headers
    ws.append(header)

    def _extract_scores(entry: dict) -> list:
        row = []
        for c in criteria:
            obj = entry.get(c, {})
            if isinstance(obj, dict):
                row.append(obj.get("score", ""))
                row.append(obj.get("reason", ""))
            else:
                row.extend(["", ""])
        row.append(entry.get("Image Summary", ""))
        return row

    for cid in all_cids:
        with_entry = with_by_cid.get(cid, {})
        wo_entry = wo_by_cid.get(cid, {})
        detail_entry = with_entry or wo_entry

        image_url = detail_entry.get("Image URL", "")
        if isinstance(image_url, list):
            image_url = ", ".join(image_url)

        row = [
            cid,
            detail_entry.get("Problem", ""),
            detail_entry.get("Solution", ""),
            image_url,
        ]

        if has_both:
            row += _extract_scores(wo_entry)
            row += _extract_scores(with_entry)
        elif wo_by_cid:
            row += _extract_scores(wo_entry)
        else:
            row += _extract_scores(with_entry)

        ws.append(row)

    wb.save(output_path)
    print(f"Excel file saved: {output_path}")


def run(
    provider,
    model,
    input_file,
    output_filename,
    use_images,
    temperature,
    limit=None,
):
    print(f"\n{'='*60}")
    print(f"  Provider: {provider}")
    print(f"  Model:    {model}")
    print(f"  Images:   {'YES' if use_images else 'NO'}")
    print(f"  Input:    {input_file}")
    print(f"  Output:   {output_filename}")
    print(f"{'='*60}\n")

    input_path = Path(os.path.join(BASE_DIR, input_file))
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    ext = input_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        print(f"ERROR: Unsupported file type '{ext}'. Use .csv, .xlsx, or .xls")
        return

    df = df.dropna(how="all")
    df.columns = df.columns.str.lower().str.strip()

    if "problem" not in df.columns or "solution" not in df.columns:
        print("ERROR: Input file must have 'problem' and 'solution' columns")
        return

    instructions = _build_system_instructions()

    output_schema = {
        "type": "object",
        "properties": {
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
        },
        "required": METRICS + (["Image_contains_summary"] if use_images else []),
        "additionalProperties": False,
    }
    if use_images:
        output_schema["properties"]["Image_contains_summary"] = {"type": "string"}

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print(f"ERROR: API key not found. Set {'OPENAI_API_KEY' if provider == 'openai' else 'GEMINI_API_KEY'} in .env file")
        return

    call_fn = llm_client.call_openai if provider == "openai" else llm_client.call_gemini

    results = []
    total = len(df)

    for idx, row in df.iterrows():
        if limit is not None and len(results) >= limit:
            break

        if pd.isna(row.get("problem")) or pd.isna(row.get("solution")):
            continue

        cid = str(row["cid"]) if pd.notna(row.get("cid")) else f"student_{_generate_id(6)}"
        problem = str(row["problem"])
        solution = str(row["solution"])

        image_urls = None
        if pd.notna(row.get("documents")):
            image_urls = [u.strip() for u in str(row["documents"]).split(",") if u.strip()]
            image_urls = [_to_direct_url(u) for u in image_urls]

        if not image_urls:
            continue

        print(f"  [{idx+1}/{total}] Processing CID: {cid}...", end=" ", flush=True)

        user_text = get_user_prompt(problem, solution)
        send_images = image_urls if (use_images and image_urls) else None

        try:
            response_output = call_fn(
                api_key=api_key,
                model=model,
                system_prompt=instructions,
                user_text=user_text,
                image_urls=send_images,
                output_schema=output_schema,
                temperature=temperature,
            )
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            response_output = None

        result = {
            "CID": cid,
            "Problem": problem,
            "Solution": solution,
            "Novelty": response_output.get("Novelty") if response_output else None,
            "Usefulness": response_output.get("Usefulness") if response_output else None,
            "Feasibility": response_output.get("Feasibility") if response_output else None,
            "Scalability": response_output.get("Scalability") if response_output else None,
            "Sustainability": response_output.get("Sustainability") if response_output else None,
            "Image Summary": response_output.get("Image_contains_summary", "") if response_output else "",
        }

        if use_images:
            result["Image URL"] = image_urls if image_urls else []

        results.append(result)

    output_dir = os.path.join(BASE_DIR, "output_score")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    suffix = "_with_images" if use_images else "_wo_images"
    json_path = output_path.replace(".xlsx", f"{suffix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON backup saved: {json_path}")

    _write_excel(results, output_path, use_images)

    print(f"\nDone! Processed {len(results)} submissions.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai", choices=["openai", "google"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--input", default="200 Golden_dataset_2.O-3.xlsx")
    parser.add_argument("--output", default="score_results.xlsx")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=None, help="Process only this many rows (for testing)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--both", action="store_true", help="Run without attachments first, then with attachments")
    mode.add_argument("--attachment", action="store_true", help="Run with attachments only")
    mode.add_argument("--wo-attachment", action="store_true", help="Run without attachments only")

    args = parser.parse_args()

    if not args.both and not args.attachment and not args.wo_attachment:
        args.both = True

    if args.both or args.wo_attachment:
        run(
            provider=args.provider,
            model=args.model,
            input_file=args.input,
            output_filename=args.output,
            use_images=False,
            temperature=args.temperature,
            limit=args.limit,
        )

    if args.both or args.attachment:
        run(
            provider=args.provider,
            model=args.model,
            input_file=args.input,
            output_filename=args.output,
            use_images=True,
            temperature=args.temperature,
            limit=args.limit,
        )