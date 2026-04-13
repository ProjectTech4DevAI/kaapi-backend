from dotenv import load_dotenv
import os
import random
import json
import string
import pandas as pd
from pathlib import Path
import re
import llm_client
import llm_batch
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt import (
    get_system_role,
    get_evaluation_objective,
    get_evaluation_method,
    get_scoring_criteria,
    get_few_shot_examples,
    get_feedback_instructions
)
load_dotenv()

METRICS = ["Novelty", "Usefulness", "Feasibility", "Scalability", "Sustainability"]
BASE_DIR = Path(__file__).parent


def get_user_prompt(problem_identified: str, description: str, solution: str) -> str:
    return (
        f"# Problem Statement\n\n{problem_identified}\n\n"
        f"# Solution to the problem statement:\n\n{description}\n---\n{solution}\n\n"
        "First score this submission across Novelty, Usefulness, Feasibility, Scalability, and Sustainability.\n"
        "Then use those scores to calibrate and generate mentor feedback.\n"
        "IMPORTANT: The Idea_Feedback MUST be written in the same language as the student's problem and solution text above.\n"
        "If the student wrote in Hindi, write feedback in Hindi. If in Telugu, write in Telugu. If in English, write in English.\n"
        "If mixed languages, use the dominant language of the solution text.\n"
        "If attachments are present, summarize what they show in Attachment_Summary.\n"
        "Respond in the required JSON format.\n\n"
        "# You must output a single JSON object with scores for each metric (score + reason), "
        "an Attachment_Summary field (summarize what images/attachments show, or empty string if none), "
        "and an Idea_Feedback field containing the full feedback text following the format above."
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
            "required_keys": [f"{m}_score" for m in METRICS] + [f"{m}_reason" for m in METRICS],
            "structure": {
                f"{m}_score": "1-10" for m in METRICS
            } | {
                f"{m}_reason": "string (one line why this score)" for m in METRICS
            },
        },
    }

    examples = {"name": "Example Dataset", "list": get_few_shot_examples()}

    return (
        "# Evaluation System Schema\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "# Few-Shot Examples\n"
        f"{json.dumps(examples, indent=2, ensure_ascii=False)}\n\n"
        f"{get_feedback_instructions()}\n\n"
        "# You must output a single JSON object with scores for each metric (score + reason), "
        "an Attachment_Summary field (summarize what images/attachments show, or empty string if none), "
        "and an Idea_Feedback field containing the full feedback text following the format above."
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


def _collect_image_urls(row) -> list[str] | None:
    """Collect image URLs from photo1 and photograph columns."""
    urls = []
    for col in ("photo1", "photograph"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            for u in str(val).split(","):
                u = u.strip()
                if u:
                    urls.append(_to_direct_url(u))
    return urls if urls else None


def _write_excel(results: List[dict], output_path: str, input_columns: List[str]) -> None:
    from openpyxl import Workbook

    criteria = ["Novelty", "Usefulness", "Feasibility", "Scalability", "Sustainability"]

    output_headers: List[str] = []
    for c in criteria:
        output_headers.extend([f"{c}_score", f"{c}_reason"])
    output_headers.extend(["Attachment_Summary", "Idea_Feedback", "Skipped_Images"])

    wb = Workbook()
    ws = wb.active

    # Column headers: input columns + output columns
    header = list(input_columns) + output_headers
    ws.append(header)

    for entry in results:
        row = [entry.get(col, "") for col in input_columns]
        for key in output_headers:
            row.append(entry.get(key, ""))
        ws.append(row)

    wb.save(output_path)
    print(f"Excel file saved: {output_path}")


def run(
    provider,
    model,
    input_file,
    output_filename,
    temperature,
    limit=None,
    batch=False,
    poll_interval=30,
    concurrency=10,
):
    print(f"\n{'='*60}")
    print(f"  Provider: {provider}")
    print(f"  Model:    {model}")
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

    if "problem_identified" not in df.columns:
        print("ERROR: Input file must have 'problem_identified' column")
        return
    if "description" not in df.columns or "solution" not in df.columns:
        print("ERROR: Input file must have 'description' and 'solution' columns")
        return

    instructions = _build_system_instructions()

    output_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    for m in METRICS:
        output_schema["properties"][f"{m}_score"] = {"type": "number"}
        output_schema["properties"][f"{m}_reason"] = {"type": "string"}
        output_schema["required"].extend([f"{m}_score", f"{m}_reason"])
    output_schema["properties"]["Idea_Feedback"] = {"type": "string"}
    output_schema["properties"]["Attachment_Summary"] = {"type": "string"}
    output_schema["required"].extend(["Idea_Feedback", "Attachment_Summary"])

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print(f"ERROR: API key not found. Set {'OPENAI_API_KEY' if provider == 'openai' else 'GEMINI_API_KEY'} in .env file")
        return

    # Prepare row data
    rows_data = []
    for idx, row in df.iterrows():
        if limit is not None and len(rows_data) >= limit:
            break
        innovation_id = str(row["innovation_id"]) if pd.notna(row.get("innovation_id")) else f"innov_{_generate_id(6)}"
        problem_identified = str(row["problem_identified"]) if pd.notna(row.get("problem_identified")) else ""
        description = str(row["description"]) if pd.notna(row.get("description")) else ""
        solution = str(row["solution"]) if pd.notna(row.get("solution")) else ""
        image_urls = _collect_image_urls(row)
        user_text = get_user_prompt(problem_identified, description, solution)

        rows_data.append({
            "idx": idx,
            "row": row,
            "innovation_id": innovation_id,
            "user_text": user_text,
            "image_urls": image_urls,
        })

    total = len(rows_data)
    results = []

    if batch:
        # ── Batch mode: submit all at once, poll, collect results ──
        print(f"\n  Batch mode: preparing {total} requests...")

        row_dicts = [
            {
                "custom_id": rd["innovation_id"],
                "model": model,
                "system_prompt": instructions,
                "user_text": rd["user_text"],
                "image_urls": rd["image_urls"],
                "output_schema": output_schema,
                "temperature": temperature,
            }
            for rd in rows_data
        ]

        if provider == "openai":
            batch_requests, skipped_images = llm_batch.create_openai_batch_requests(row_dicts)
            batch_results = llm_batch.submit_openai_batch(
                api_key=api_key,
                requests=batch_requests,
                poll_interval=poll_interval,
            )
        else:
            batch_results, skipped_images = llm_batch.submit_gemini_flex(
                api_key=api_key,
                model=model,
                rows=row_dicts,
            )

        for rd in rows_data:
            row = rd["row"]
            cid = rd["innovation_id"]
            response_output = batch_results.get(cid)

            result = {col: (str(row[col]) if pd.notna(row[col]) else "") for col in df.columns}
            for m in METRICS:
                result[f"{m}_score"] = response_output.get(f"{m}_score") if response_output else None
                result[f"{m}_reason"] = response_output.get(f"{m}_reason", "") if response_output else ""
            result["Attachment_Summary"] = response_output.get("Attachment_Summary", "") if response_output else ""
            result["Idea_Feedback"] = response_output.get("Idea_Feedback", "") if response_output else ""
            result["Skipped_Images"] = ", ".join(skipped_images.get(cid, []))
            results.append(result)

    else:
        # ── Concurrent mode ──
        call_fn = llm_client.call_openai if provider == "openai" else llm_client.call_gemini
        workers = min(concurrency, total)
        print(f"\n  Concurrent mode: {workers} workers, {total} requests\n")

        def _process_one(i: int, rd: dict):
            try:
                response_output = call_fn(
                    api_key=api_key,
                    model=model,
                    system_prompt=instructions,
                    user_text=rd["user_text"],
                    image_urls=rd["image_urls"],
                    output_schema=output_schema,
                    temperature=temperature,
                )
                return i, rd, response_output, None
            except Exception as e:
                return i, rd, None, e

        done_count = 0
        result_map = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_one, i, rd): i
                for i, rd in enumerate(rows_data)
            }
            for future in as_completed(futures):
                i, rd, response_output, err = future.result()
                done_count += 1
                if err:
                    print(f"  [{done_count}/{total}] {rd['innovation_id']} ERROR: {err}")
                else:
                    print(f"  [{done_count}/{total}] {rd['innovation_id']} OK")
                result_map[i] = (rd, response_output)

        for i in range(total):
            rd, response_output = result_map[i]
            row = rd["row"]
            result = {col: (str(row[col]) if pd.notna(row[col]) else "") for col in df.columns}
            for m in METRICS:
                result[f"{m}_score"] = response_output.get(f"{m}_score") if response_output else None
                result[f"{m}_reason"] = response_output.get(f"{m}_reason", "") if response_output else ""
            result["Attachment_Summary"] = response_output.get("Attachment_Summary", "") if response_output else ""
            result["Idea_Feedback"] = response_output.get("Idea_Feedback", "") if response_output else ""
            result["Skipped_Images"] = ""
            results.append(result)

    # e.g. output_score_feedback/openai/gpt-4o-mini/results.xlsx
    model_folder = model.replace("/", "_")
    output_dir = os.path.join(BASE_DIR, "output_score_feedback", provider, model_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    json_path = output_path.replace(".xlsx", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON backup saved: {json_path}")

    _write_excel(results, output_path, list(df.columns))

    print(f"\nDone! Processed {len(results)} submissions.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai", choices=["openai", "google"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--input", default="input.xlsx")
    parser.add_argument("--output", default="evaluation_results.xlsx")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=None, help="Process only this many rows (for testing)")
    parser.add_argument("--batch", action="store_true", help="Use OpenAI Batch API (cheaper, higher throughput)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between batch status checks (default 30)")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests in non-batch mode (default 10)")

    args = parser.parse_args()

    run(
        provider=args.provider,
        model=args.model,
        input_file=args.input,
        output_filename=args.output,
        temperature=args.temperature,
        limit=args.limit,
        batch=args.batch,
        poll_interval=args.poll_interval,
        concurrency=args.concurrency,
    )
