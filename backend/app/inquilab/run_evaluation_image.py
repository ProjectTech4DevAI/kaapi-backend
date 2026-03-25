"""
Image Summary Script — Extracts image_summary and idea_summary from student submissions.

HOW TO USE:
1. Scroll down to the bottom of this file (if __name__ == "__main__")
2. Set PROVIDER, MODEL, INPUT_FILE, OUTPUT_FILENAME, TEMPERATURE
3. Run:  uv run -m app.inquilab.script_image
4. Output goes to: app/inquilab/output_image/<filename>.xlsx and .json
"""

from typing import List
from app.services.llm.providers import LLMProvider
from dotenv import load_dotenv
import os
from app.services.llm.jobs import resolved_input_context
from app.models.llm.request import (
    LLMCallRequest,
    QueryParams,
    LLMCallConfig,
    TextLLMParams,
    KaapiCompletionConfig,
    ConfigBlob,
)

import random
import json
import string
import pandas as pd
from pathlib import Path
from app.services.llm.mappers import transform_kaapi_config_to_native
from logging import Logger
import re

load_dotenv()

logger = Logger(__name__)

BASE_DIR = Path(__file__).parent


def get_system_role() -> str:
    """
    WHO is the AI pretending to be?
    Edit this to change the evaluator's persona and background.
    """
    return """You are a senior innovation evaluator for the School Innovation Marathon (SIM) - India's largest student innovation platform.
You have 15+ years of experience evaluating educational innovations and understand the Indian student context deeply.

Your evaluation philosophy:
- CONTEXTUAL: Recognize that these are school students (grades 6-12), not professional innovators
- BALANCED: Acknowledge effort while maintaining rigorous standards for true innovation
- FAIR: Consider resource constraints and educational context of Indian students
- EVIDENCE-BASED: Score based on specific evidence in submissions, not assumptions
- GROWTH-ORIENTED: Evaluate current state while recognizing potential for development
- You evaluate with the perspective of someone who has seen thousands of student innovations and can distinguish between genuine innovation, good effort, and superficial attempts.
- SUBMISSIONS MAY INCLUDE MULTIPLE DATA TYPES:
    1. TEXT
    - Idea Title
    - Problem statement
    - Solution description
    2. IMAGES
    - Ideas details written on paper
    - Prototype photos
    - Diagrams of solution or story board sketch
    - Experimental setups
    - Hardware builds
    - Design sketches
    3. PDF DOCUMENTS
    - Technical reports
    - Ideas Written on paper
    - Research notes
    - Design documentation
    - Calculations
    - Implementation steps
- EVALUATION RULES FOR MULTIMODAL INPUT:
    1. When Text & attachments (image/pdf) are present:
    - Consider Problem statement , Solution text as main source and attachments will be supporting evidence.
    - Do not assume claims that are not visible or stated.
    - If text and attachments evidence contradict, trust the text description if it's conveying solution details properly, if not Trust attachments & understand the idea from it.
    2. When only attachments (image/pdf) are present: Use attachments as the main source for extracting problem statements , solutions and gathering evidence to validate the idea.
    3. When only text Input exists: Evaluate based only on text.
- Attachment evaluation rules:
    1. Rule 1: Prototype Confirmation Rule : If a prototype only confirms the idea without adding new design or working clarity, it should increase confidence but not change scores.
    2. Rule 2: Prototype Impact on Scoring : Prototype evidence should only change scores if it introduces new functional insight or reduces uncertainty about how the solution works.
    3. Rule 3: Prototype vs Effort : A prototype demonstrates student effort and experimentation, which should be acknowledged in evaluation, but effort alone should not lead to higher scores without additional functional evidence.
    4. Rule 4: Irrelevant Evidence Handling : If additional images or documents are not clearly related to the problem or solution, they must be ignored and should not influence scoring.
    5. Rule 5: Generic Solution without Explanation : If a student suggests a widely known solution without explaining how it works or how they will implement it, assign: Low novelty (1-2), Mid-range feasibility and scalability (6-7).
    6. Rule 6: Prototype Adds New Design Evidence -> Scores Increase.
    7. Rule 7: Prototype Only Confirms Existing Understanding -> Scores Do Not Increase.
    8. Rule 8: Design Clarity vs Technical Depth (Score Ceiling) : Even when a prototype adds clear design understanding, if it does not explain deeper aspects such as why the design works better, performance, durability, or optimization, scores should not reach the highest range (9-10).

CRITICAL SCORING GUARDRAILS:
- Scores MUST be justified by explicit evidence in the submission.
- Do NOT assume resources, infrastructure, or execution capability unless stated.
- If the idea and problem donot relate to each other then score least to all parameters.
- Generic textbook or widely known ideas must be scored low on novelty unless clear original adaptation is shown.
- If the model cannot clearly interpret the attachment, do not fabricate details. Score based on available text evidence only.

HUMAN EVALUATION PRINCIPLES:
- Recognize visible student effort and original thinking even if execution is incomplete
- Reward contextual relevance and real-world problem awareness
- Separate sustainability intention from operational sustainability
- Avoid over-penalizing grammar or presentation
- Do not inflate scores without evidence

When evidence is partial:
-> score conservatively but acknowledge effort
-> use mid-range scores when student reasoning is present but incomplete"""


def get_evaluation_objective() -> str:
    """
    WHAT should the AI evaluate from images?
    Edit this to change the evaluation goals.
    """
    return """Evaluate student innovation submissions to identify genuinely promising ideas that demonstrate:
1. Real problem understanding and original thinking
2. Practical solutions that could realistically be implemented
3. Potential for meaningful impact in the Indian context
4. Consideration of sustainability and scalability factors
5. Evidence of student's own creative thinking vs. copied ideas

Distinguish clearly between:
- intention vs execution detail
- concept vs deployable solution
- originality vs adaptation

Your evaluation directly impacts which students advance in India's premier innovation program, so accuracy and fairness are paramount."""


def get_evaluation_method() -> str:
    """
    HOW should the AI evaluate?
    Edit this to change the step-by-step evaluation process.
    """
    return """EVALUATION PROCESS:

STEP 1: COMPREHENSION
Either from Text or attachments, try to
- Read the problem carefully - understand what issue the student is trying to solve
- Read the solution - understand their proposed approach
- Identify the student's level of detail, originality, and thinking depth

STEP 2: CONTEXTUAL ASSESSMENT
- Consider this is a school student's work (grades 6-12)
- Assess against the backdrop of Indian educational and social context
- Look for evidence of personal observation vs. generic problem identification
- Evaluate solution practicality within Indian resource constraints
- Evidence types and what to understand from it:
    a. Evidence from Problem Statement (Text) - Understand problem students are trying to solve
    b. Evidence from Solution Statement (Text) - Understand idea details from solution
    c. Evidence from Attachments - Interpret the attachments to identify problem , Idea , Solution design/model etc for supporting evidences if any.

STEP 3: EVIDENCE-BASED ANALYSIS
- Analyze what the image(s) show — prototype, diagram, sketch, experimental setup, etc.
- Determine whether it appears student-made
- Assess how well it supports the stated problem and solution

STEP 4: OUTPUT GENERATION
- Return ONLY valid JSON with exact structure specified
- image_summary: what the image shows and how it supports the submission
- idea_summary: the student's idea extracted from image context"""


def get_user_prompt(problem: str, solution: str) -> str:
    """
    USER PROMPT — This is sent for each submission.
    Edit this to change what the AI sees for each student's work.
    """
    return (
        f"# Problem Statement\n{problem}\n\n"
        f"# Proposed Solution\n{solution}\n\n"
        "Based on the attached image(s) and the above problem and solution, provide:\n"
        "1. **image_summary**: A brief summary of what the image shows (prototype, diagram, sketch, etc.), "
        "whether it appears student-made, and how well it supports the stated problem and solution.\n"
        "2. **idea_summary**: A summary of the student's idea extracted from the image, "
        "including the problem they are solving and the proposed solution approach.\n\n"
        "Respond in JSON format."
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
        "output_rules": {
            "format": "Return ONLY valid JSON object",
            "required_keys": ["image_summary", "idea_summary"],
            "structure": {
                "image_summary": "string (brief summary of what the image(s) show — prototype, diagram, sketch, experimental setup, etc., whether it appears student-made, and how well it supports the stated problem and solution)",
                "idea_summary": "string (summary of the student's idea extracted from the image(s), including the problem they are solving and the proposed solution approach)",
            },
        },
    }

    return (
        "# Image Summary System Schema\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "# You must output a single JSON object with image_summary and idea_summary fields."
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


def _write_excel(results: List[dict], output_path: str) -> None:
    """
    Writes Excel with columns: CID | Problem | Solution | Image URL | Image Summary | Idea Summary
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active

    # Header row
    ws.append(["CID", "Problem", "Solution", "Image URL", "Image Summary", "Idea Summary"])

    for entry in results:
        image_url = entry.get("Image URL", "")
        if isinstance(image_url, list):
            image_url = ", ".join(image_url)

        ws.append([
            str(entry.get("CID", "")),
            entry.get("Problem", ""),
            entry.get("Solution", ""),
            image_url,
            entry.get("Image Summary", ""),
            entry.get("Idea Summary", ""),
        ])

    wb.save(output_path)
    print(f"Excel file saved: {output_path}")


def run(
    provider: str,
    model: str,
    input_file: str,
    output_path: str,
    temperature: float,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Provider: {provider}")
    print(f"  Model:    {model}")
    print(f"  Input:    {input_file}")
    print(f"  Output:   {output_path}")
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
            "image_summary": {
                "type": "string",
                "description": "Brief plain-text summary with bullet points describing what the image contains and what the student is conveying about their innovation.",
            },
            "idea_summary": {
                "type": "string",
                "description": "Summary of the student's idea extracted from the image, including the problem they are solving and the proposed solution approach.",
            },
        },
        "required": ["image_summary", "idea_summary"],
        "additionalProperties": False,
    }

    completion_config = KaapiCompletionConfig(
        provider=provider,
        type="text",
        params=TextLLMParams(
            model=model,
            instructions=instructions,
            temperature=temperature,
            output_schema=output_schema,
        ).model_dump(exclude_none=True),
    )

    completion_config, warnings = transform_kaapi_config_to_native(completion_config)
    configuration = LLMCallConfig(blob=ConfigBlob(completion=completion_config))
    provider_class = LLMProvider.get_provider_class(provider_type=provider)

    if provider == "openai":
        credential = {"api_key": os.getenv("OPENAI_API_KEY")}
    else:
        credential = {"api_key": os.getenv("GEMINI_API_KEY")}

    client = provider_class.create_client(credentials=credential)
    provider_instance = provider_class(client=client)

    results = []
    total = len(df)

    for idx, row in df.iterrows():
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
        inputs = [{"type": "text", "content": {"format": "text", "value": user_text}}]

        for img_url in image_urls:
            inputs.append({"type": "image", "content": {"format": "url", "value": img_url}})

        query = QueryParams(input=inputs)
        request = LLMCallRequest(query=query, config=configuration)

        try:
            with resolved_input_context(query_input=request.query.input) as resolved_input:
                response, error = provider_instance.execute(
                    completion_config=request.config.blob.completion,
                    query=query,
                    resolved_input=resolved_input,
                    include_provider_raw_response=False,
                )

            if error is not None or response is None:
                print(f"ERROR: {error}")
                response_output = None
            else:
                response_output = json.loads(response.response.output.content.value)
                print("OK")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            response_output = None

        results.append({
            "CID": cid,
            "Problem": problem,
            "Solution": solution,
            "Image URL": image_urls,
            "Image Summary": response_output.get("image_summary", "") if isinstance(response_output, dict) else "",
            "Idea Summary": response_output.get("idea_summary", "") if isinstance(response_output, dict) else "",
        })

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save JSON backup
    json_path = output_path.replace(".xlsx", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON backup saved: {json_path}")

    # Save Excel
    _write_excel(results, output_path)

    print(f"\nDone! Processed {len(results)} submissions.")


if __name__ == "__main__":

    PROVIDER = "openai"
    MODEL = "gpt-4o-mini"
    INPUT_FILE = "200 Golden_dataset_2.O-3.xlsx"
    OUTPUT_PATH = os.path.join(BASE_DIR, "output_image", "image_summary_results.xlsx")
    TEMPERATURE = 0.4

    run(
        provider=PROVIDER,
        model=MODEL,
        input_file=INPUT_FILE,
        output_path=OUTPUT_PATH,
        temperature=TEMPERATURE,
    )
