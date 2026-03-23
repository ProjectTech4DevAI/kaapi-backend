"""
Scoring rubric and message builder for the School Innovation Marathon (SIM).
Evaluates student innovations across 5 dimensions with Indian educational context.
"""

from typing import Dict, Any, List, Literal
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
from app.models.llm.response import LLMCallResponse
from app.core.batch.openai import OpenAIBatchProvider
from openai import OpenAI

import random
import json
import string
import pandas as pd
from pathlib import Path
from typing import Any
from pydantic import BaseModel, field_validator
from app.services.llm.mappers import transform_kaapi_config_to_native
from logging import Logger
load_dotenv()

logger = Logger(__name__)

METRICS = ["novelty", "usefulness", "feasibility", "scalability", "sustainability"]
BASE_DIR = Path(__file__).parent

def system_prompt() -> Dict[str, Any]:
    """
    Returns the comprehensive evaluation system for School Innovation Marathon submissions.

    This schema evaluates student innovations across 5 critical dimensions with Indian
    educational context, ensuring fair assessment for grades 6-12 students while maintaining
    high standards for genuine innovation recognition.
    """
    return {
        "task": {
            "role": """You are a senior innovation evaluator for the School Innovation Marathon (SIM) - India's largest student innovation platform.
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
                            2. Rule 2: Prototype Impact on Scoring : Prototype evidence should only change scores if it introduces new functional insight or reduces uncertainty about how the solution works. If the prototype only confirms a known or already understood concept (like small sapling can grow in a coconut shell), it should increase evaluator confidence but not change the score.
                            3. Rule 3: Prototype vs Effort : A prototype demonstrates student effort and experimentation, which should be acknowledged in evaluation, but effort alone should not lead to higher scores without additional functional evidence.
                            4. Rule 4: Irrelevant Evidence Handling : If additional images or documents are not clearly related to the problem or solution, they must be ignored and should not influence scoring.
                            5. Rule 5: Generic Solution without Explanation : If a student suggests a widely known solution without explaining how it works or how they will implement it, assign:Low novelty (1–2),Mid-range feasibility and scalability (6–7) (because the idea is realistic, but not demonstrated).
                            6. Rule 6: Prototype Adds New Design Evidence → Scores Increase :When a prototype clearly shows the design—what the solution is, how it looks, and how it is used—and this information was not evident from the text, it provides new evidence. In such cases, scores for parameters like feasibility, usefulness, and novelty should increase.
                            7. Rule 7: Prototype Only Confirms Existing Understanding → Scores Do Not Increase : If the prototype only confirms what was already understood from the text and does not add new clarity about the design or working of the solution, it should increase evaluator confidence but should not change the scores.
                            8. Rule 8: Design Clarity vs Technical Depth (Score Ceiling) : Even when a prototype adds clear design understanding, if it does not explain deeper aspects such as why the design works better, performance, durability, or optimization, scores should not reach the highest range (9–10).


                        
                        CRITICAL SCORING GUARDRAILS:
                        - Scores MUST be justified by explicit evidence in the submission.
                        - Do NOT assume resources, infrastructure, or execution capability unless stated.
                        - If the idea and problem donot relate to each other then score least to all parameters.
                        - Generic textbook or widely known ideas must be scored low on novelty unless clear original adaptation is shown.
                        - If the model cannot clearly interpret the attachment, do not fabricate details.Score based on available text evidence only.


                        HUMAN EVALUATION PRINCIPLES:
                        - Recognize visible student effort and original thinking even if execution is incomplete
                        - Reward contextual relevance and real-world problem awareness
                        - Separate sustainability intention from operational sustainability
                        - Avoid over-penalizing grammar or presentation
                        - Do not inflate scores without evidence
                        
                        When evidence is partial:
                        → score conservatively but acknowledge effort
                        → use mid-range scores when student reasoning is present but incomplete""",

            "objective": """Evaluate student innovation submissions to identify genuinely promising ideas that demonstrate:
                    1. Real problem understanding and original thinking
                    2. Practical solutions that could realistically be implemented
                    3. Potential for meaningful impact in the Indian context
                    4. Consideration of sustainability and scalability factors
                    5. Evidence of student's own creative thinking vs. copied ideas

                    Distinguish clearly between:
                    - intention vs execution detail
                    - concept vs deployable solution
                    - originality vs adaptation

                    Your evaluation directly impacts which students advance in India's premier innovation program, so accuracy and fairness are paramount.""",

            "method": """EVALUATION PROCESS:

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
    c. Evidence from Attachments - Interpret the attachments to identify problem , Idea , Soultion design/model etc for supporting evidences if any.


STEP 3: EVIDENCE-BASED SCORING
- Score each parameter 1-10 based on specific evidence in the submission
- Provide one clear, concise reason for each score focusing on what you observed
- Be consistent: similar quality submissions should receive similar scores
- Be fair: don't penalize for age-appropriate language or presentation style
- Assess real-world execution factors: cost, infrastructure, technical complexity, maintenance

STEP 4: OUTPUT GENERATION
- Return ONLY valid JSON with exact structure specified
- Each reason should be one clear sentence explaining the score
- Ensure scores align with the detailed rubrics provided"""
        },

        "output_rules": {
            "format": "Return ONLY valid JSON object",
            "required_keys": ["image_summary", "idea_summary"],
            "structure": {
                "image_summary": "string (brief summary of what the image(s) show — prototype, diagram, sketch, experimental setup, etc., whether it appears student-made, and how well it supports the stated problem and solution)",
                "idea_summary": "string (summary of the student's idea extracted from the image(s), including the problem they are solving and the proposed solution approach)"
            }
        }
    }

### ----------------------  ITERATION BASED RESPONSE ------------------------------------------
REQUIRED_COLUMNS = {"problem", "solution"}


def _read_and_validate_file(file_path: Path) -> pd.DataFrame:
    ext = file_path.suffix.lower()
    file_data = None
    if ext == ".csv":
        file_data = pd.read_csv(file_path)
    elif ext in {".xlsx", ".xls"}:
        file_data = pd.read_excel(file_path)

    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .xlsx, or .xls")

    file_data = file_data.dropna(how="all")
    file_data.columns = file_data.columns.str.lower().str.strip()
    missing = REQUIRED_COLUMNS - set(file_data.keys())
    if missing:
        raise ValueError(f"Input dict must have both 'problem' and 'solution' keys")

    return file_data


def _validate_dict_input(input_data: List[dict[str, str]]) -> None:
    for item in input_data:
        missing = REQUIRED_COLUMNS - set(item.keys())
        if missing:
            raise ValueError("Input dict must have both 'problem' and 'solution' keys")
        if not item.get("problem") or not item.get("solution"):
            raise ValueError(
                "'problem' and 'solution' values must be non-empty strings"
            )


def _generate_id(length=5):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


class EvaluationInput(BaseModel):
    problem: str
    solution: str
    custom_id: str | None
    image: str | List[str] | None

    @field_validator("problem", "solution")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Value must be a non-empty string")
        return v.strip()

def _to_direct_url(url: str) -> str:
    import re
    """Convert Google Drive sharing links to direct download URLs."""
    match = re.match(r"https://drive\.google\.com/file/d/([^/]+)", url.strip())
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return url.strip()

def run_inference_batch(
    *,
    input: Path | List[dict[str, str]],
    output_path: str,
    provider_name: Literal["openai", "google"] = "openai",
    model_name: str = "gpt-4o-mini"
) -> None:

    data: List[dict[str, str]] | None = None

    if isinstance(input, Path):
        if os.path.exists(input):
            data = _read_and_validate_file(input)

            data = [
                EvaluationInput(
                    problem=row["problem"],
                    solution=row["solution"],
                    custom_id=str(row["cid"]) if pd.notna(row.get("cid")) else None,
                    image=str(row["documents"]) if pd.notna(row.get("documents")) else None,
                )
                for _, row in data.iterrows()
                if pd.notna(row["problem"]) and pd.notna(row["solution"])
            ]
        else:
            raise ValueError(f"File doesn't exist : {input}")

    elif isinstance(input, List):
        _validate_dict_input(input_data=input)
        data = [
            EvaluationInput(
                problem=item["problem"],
                solution=item["solution"],
                image=item["image"] if item.get("image") else None,
            )
            for item in input
            if item["problem"] and item["solution"]
        ]
    else:
        raise ValueError("make sure the input is either path or valid dict")

    if data:
        # creating batch jsonl
        # step 1: create configuration
        output = []
        system_instructions = json.dumps(system_prompt())

        completion_config = KaapiCompletionConfig(
            provider=provider_name,
            type="text",
            params=TextLLMParams(
                model=model_name,
                instructions=system_instructions,
                temperature=0.4,
                output_schema={
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
                },
            ).model_dump(exclude_none=True),
        )

        completion_config, warnings = transform_kaapi_config_to_native(
            completion_config
        )

        configuration = LLMCallConfig(blob=ConfigBlob(completion=completion_config))
        provider_class = LLMProvider.get_provider_class(provider_type=provider_name)

        if provider_name == "openai":
            credential = {"api_key": os.getenv("OPENAI_API_KEY")}
        else:
            credential = {"api_key": os.getenv("GEMINI_API_KEY")}
        client = provider_class.create_client(credentials=credential)
        provider_instance = provider_class(client=client)
        response: LLMCallResponse | None

        for input_dict in data:
            input_dict: dict = input_dict.model_dump()
            custom_id = (
                input_dict.get("custom_id") or f"student_{_generate_id(length=6)}"
            )
            problem = input_dict.get("problem")
            solution = input_dict.get("solution")
            image = input_dict.get("image", None)

            if not image:
                continue

            if isinstance(image, str):
                image = _to_direct_url(image)
            elif isinstance(image, list):
                image = [_to_direct_url(img) for img in image]

            inputs = [
                {
                    "type": "text",
                    "content": {
                        "format": "text",
                        "value": f"# Problem Statement\n{problem}\n\n# Proposed Solution\n{solution}\n\nBased on the attached image(s) and the above problem and solution, provide:\n1. **image_summary**: A brief summary of what the image shows (prototype, diagram, sketch, etc.), whether it appears student-made, and how well it supports the stated problem and solution.\n2. **idea_summary**: A summary of the student's idea extracted from the image, including the problem they are solving and the proposed solution approach.\n\nRespond in JSON format.",
                    },
                }
            ]

            if image is not None:
                images = [image] if isinstance(image, str) else image
                for img in images:
                    inputs.append({"type": "image", "content": {"format": "url", "value": img}})

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
                    logger.error(f"[run_inference_batch] Error for CID {custom_id}: {error}")
                    response_output = None
                else:
                    response_output = json.loads(response.response.output.content.value)
            except Exception as e:
                logger.error(f"[run_inference_batch] Exception for CID {custom_id}: {e}")
                response_output = None

            output.append(
                {
                    "CID": custom_id,
                    "Problem": problem,
                    "Solution": solution,
                    "Image URL": image,
                    "Image Summary": response_output.get("image_summary") if isinstance(response_output, dict) else response_output,
                    "Idea Summary": response_output.get("idea_summary") if isinstance(response_output, dict) else None,
                }
            )
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(output, f, indent=2)

    else:
        raise ValueError("batch id didn't got")


if __name__ == "__main__":
    provider_name = "openai"
    model_name = "gpt-4.1-mini"

    OUTPUT_FOLDER = os.path.join(BASE_DIR, f"summary_output_imagev1/{provider_name}/{model_name}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    OUTPUT_FILE = f"{model_name}.json"
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

    run_inference_batch(
        input=Path(os.path.join(BASE_DIR, "200 Golden_dataset_2.O-3.xlsx")),
        output_path=OUTPUT_FILE_PATH,
        provider_name=provider_name,
        model_name=model_name
    )