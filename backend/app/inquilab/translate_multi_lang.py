"""
This code is for translating the 
indic language to english with different providers and llm models 

The motive of this code is to find the sweet spot in terms of
reliability, latency, accuracy and cost

Using kaapi own unified api llm calls for this experiment
"""

from typing import Dict, Any, List
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

load_dotenv()

BASE_DIR = Path(__file__).parent


TRANSLATION_SYSTEM_PROMPT = """You are a professional translator specializing in Indian languages. Your task is to translate student innovation submissions (problem statements and solutions) from any Indian language into clear, accurate English, and provide a brief English summary.

TRANSLATION GUIDELINES:

1. ACCURACY
   - Translate the meaning faithfully — do not add, remove, or interpret content.
   - Preserve technical terms, product names, and proper nouns as-is.
   - If a word or phrase has no direct English equivalent, use the closest natural English expression and include the original term in parentheses.

2. LANGUAGE HANDLING
   - The input may be in any Indian language: Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Odia, Punjabi, Assamese, Urdu, or others.
   - The input may be code-mixed (e.g., Hindi-English, Telugu-English). Retain the English portions as-is and translate only the non-English parts.
   - If the input is already fully in English, return it unchanged.

3. STRUCTURE
   - Maintain the original structure: if the input has separate problem and solution sections, keep them separate in the output.
   - Preserve paragraph breaks and any bullet points or numbered lists.

4. CLARITY
   - Use simple, clear English appropriate for understanding a school student's (grades 6-12) submission.
   - Fix obvious grammatical issues in the translation to produce natural-sounding English, but do not elevate the vocabulary or sophistication beyond what the student expressed.

5. SUMMARIZATION
   - After translating, write a concise English summary (2-4 sentences) that captures the core problem and the proposed solution.
   - The summary should be in simple English and help a reviewer quickly understand what the student is proposing.

6. OUTPUT FORMAT
   Return ONLY a JSON object with the following structure:
   {
     "problem_translated": "<translated problem statement in English>",
     "solution_translated": "<translated solution statement in English>",
     "summarization": "<concise English summary of the problem and solution>"
   }

   - If the input is already in English, still provide the fields with the original text.
   - Do not include any text outside the JSON object.
"""


def get_translation_system_prompt() -> str:
    """Returns the system prompt for translating Indic language submissions to English."""
    return TRANSLATION_SYSTEM_PROMPT


def build_translation_user_message(problem: str, solution: str) -> str:
    """Builds the user message for the translation LLM call."""
    return f"""Translate the following student innovation submission to English.

Problem:
{problem}

Solution:
{solution}"""

def run_inference_batch_openai(input: Path | List[dict[str, str]]):
    REQUIRED_COLUMNS = {"problem", "solution"}
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_summary_indic/inference_batch_openai_4o_mini")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    OUTPUT_FILE = "parsed_resultv1.json"
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

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
                raise ValueError(
                    "Input dict must have both 'problem' and 'solution' keys"
                )
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

        @field_validator("problem", "solution")
        @classmethod
        def must_be_non_empty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("Value must be a non-empty string")
            return v.strip()

    data: List[dict[str, str]] | None = None

    if isinstance(input, Path):
        if os.path.exists(input):
            data = _read_and_validate_file(input)

            data = [
                EvaluationInput(
                    problem=row["problem"],
                    solution=row["solution"],
                    custom_id=str(row["cid"]) if pd.notna(row.get("cid")) else None
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
                solution=item["solution"]
            )
            for item in input
            if item["problem"] and item["solution"]
        ]
    else:
        raise ValueError("make sure the input is either path or valid dict")

    
    if data:
        # creating batch jsonl
        # step 1: create configuration
        system_instruction = get_translation_system_prompt()

        output = []

        configuration = LLMCallConfig(
            blob=ConfigBlob(
                completion=KaapiCompletionConfig(
                    
                )
            )
        )

        for input_dict in data:
            input_dict = input_dict.model_dump()
            custom_id = (
                input_dict.get("custom_id") or f"student_{_generate_id(length=6)}"
            )
            problem = input_dict.get("problem")
            solution = input_dict.get("solution")




        #     request = {
        #         "custom_id": custom_id,
        #         "method": "POST",
        #         "url": "/v1/responses",
        #         "body": {
        #             "model": "gpt-4o-mini",
        #             "instructions": system_instruction,
        #             "input": build_translation_user_message(problem, solution),
        #             "text": {
        #                 "format": {
        #                     "type": "json_schema",
        #                     "name": "output",
        #                     "strict": True,
        #                     "schema": {
        #                         "type": "object",
        #                         "properties": {
        #                             "problem_translated": {"type": "string"},
        #                             "solution_translated": {"type": "string"},
        #                             "summarization": {"type": "string"}
        #                         },
        #                         "required": [
        #                             "problem_translated",
        #                             "solution_translated",
        #                             "summarization"
        #                         ],
        #                         "additionalProperties": False,
        #                     },
        #                 }
        #             },
        #         },
        #     }

        #     jsonl_data.append(request)

        # # Submit batch to openai
        # openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # provider = OpenAIBatchProvider(client=openai_client)
        # # create batch job
        # result = provider.create_batch(jsonl_data, {})

        # # wait for batch job to get suceed
        # batch_id = result.get("provider_batch_id", "")
        # batch_data: dict | None = None
        # if batch_id:
        #     while True:
        #         batch_result = provider.get_batch_status(batch_id)
        #         if batch_result.get("provider_status") in (
        #             "completed",
        #             "failed",
        #             "expired",
        #             "cancelled",
        #         ):
        #             batch_data = batch_result
        #             break

        #     output_file_id = batch_data.get("provider_output_file_id", None)
        #     if not output_file_id:
        #         raise ValueError("No output file — batch may have failed")

        #     content = provider.download_batch_results(output_file_id)

        #     parsed_results = []
        #     for item in content:
        #         cid = item["custom_id"]
        #         if item.get("error"):
        #             parsed_results.append({"custom_id": cid, "error": item["error"]})
        #         else:
        #             text = item["response"]["body"]["output"][0]["content"][0]["text"]
        #             parsed_results.append(
        #                 {"custom_id": cid, "scores": json.loads(text)}
        #             )

        #     with open(OUTPUT_FILE_PATH, "w") as f:
        #         json.dump(parsed_results, f, indent=2)

        # else:
        #     raise ValueError("batch id didn't got")



if __name__ == '__main__':
    
    run_inference_batch_openai(Path(os.path.join(BASE_DIR, "200_Golden_dataset_2.0.xlsx")))