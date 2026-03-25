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
from app.services.llm.mappers import transform_kaapi_config_to_native
from logging import Logger

logger = Logger(__name__)

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

from typing import Literal
def build_translation_user_message(
        problem: str,
        solution: str, 
    ) -> str:
    """Builds the user message for the translation LLM call."""
    return f"""Translate the following student innovation submission to English.

Problem:
{problem}

Solution:
{solution}"""


def run_inference_batch_openai(
        input: Path | List[dict[str, str]],
        output_path: str,
        provider_name: Literal["openai", "google"] = "openai",
        model_name: str = "gpt-4o-mini"
    ):
    logger.info(f"\n\n------------------RUNNING MODEL: {provider_name}/{model_name}------------------\n\n")

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
                    custom_id=str(row["cid"]) if pd.notna(row.get("cid")) else None,
                )
                for _, row in data.iterrows()
                if pd.notna(row["problem"]) and pd.notna(row["solution"])
            ]
        else:
            raise ValueError(f"File doesn't exist : {input}")

    elif isinstance(input, List):
        _validate_dict_input(input_data=input)
        data = [
            EvaluationInput(problem=item["problem"], solution=item["solution"])
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
        completion_config = KaapiCompletionConfig(
            provider=provider_name,
            type="text",
            params=TextLLMParams(
                model=model_name,
                instructions=system_instruction,
                temperature=0.4,
                output_schema={
                    "type": "object",
                    "properties": {
                        "problem_translated": {"type": "string"},
                        "solution_translated": {"type": "string"},
                        "summarization": {"type": "string"},
                    },
                    "required": [
                        "problem_translated",
                        "solution_translated",
                        "summarization",
                    ],
                    "additionalProperties": False,
                },
            ).model_dump(exclude_none=True),
        )

        completion_config, warnings = transform_kaapi_config_to_native(completion_config)


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
            input_dict = input_dict.model_dump()
            custom_id = (
                input_dict.get("custom_id") or f"student_{_generate_id(length=6)}"
            )
            problem = input_dict.get("problem")
            solution = input_dict.get("solution")

            query = QueryParams(
                input=build_translation_user_message(problem, solution),
            )

            request = LLMCallRequest(query=query, config=configuration)

            with resolved_input_context(
                query_input=request.query.input
            ) as resolved_input:
                response, error = provider_instance.execute(
                    completion_config=request.config.blob.completion,
                    query=query,
                    resolved_input=resolved_input,
                    include_provider_raw_response=False,
                )
            response_output:dict[str, Any] | None = json.loads(response.response.output.content.value)
            output.append(
                {
                    "CID": custom_id,
                    "Problem": problem,
                    "Solution": solution,
                    "problem_translated": response_output.get("problem_translated", ""),
                    "solution_translated": response_output.get("solution_translated", ""),
                    "summarization": response_output.get("summarization", "")
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

      

if __name__ == "__main__":
    provider_name = "openai"
    model_name = "gpt-4.1-mini"

    OUTPUT_FOLDER = os.path.join(BASE_DIR, f"output_translate/{provider_name}/{model_name}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    OUTPUT_FILE = f"{model_name}.json"
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

    run_inference_batch_openai(
        Path(os.path.join(BASE_DIR, "200 Golden_dataset_2.O-3.xlsx")),
        output_path=OUTPUT_FILE_PATH,
        provider_name=provider_name,
        model_name=model_name,
    )
