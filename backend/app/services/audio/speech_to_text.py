"""Speech-to-Text service supporting OpenAI and Gemini providers."""
import os
import logging
import csv
import base64
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import BinaryIO, Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from uuid import uuid4
from typing import List, Literal, Dict
from google import genai
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech, RecognizeResponse
from google.api_core.client_options import ClientOptions
from openai import OpenAI

from app.services.audio.utils.ogg_to_wav_converter_downloader import (
    download_audio_from_url,
)
from app.models.evaluation import (
    ProviderConfig,
    TranscriptionRequest,
    FileData,
    WERResult,
    WERComparisonResult,
    WERBatchItem,
    WERBatchResult,
    WERSummaryStats,
    WERBatchSummary,
    WERModelStats,
    WEROverallSummary,
)
from app.services.audio.utils.calculate_wer import tokenize, calculate_wer

load_dotenv()
logger = logging.getLogger(__name__)


# Default prompt for transcription
DEFAULT_PROMPT = (
    "Generate a verbatim speech-to-text transcript of the given audio file "
    "in the same language as of the audio in the same script too. "
    "make sure the transcription as close possible to the audio provided"
)
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
STT_LOCATION = os.getenv("GOOGLE_PROJECT_LOCATION")


class SpeechToTextService:
    def __init__(self, provider, api_key: str | None = None):
        if api_key is None:
            raise ValueError(
                "Missing OpenAI API Key for Client STT Client initialization"
            )
        self.provide = provider
        self.openai_client = None
        self.gemini_client = None
        if provider == "openai":
            self.openai_client = OpenAI(api_key=api_key)
        elif provider == "gemini":
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            raise ValueError("This provider is not supported yet.")

    def transcribe_with_openai(
        self,
        audio_file: BinaryIO | str,
        model: str = "gpt-4o-transcribe",
        prompt: str | None = None,
    ):
        if self.openai_client is None:
            raise ValueError("OpenAI client is not initialized.")
        try:
            # Handle file path vs file object
            if isinstance(audio_file, str):
                audio_file = open(audio_file, "rb")

            transcription = self.openai_client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                # language="hi",
                response_format="text",
                prompt=prompt or DEFAULT_PROMPT,
            )
            logger.info(f"Successfully transcribed audio using OpenAI model: {model}")
            return transcription
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {str(e)}", exc_info=True)
            raise

    def transcribe_with_gemini(
        self,
        audio_file_path: str,
        model: str = "gemini-2.5-flash",
        prompt: str | None = None,
    ):
        if self.gemini_client is None:
            raise ValueError("Gemini client is not initialized")
        try:
            # Upload file to Geminic
            gemini_file = self.gemini_client.files.upload(file=audio_file_path)
            logger.info(f"Uploaded file to Gemini: {gemini_file.name}")

            # Generate transcription
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=[prompt or DEFAULT_PROMPT, gemini_file],
            )

            logger.info(f"Successfully transcribed audio using Gemini model: {model}")
            return response.text or None

        except Exception as e:
            logger.error(f"Gemini transcription failed: {str(e)}", exc_info=True)
            raise

    def transcribe(
        self,
        audio_file: BinaryIO | str,
        provider: Literal["openai", "gemini"] = "openai",
        model: str | None = None,
        prompt: str | None = None,
    ):
        transcription = None
        if provider == "openai":
            transcription = self.transcribe_with_openai(
                audio_file=audio_file,
                model=model or "gpt-4o-transcribe",
                prompt=prompt,
            )
            return transcription
        elif provider == "gemini":
            # Gemini requires file path, not file object
            file_path = audio_file if isinstance(audio_file, str) else audio_file.name
            transcription = self.transcribe_with_gemini(
                audio_file_path=file_path,
                model=model or "gemini-2.5-flash",
                prompt=prompt,
            )
            return transcription
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'openai' or 'gemini'."
            )


def transcribe_audio_with_chirp_v3(audio_file_path: str):
    with open(audio_file_path, "rb") as file:
        audio_content = file.read()

    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{STT_LOCATION}-speech.googleapis.com"
        )
    )

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["auto"],
        model="chirp_3",
    )
    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{STT_LOCATION}/recognizers/_",
        config=config,
        content=audio_content,
    )
    response: RecognizeResponse = client.recognize(request=request)

    transcript = None
    for result in response.results:
        transcript = result.alternatives[0].transcript
    return transcript


def transcribe_audio_with_indic_conformer(audio_file_path: str):
    indic_conformer_api_url = str(os.getenv("AI4B_STT_URL"))
    with open(audio_file_path, "rb") as file:
        audio_content = file.read()

    response = requests.post(
        url=indic_conformer_api_url,
        data={"language_code": "hi", "decoding_strategy": "ctc"},
        files={"audio_file": audio_content},
    )
    logger.info(response.json())
    transcription = response.json()["transcription"]
    return transcription


# util functions for direct usage
def transcribe_audio(
    audio_file: str,
    provider: Literal["openai", "gemini", "google-stt", "ai4b"] = "openai",
    model: str | None = None,
    api_key: str | None = None,
    prompt: str | None = None,
):
    if provider == "google-stt":
        return transcribe_audio_with_chirp_v3(audio_file_path=audio_file)
    if provider == "ai4b":
        return transcribe_audio_with_indic_conformer(audio_file_path=audio_file)
    stt_service = SpeechToTextService(provider=provider, api_key=api_key)
    return stt_service.transcribe(
        audio_file=audio_file,
        provider=provider,
        model=model,
        prompt=prompt,
    )


# STT csv parser


def process_single_csv_row(row_data):
    idx, audio_url, ground_truth = row_data

    try:
        audio_bytes, content_type = download_audio_from_url(audio_url)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "status": "success",
            "row": idx,
            "audio_url": audio_url,
            "ground_truth": ground_truth,
            "audio_base64": audio_base64,
            "media_type": content_type,
            "file_size": len(audio_bytes),
        }
    except requests.Timeout:
        return {
            "status": "error",
            "row": idx,
            "audio_url": audio_url,
            "error": "Download timeout",
        }
    except requests.RequestException as e:
        return {
            "status": "error",
            "row": idx,
            "audio_url": audio_url,
            "error": f"Download failed: {str(e)}",
        }
    except Exception as e:
        return {
            "status": "error",
            "row": idx,
            "audio_url": audio_url,
            "error": f"Unexpected error: {str(e)}",
        }


async def process_batch_csv(csv_file):
    csv_body = await csv_file.read()
    csv_content = csv_body.decode("utf-8")

    csv_reader = csv.DictReader(StringIO(csv_content))
    required_headers = {"audio_url", "ground_truth"}

    if not required_headers.issubset(csv_reader.fieldnames):
        raise ValueError(f"CSV must have headers: {required_headers}")

    rows_to_process = []
    for idx, row in enumerate(csv_reader, start=1):
        audio_url = row["audio_url"].strip()
        ground_truth = row["ground_truth"].strip()

        if not audio_url or not ground_truth:
            continue  # do not throw error. continue silently
        rows_to_process.append((idx, audio_url, ground_truth))

    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_row = {
            executor.submit(process_single_csv_row, row_data): row_data
            for row_data in rows_to_process
        }
        for future in as_completed(future_to_row):
            result = future.result()
            if result["status"] == "success":
                results.append(result)
            else:
                errors.append(result)
    return {
        "success": results,
        "errors": errors,
        "total_rows": len(rows_to_process),
        "processed": len(results),
        "failed": len(errors),
    }


def _get_api_key(provider: str) -> str | None:
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    elif provider == "gemini":
        return os.getenv("GEMINI_API_KEY")

    return None


def _transcribe_single_file_provider(
    file_data: FileData, provider_config: ProviderConfig
):
    # for execution by threadpool
    provider = provider_config.provider

    model = provider_config.model

    audio_bytes = base64.b64decode(file_data.audio_base64) or None
    media_extention = (
        file_data.media_type.split("/")[-1] if file_data.media_type else ".ogg"
    )
    temp_file = f"/tmp/{uuid4()}.{media_extention}"

    try:
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)  # type: ignore

        # extract the api key
        api_key = _get_api_key(provider)  # type: ignore

        transcript = transcribe_audio(
            audio_file=temp_file,
            provider=provider,  # type: ignore
            model=model,
            api_key=api_key,
        )

        return {
            "status": "success",
            "file_id": file_data.file_id or None,
            "ground_truth": file_data.ground_truth,
            "provider": provider,
            "model": model,
            "transcript": transcript,
        }
    except Exception as e:
        return {
            "status": "error",
            "file_id": file_data.file_id,
            "provider": provider,
            "model": model,
            "error": str(e),
        }
    finally:
        # clean the temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


# threadpool based integration of batch transcriptipn
def process_batch_transcription(files, providers, max_workers=4):
    # create all tasks and push into a list
    tasks = [
        (file_data, provider_config)
        for file_data in files
        for provider_config in providers
    ]

    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _transcribe_single_file_provider, file_data, provider_config
            ): (file_data, provider_config)
            for file_data, provider_config in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "success":
                results.append(result)
            else:
                errors.append(result)

    return {
        "success": results,
        "errors": errors,
        "total_tasks": len(tasks),
        "processed": len(results),
        "failed": len(errors),
    }


# STT evaluation route handler
async def evaluate_stt():
    pass


def calculate_wer_individual(
    ground_truth: str,
    hypothesis: str,
    mode: Literal["strict", "lenient", "both"] = "both",
) -> WERComparisonResult | WERResult:
    """
    Calculate WER for a single transcription against ground truth.

    Args:
        ground_truth: The reference/expected transcription
        hypothesis: The transcribed text to evaluate
        mode: "strict" for exact matching, "lenient" for phonetic/spelling tolerance,
              "both" to return both calculations

    Returns:
        WERComparisonResult if mode="both", otherwise WERResult for the specified mode
    """
    ref_tokens = tokenize(ground_truth)
    hyp_tokens = tokenize(hypothesis)

    ref_count = len(ref_tokens)
    hyp_count = len(hyp_tokens)  # hypothesis tokes

    if mode == "strict":
        wer, subs, dels, ins, sem = calculate_wer(ref_tokens, hyp_tokens, lenient=False)
        return WERResult(
            wer=wer,
            substitutions=subs,
            deletions=dels,
            insertions=ins,
            semantic_errors=sem,
            reference_word_count=ref_count,
            hypothesis_word_count=hyp_count,
        )

    if mode == "lenient":
        wer, subs, dels, ins, sem = calculate_wer(ref_tokens, hyp_tokens, lenient=True)
        return WERResult(
            wer=wer,
            substitutions=subs,
            deletions=dels,
            insertions=ins,
            semantic_errors=sem,
            reference_word_count=ref_count,
            hypothesis_word_count=hyp_count,
        )

    # mode == "both"
    wer_strict, s_strict, d_strict, i_strict, sem_strict = calculate_wer(
        ref_tokens, hyp_tokens, lenient=False
    )
    wer_lenient, s_lenient, d_lenient, i_lenient, sem_lenient = calculate_wer(
        ref_tokens, hyp_tokens, lenient=True
    )

    return WERComparisonResult(
        ground_truth=ground_truth,
        hypothesis=hypothesis,
        strict=WERResult(
            wer=wer_strict,
            substitutions=s_strict,
            deletions=d_strict,
            insertions=i_strict,
            semantic_errors=sem_strict,
            reference_word_count=ref_count,
            hypothesis_word_count=hyp_count,
        ),
        lenient=WERResult(
            wer=wer_lenient,
            substitutions=s_lenient,
            deletions=d_lenient,
            insertions=i_lenient,
            semantic_errors=sem_lenient,
            reference_word_count=ref_count,
            hypothesis_word_count=hyp_count,
        ),
    )


def _process_wer_item(item: WERBatchItem) -> WERBatchResult:
    result = calculate_wer_individual(
        ground_truth=item.ground_truth, hypothesis=item.hypothesis, mode="both"
    )
    # result is WERComparisonResult with strict and lenient WERResult
    return WERBatchResult(
        id=item.id,
        ground_truth=item.ground_truth,
        hypothesis=item.hypothesis,
        model=item.model,
        strict=result.strict,
        lenient=result.lenient,
    )


def calculate_wer_batch(
    items: List[WERBatchItem], max_workers: int = 4
) -> List[WERBatchResult]:
    """
    Calculate WER for multiple transcriptions in batch using ThreadPoolExecutor.

    Args:
        items: List of WERBatchItem containing id, ground_truth, and hypothesis
        max_workers: Maximum number of concurrent workers for ThreadPoolExecutor

    Returns:
        List of WERBatchResult with both strict and lenient WER for each item
    """
    results: List[WERBatchResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_wer_item, item): item for item in items}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                item = futures[future]
                logger.error(f"WER calculation failed for item {item.id}: {e}")

    return results


def calculate_wer_summary_stats(
    results: List[WERBatchResult], mode: str
) -> WERSummaryStats:
    """
    Calculate summary statistics for a list of WER results.

    Args:
        results: List of WERBatchResult from batch WER calculation
        mode: "strict" or "lenient" - which WER results to summarize

    Returns:
        WERSummaryStats with aggregate statistics
    """
    if not results:
        return WERSummaryStats(
            count=0,
            avg_wer=0.0,
            min_wer=0.0,
            max_wer=0.0,
            avg_substitutions=0.0,
            avg_deletions=0.0,
            avg_insertions=0.0,
            avg_semantic_errors=0.0,
            total_reference_words=0,
            total_hypothesis_words=0,
        )

    # Extract WER results based on mode (strict or lenient)
    wer_results = [getattr(r, mode) for r in results]
    n = len(wer_results)

    wer_values = [w.wer for w in wer_results]

    return WERSummaryStats(
        count=n,
        avg_wer=sum(wer_values) / n,
        min_wer=min(wer_values),
        max_wer=max(wer_values),
        avg_substitutions=sum(w.substitutions for w in wer_results) / n,
        avg_deletions=sum(w.deletions for w in wer_results) / n,
        avg_insertions=sum(w.insertions for w in wer_results) / n,
        avg_semantic_errors=sum(w.semantic_errors for w in wer_results) / n,
        total_reference_words=sum(w.reference_word_count for w in wer_results),
        total_hypothesis_words=sum(w.hypothesis_word_count for w in wer_results),
    )


def calculate_wer_batch_with_summary(
    items: List[WERBatchItem], max_workers: int = 4
) -> tuple[List[WERBatchResult], WERBatchSummary]:
    """
    Calculate WER for batch items and return results with summary statistics.

    Args:
        items: List of WERBatchItem containing id, ground_truth, hypothesis, and optional model
        max_workers: Maximum number of concurrent workers for ThreadPoolExecutor

    Returns:
        Tuple of (results list, summary with overall and model-wise stats)
    """
    results = calculate_wer_batch(items, max_workers)

    # Calculate overall statistics
    overall = WEROverallSummary(
        strict=calculate_wer_summary_stats(results, "strict"),
        lenient=calculate_wer_summary_stats(results, "lenient"),
    )

    # Group results by model and calculate per-model statistics
    model_groups: Dict[str, List[WERBatchResult]] = {}
    for r in results:
        if r.model:
            if r.model not in model_groups:
                model_groups[r.model] = []
            model_groups[r.model].append(r)

    by_model: List[WERModelStats] = []
    for model_name in sorted(model_groups.keys()):
        model_results = model_groups[model_name]
        by_model.append(
            WERModelStats(
                model=model_name,
                strict=calculate_wer_summary_stats(model_results, "strict"),
                lenient=calculate_wer_summary_stats(model_results, "lenient"),
            )
        )

    summary = WERBatchSummary(overall=overall, by_model=by_model)

    return results, summary


if __name__ == "__main__":
    # oai_api_key = os.getenv("OPENAI_API_KEY")
    # gemini_api_key=os.getenv("GEMINI_API_KEY")
    ai4b_file_path = "/Users/prajna/Downloads/audio_hindi_2.ogg"

    audio_file_path = "/Users/prajna/Desktop/t4d/ai-platform/backend/app/services/audio/sample_data/ogg_files/1756121051765345.ogg"
    ai4b_response = transcribe_audio_with_indic_conformer(
        audio_file_path=ai4b_file_path
    )
    print(ai4b_response)
    # stt_eval_file_path = "/Users/prajna/Desktop/t4d/ai-platform/backend/app/services/audio/audio_sample_stt.csv"

    # transcript=transcribe_audio(audio_file=audio_file_path, provider="google-stt")
    # transcript=transcribe_audio(audio_file=audio_file_path, provider="openai", api_key=oai_api_key)
    # transcript=transcribe_audio_with_chirp_v3(audio_file_path)
    # print(transcript)

    # with open(stt_eval_file_path, "rb") as file:
    #     processed = process_batch_csv(file)

    # print(processed)
