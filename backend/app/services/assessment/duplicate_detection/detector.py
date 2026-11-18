"""Duplicate detector: the Google GCP provider plus RAG Engine search.

Flow for one submission (strict and lenient run in parallel, each with its own config):
1. Search: RAG Engine returns the top max_num_results entries from the check's vector store.

2. Tag: after the RAG search, each retrieved candidate is labelled C1, C2, ... Cn and
   wrapped in a <candidate> tag. The candidates are sent to the LLM call along with the
   input query and the respective system prompt (strict or lenient).

3. Ask: Gemini to judge the submission against the retrieved candidates and returns
   structured JSON in the shape set by the config's json_output_schema, citing
   candidates by their tags (e.g. "C3") rather than their verbose text.

4. Mapper: the code validates the answer and swaps each tag with the entry's full text.

5. Result: strict gives is_duplicate plus one citation; lenient gives scored entries in list, highest first.

Network only: no database connection is held while a provider call is in flight.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from google.auth.transport.requests import AuthorizedSession
from google.genai import types as genai_types
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import NonRecordingSpan, Status, StatusCode
from pydantic import BaseModel, ValidationError

from app.core.cloud.storage import build_gcp_sa_credentials
from app.core.config import settings
from app.core.telemetry import (
    set_gen_ai_request_attributes,
    set_gen_ai_response_attributes,
    suppress_http_instrumentation,
)
from app.models.assessment.duplicate_detection import (
    DetectionErrors,
    DetectionLevel,
    DuplicateDetectionResult,
    LenientMatch,
    StrictVerdict,
)
from app.models.config.assessment_blob import AssessmentConfigBlob
from app.models.llm import NativeCompletionConfig, QueryParams
from app.models.llm.request import CompletionType
from app.services.assessment.api.batch import PipelineStage, stage_params, stage_prompt
from app.services.assessment.validators import normalize_llm_text
from app.services.llm.mappers import map_kaapi_to_google_params
from app.services.llm.providers.google_gcp import GoogleGCPClient, GoogleGCPProvider

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

MS_PER_SECOND = 1000
LLM_OPERATION = "chat"
GEN_AI_CHAT_OP = "gen_ai.chat"


def step_context() -> otel_context.Context:
    """
    For the Sentry waterfall trace: each step (search, LLM call, callback) shows as its own bar,
    and finished steps stay visible even if the worker dies mid-run.
    """
    task_span = trace.get_current_span()

    if isinstance(task_span, ReadableSpan) and task_span.parent:
        return trace.set_span_in_context(NonRecordingSpan(task_span.parent))

    return otel_context.get_current()


class _LenientOutput(BaseModel):
    matches: list[LenientMatch]


class DuplicateDetector(GoogleGCPProvider):
    """Runs the strict and lenient checks: search the corpus -> ask the model to judge -> validate and map the answer."""

    def __init__(self, client: GoogleGCPClient):
        super().__init__(client)
        self._http = AuthorizedSession(build_gcp_sa_credentials(client.sa_info))

    def close(self) -> None:
        self._http.close()
        self.client.genai.close()

    @staticmethod
    def _build_text_config(params: dict[str, Any]) -> genai_types.GenerateContentConfig:
        """The provider's config plus the temperature and a short per-call timeout."""
        config = GoogleGCPProvider._build_text_config(params)

        config.temperature = params.get("temperature")
        # Short timeout: a stuck call would otherwise hold a worker slot until the task time limit.
        config.http_options = genai_types.HttpOptions(
            timeout=settings.DEDUP_MODEL_TIMEOUT * MS_PER_SECOND,
            retry_options=genai_types.HttpRetryOptions(
                attempts=settings.DEDUP_MODEL_ATTEMPTS
            ),
        )

        return config

    def search(
        self, corpora: list[str], query: str, top_k: int
    ) -> tuple[list[str], str | None]:
        """
        Vertex AI RAG Engine search (retrieveContexts): finds the candidates for one check.

        corpora: the vector store(s) to search, the config's knowledge_base_ids.
        query: the submission text to match against.
        top_k: how many of the most similar chunks to retrieve (the config's max_num_results).

        Returns the candidate texts in rank order, or an error message if the search failed.
        A long entry can come back as several chunks; these are joined into one candidate.
        """
        _, project, _, location, *_ = corpora[0].split("/")

        try:
            response = self._http.post(
                f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
                f"/locations/{location}:retrieveContexts",
                json={
                    "vertex_rag_store": {
                        "rag_resources": [{"rag_corpus": c} for c in corpora]
                    },
                    "query": {"text": query, "rag_retrieval_config": {"top_k": top_k}},
                },
                timeout=settings.DEDUP_SEARCH_TIMEOUT,
            )
            response.raise_for_status()

        except requests.RequestException as exc:
            status = getattr(exc.response, "status_code", None)

            error = (
                f"[VERTEX] Corpus search failed (code: {status or type(exc).__name__}). "
                f"Check the corpus name, its region and the service account's access."
            )

            logger.error(f"[DuplicateDetector.search] {error}", exc_info=True)

            return [], error

        # A long entry comes back as several chunks; joining them lets a citation be the whole entry.
        by_source: dict[str, list[str]] = {}

        for context in response.json().get("contexts", {}).get("contexts", []):
            by_source.setdefault(context.get("sourceUri", ""), []).append(
                context.get("text", "").strip()
            )

        return ["\n".join(chunks) for chunks in by_source.values()], None

    def _ask(
        self, blob: AssessmentConfigBlob, query: str, level: DetectionLevel
    ) -> tuple[str | None, dict[str, str], str | None]:
        """Model output for one level, with the tag -> corpus text map it may cite."""
        params = stage_params(blob, PipelineStage.ASSESSMENT)

        with (
            tracer.start_as_current_span(f"duplicate_detection.{level}.search") as span,
            suppress_http_instrumentation(),
        ):
            # RAG SEARCH
            candidates, error = self.search(
                params["knowledge_base_ids"], query, params["max_num_results"]
            )

            span.set_attribute("duplicate_detection.candidates", len(candidates))

            if error:
                span.set_status(Status(StatusCode.ERROR, error))

        if error or not candidates:
            return None, {}, error

        # The model answers with a tag like "C1" and the code swaps in that entry's text,
        # so a citation is always a real corpus entry and never text the model wrote.
        by_tag = {f"C{i}": text for i, text in enumerate(candidates, start=1)}

        listing = "\n\n".join(
            f'<candidate id="{tag}">\n{text}\n</candidate>'
            for tag, text in by_tag.items()
        )

        prompt = normalize_llm_text(stage_prompt(blob, PipelineStage.ASSESSMENT) or "")
        for column in blob.input_schema:
            prompt = prompt.replace("{" + column + "}", query)

        prompt = f"{prompt}\n\n<candidates>\n{listing}\n</candidates>"

        mapped, _ = map_kaapi_to_google_params(params)
        # The mapper drops temperature for thinking models; kept so the same input gets the same verdict.
        mapped["temperature"] = params.get("temperature")

        # Sentry Waterfall Trace
        with (
            tracer.start_as_current_span(f"chat {params['model']}") as span,
            suppress_http_instrumentation(),
        ):
            # Named and typed like the app's other LLM calls so Sentry lists its model and tokens.
            span.set_attribute("sentry.op", GEN_AI_CHAT_OP)
            span.set_attribute("duplicate_detection.level", level)
            set_gen_ai_request_attributes(
                span,
                provider=blob.assessment.provider,
                model=params["model"],
                operation=LLM_OPERATION,
                organization_id=None,
                project_id=None,
                params=params,
            )

            # LLM CALL: the model judges the submission against the retrieved candidates and cites them by tag.
            response, error = self.execute(
                completion_config=NativeCompletionConfig(
                    provider=f"{blob.assessment.provider}-native",
                    type=CompletionType.TEXT,
                    params=mapped,
                ),
                query=QueryParams(input=prompt),
                resolved_input=prompt,
            )

            if response is None:
                span.set_status(Status(StatusCode.ERROR, error))
                return None, by_tag, error

            set_gen_ai_response_attributes(span, response=response)

        return response.response.output.content.value, by_tag, None

    def check_strict(
        self, blob: AssessmentConfigBlob, query: str
    ) -> tuple[StrictVerdict | None, str | None]:
        """Strict verdict; a duplicate is accepted only when it cites a retrieved entry."""
        text, by_tag, error = self._ask(blob, query, DetectionLevel.STRICT)

        if error:
            return None, error

        # Nothing similar in the corpus, so nothing to be a copy of.
        if text is None:
            return StrictVerdict(is_duplicate=False), None

        try:
            verdict = StrictVerdict.model_validate_json(text)

            if not verdict.is_duplicate:
                return StrictVerdict(is_duplicate=False), None

            return (
                StrictVerdict(is_duplicate=True, citation=by_tag[verdict.citation]),
                None,
            )

        except (ValidationError, KeyError):
            return None, self._invalid_output(DetectionLevel.STRICT)

    def check_lenient(
        self, blob: AssessmentConfigBlob, query: str
    ) -> tuple[list[LenientMatch] | None, str | None]:
        """Scored matches, highest first; an entry cited twice keeps its higher score."""
        text, by_tag, error = self._ask(blob, query, DetectionLevel.LENIENT)

        if error:
            return None, error

        if text is None:
            return [], None

        try:
            best: dict[str, int] = {}

            for match in _LenientOutput.model_validate_json(text).matches:
                best[match.citation] = max(match.score, best.get(match.citation, 0))

            ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)

            return [
                LenientMatch(score=score, citation=by_tag[tag]) for tag, score in ranked
            ], None

        except (ValidationError, KeyError):
            return None, self._invalid_output(DetectionLevel.LENIENT)

    @staticmethod
    def _invalid_output(level: DetectionLevel) -> str:
        error = (
            f"[KAAPI] The model's answer for the {level} check did not match the "
            f"expected shape. Retry; if it persists, review the config's output schema."
        )

        logger.warning(f"[DuplicateDetector._invalid_output] {error}")

        return error

    def detect(
        self, *, query: str, strict: AssessmentConfigBlob, lenient: AssessmentConfigBlob
    ) -> DuplicateDetectionResult:
        """
        Run both checks in parallel, so a job takes as long as the slower check, not both.
        Each level reports its own result or error; one failing does not drop the other.
        """

        with ThreadPoolExecutor(
            max_workers=2, initializer=otel_context.attach, initargs=(step_context(),)
        ) as pool:
            # strict and lenient run in parallel, each with its own config and vector store.
            strict_future = pool.submit(self.check_strict, strict, query)
            lenient_future = pool.submit(self.check_lenient, lenient, query)

            verdict, strict_error = strict_future.result()
            matches, lenient_error = lenient_future.result()

        return DuplicateDetectionResult(
            strict=verdict,
            lenient=matches,
            errors=DetectionErrors(strict=strict_error, lenient=lenient_error),
        )
