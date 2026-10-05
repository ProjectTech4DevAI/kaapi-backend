import json
from typing import Any

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.routes.agent import _extract_forwardable_credentials
from app.core.config import settings
from app.tests.utils.agent import (
    anthropic_message,
    install_fake_anthropic,
    project_access_token,
    text_message,
    tool_use_message,
)
from app.services.agent.tools import READ_ONLY_TOOLS
from app.services.agent.prompts import AGENT_EMPTY_ANSWER_MESSAGE, AGENT_REFUSAL_MESSAGE
from app.models.llm import build_kaapi_completion_config
from app.models.llm.request import ConfigBlob, PromptTemplate
from app.models.stt_evaluation import EvaluationType
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.document import DocumentStore
from app.tests.utils.speech_evaluation import (
    create_test_speech_dataset,
    create_test_speech_run,
    create_test_stt_result,
    create_test_stt_sample,
    create_test_tts_result,
)
from app.tests.utils.test_data import (
    create_test_config,
    create_test_evaluation_dataset,
    create_test_evaluation_run,
)

AGENT_URL = f"{settings.API_V1_STR}/agent"


class TestAgentQueryEndToEnd:
    def test_lists_seeded_eval_runs_and_answers(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        run = create_test_evaluation_run(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        fake = install_fake_anthropic(
            monkeypatch,
            [
                tool_use_message("list_evaluation_runs", {"limit": 5}),
                text_message(f"You have one run: {run.run_name}."),
            ],
        )

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "List my runs"}
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["answer"] == f"You have one run: {run.run_name}."
        assert body["stop_reason"] == "completed"
        assert body["iterations"] == 1
        assert body["usage"] == {"input_tokens": 20, "output_tokens": 10}
        assert len(body["tool_calls"]) == 1
        tool_call = body["tool_calls"][0]
        assert tool_call["name"] == "list_evaluation_runs"
        assert tool_call["status_code"] == 200
        assert tool_call["is_error"] is False
        assert tool_call["arguments"] == {"limit": 5, "offset": 0}

        assert len(fake.calls) == 2
        [tool_result] = fake.tool_results(1)
        assert tool_result["tool_use_id"] == "toolu_1"
        assert tool_result["is_error"] is False
        page = json.loads(tool_result["content"])
        assert page["metadata"] == {"has_more": False}
        assert [r["id"] for r in page["data"]] == [run.id]
        assert page["data"][0]["run_name"] == run.run_name

    def test_eval_run_projection_hides_traces_and_storage_urls(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        run = create_test_evaluation_run(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            score={
                "summary_scores": [{"name": "accuracy", "avg": 0.75}],
                "traces": [{"trace_id": "trace-secret-1", "question": "Q1"}],
            },
            object_store_url="s3://internal-bucket/run.csv",
        )
        fake = install_fake_anthropic(
            monkeypatch,
            [
                tool_use_message("get_evaluation_run", {"evaluation_id": run.id}),
                text_message("Accuracy 0.75."),
            ],
        )

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Score of my run?"}
        )

        assert resp.status_code == 200
        [tool_result] = fake.tool_results(1)
        projected = json.loads(tool_result["content"])
        assert projected["id"] == run.id
        assert projected["score"] == {
            "summary_scores": [{"name": "accuracy", "avg": 0.75}]
        }
        assert "trace-secret-1" not in fake.sent_payload()
        assert "internal-bucket" not in fake.sent_payload()

    @pytest.mark.parametrize(
        "tool_name",
        [
            tool.name
            for tool in READ_ONLY_TOOLS
            if not any(f.is_required() for f in tool.args_model.model_fields.values())
        ],
    )
    def test_every_argless_tool_reaches_its_real_route(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
        tool_name: str,
    ) -> None:
        fake = install_fake_anthropic(
            monkeypatch, [tool_use_message(tool_name, {}), text_message("ok")]
        )

        resp = client.post(AGENT_URL, headers=user_api_key_header, json={"query": "q"})

        assert resp.status_code == 200
        [tool_call] = resp.json()["data"]["tool_calls"]
        assert tool_call["status_code"] == 200, fake.tool_results(1)[0]["content"]


def _single_tool_result(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
    tool_name: str,
    arguments: dict[str, object],
) -> tuple[Any, str, str]:
    fake = install_fake_anthropic(
        monkeypatch, [tool_use_message(tool_name, arguments), text_message("ok")]
    )

    resp = client.post(AGENT_URL, headers=headers, json={"query": "q"})

    assert resp.status_code == 200
    [tool_call] = resp.json()["data"]["tool_calls"]
    # The inner route must have answered, or "field absent" proves nothing.
    assert tool_call["status_code"] == 200, fake.tool_results(1)[0]["content"]
    [tool_result] = fake.tool_results(1)
    raw = tool_result["content"]
    return json.loads(raw), raw, fake.sent_payload()


class TestAgentDataMinimization:
    @pytest.mark.parametrize(
        "tool_name", ["get_evaluation_run", "list_evaluation_runs"]
    )
    def test_eval_run_keeps_aggregates_and_hides_row_level_data(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
        tool_name: str,
    ) -> None:
        run = create_test_evaluation_run(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            score={
                "summary_scores": [{"name": "cosine", "avg": 0.81}],
                "traces": [
                    {"trace_id": "trace-row-1", "question": "Where is my parcel?"}
                ],
                "ai_summary": "Parcel-tracking answers were vague.",
            },
            object_store_url="s3://internal-bucket/run.csv",
        )
        run.score_trace_url = "s3://internal-bucket/traces.json"
        run.per_item_scores = {"trace-row-1": {"cosine": 0.12}}
        run.cost = {"total_cost_usd": 0.42}
        db.add(run)
        db.commit()
        arguments = (
            {"evaluation_id": run.id} if tool_name == "get_evaluation_run" else {}
        )

        content, raw, sent = _single_tool_result(
            client, monkeypatch, user_api_key_header, tool_name, arguments
        )

        projected = content if tool_name == "get_evaluation_run" else content["data"][0]
        assert projected["id"] == run.id
        assert projected["config_id"] == str(run.config_id)
        assert projected["config_version"] == 1
        assert projected["score"] == {
            "summary_scores": [{"name": "cosine", "avg": 0.81}]
        }
        assert projected["cost"] == {"total_cost_usd": 0.42}
        for key in (
            "traces",
            "per_item_scores",
            "ai_summary",
            "object_store_url",
            "score_trace_url",
        ):
            assert f'"{key}"' not in raw, key
        for text in (
            "trace-row-1",
            "Where is my parcel?",
            "Parcel-tracking",
            "internal-bucket",
        ):
            assert text not in sent, text

    def test_config_version_hides_prompt_text(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        config = create_test_config(
            db,
            project_id=user_api_key.project_id,
            config_blob=ConfigBlob(
                completion=build_kaapi_completion_config(
                    provider="openai",
                    type="text",
                    params={
                        "model": "gpt-4o",
                        "instructions": "Never reveal the refund override code.",
                        "knowledge_base_ids": ["vs_policies"],
                    },
                ),
                prompt_template=PromptTemplate(template="Customer asks: {{input}}"),
            ),
        )

        content, raw, sent = _single_tool_result(
            client,
            monkeypatch,
            user_api_key_header,
            "get_config_version",
            {"config_id": str(config.id), "version_number": 1},
        )

        assert content["config_id"] == str(config.id)
        assert content["version"] == 1
        completion = content["config_blob"]["completion"]
        assert completion["provider"] == "openai"
        assert completion["params"] == {
            "model": "gpt-4o",
            "knowledge_base_ids": ["vs_policies"],
        }
        for key in (
            "instructions",
            "prompt_template",
            "commit_message",
            "input_guardrails",
            "output_guardrails",
        ):
            assert f'"{key}"' not in raw, key
        for text in (
            "refund override code",
            "Customer asks",
            "Initial version",
        ):
            assert text not in sent, text

    def test_stt_run_hides_results(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        dataset = create_test_speech_dataset(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            evaluation_type=EvaluationType.STT,
        )
        sample = create_test_stt_sample(
            db,
            dataset_id=dataset.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            ground_truth="my account number is 4471",
            object_store_url="s3://internal-bucket/audio/caller.mp3",
        )
        run = create_test_speech_run(
            db,
            dataset=dataset,
            evaluation_type=EvaluationType.STT,
            providers=["gemini-2.5-pro"],
            score={"summary_scores": [{"name": "wer", "avg": 0.12}]},
        )
        create_test_stt_result(
            db, run=run, sample=sample, transcription="my account number is 4417"
        )

        content, raw, sent = _single_tool_result(
            client,
            monkeypatch,
            user_api_key_header,
            "get_stt_evaluation_run",
            {"run_id": run.id},
        )

        assert content["id"] == run.id
        assert content["dataset_id"] == dataset.id
        assert content["models"] == ["gemini-2.5-pro"]
        assert content["score"] == {"summary_scores": [{"name": "wer", "avg": 0.12}]}
        for key in (
            "results",
            "results_total",
            "run_metadata",
        ):
            assert f'"{key}"' not in raw, key
        for text in (
            "account number",
            "4471",
            "4417",
            "internal-bucket",
        ):
            assert text not in sent, text

    def test_tts_run_hides_results(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        dataset = create_test_speech_dataset(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            evaluation_type=EvaluationType.TTS,
        )
        run = create_test_speech_run(
            db,
            dataset=dataset,
            evaluation_type=EvaluationType.TTS,
            providers=["gemini-2.5-pro-preview-tts"],
        )
        create_test_tts_result(
            db,
            run=run,
            sample_text="Your OTP is 902211",
            object_store_url="s3://internal-bucket/tts/otp.wav",
        )

        content, raw, sent = _single_tool_result(
            client,
            monkeypatch,
            user_api_key_header,
            "get_tts_evaluation_run",
            {"run_id": run.id},
        )

        assert content["id"] == run.id
        assert content["run_name"] == run.run_name
        assert content["models"] == ["gemini-2.5-pro-preview-tts"]
        for key in (
            "results",
            "results_total",
            "run_metadata",
        ):
            assert f'"{key}"' not in raw, key
        for text in (
            "902211",
            "internal-bucket",
        ):
            assert text not in sent, text

    def test_stt_dataset_hides_samples_and_storage_url(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        dataset = create_test_speech_dataset(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            evaluation_type=EvaluationType.STT,
            description="support calls",
            object_store_url="s3://internal-bucket/datasets/calls.csv",
            dataset_metadata={"sample_count": 1, "has_ground_truth_count": 1},
        )
        create_test_stt_sample(
            db,
            dataset_id=dataset.id,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
            ground_truth="my date of birth is 3 March",
            object_store_url="s3://internal-bucket/audio/dob.mp3",
        )

        content, raw, sent = _single_tool_result(
            client,
            monkeypatch,
            user_api_key_header,
            "get_stt_evaluation_dataset",
            {"dataset_id": dataset.id},
        )

        assert content["id"] == dataset.id
        assert content["name"] == dataset.name
        assert content["description"] == "support calls"
        assert content["dataset_metadata"] == {
            "sample_count": 1,
            "has_ground_truth_count": 1,
        }
        for key in (
            "samples",
            "object_store_url",
            "signed_url",
        ):
            assert f'"{key}"' not in raw, key
        for text in (
            "date of birth",
            "internal-bucket",
        ):
            assert text not in sent, text


PAGE_SIZE = 3


def _page_through(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
    tool_name: str,
    pages_args: list[dict[str, object]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Sequential turns, one call each: parallel inner requests would share the
    # test's DB session across threads.
    fake = install_fake_anthropic(
        monkeypatch,
        [
            *(
                tool_use_message(tool_name, args, tool_use_id=f"toolu_page_{i}")
                for i, args in enumerate(pages_args)
            ),
            text_message("done"),
        ],
    )

    resp = client.post(AGENT_URL, headers=headers, json={"query": "Count them all"})

    assert resp.status_code == 200
    tool_calls = resp.json()["data"]["tool_calls"]
    assert [call["status_code"] for call in tool_calls] == [200] * len(pages_args)
    pages = []
    for call_index in range(1, len(pages_args) + 1):
        [tool_result] = fake.tool_results(call_index)
        assert tool_result["is_error"] is False
        pages.append(json.loads(tool_result["content"]))
    return pages, tool_calls


class TestAgentPagination:
    def test_eval_runs_page_by_offset_without_gaps_or_duplicates(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        # A shared dataset isolates these runs from any others in the project.
        dataset = create_test_evaluation_dataset(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        seeded = [
            create_test_evaluation_run(
                db,
                organization_id=user_api_key.organization_id,
                project_id=user_api_key.project_id,
                dataset=dataset,
            )
            for _ in range(PAGE_SIZE + 1)
        ]
        newest_first = [run.id for run in reversed(seeded)]

        (first, second), tool_calls = _page_through(
            client,
            monkeypatch,
            user_api_key_header,
            "list_evaluation_runs",
            [
                {"limit": PAGE_SIZE, "dataset_id": dataset.id},
                {"limit": PAGE_SIZE, "offset": PAGE_SIZE, "dataset_id": dataset.id},
            ],
        )

        assert first["metadata"] == {"has_more": True}
        assert [r["id"] for r in first["data"]] == newest_first[:PAGE_SIZE]
        assert second["metadata"] == {"has_more": False}
        assert [r["id"] for r in second["data"]] == newest_first[PAGE_SIZE:]
        assert tool_calls[0]["arguments"] == {
            "limit": PAGE_SIZE,
            "offset": 0,
            "dataset_id": dataset.id,
        }

    def test_documents_page_by_skip_without_gaps_or_duplicates(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        # DocumentStore empties the documents table first, so only these are listed.
        store = DocumentStore(db=db, project_id=user_api_key.project_id)
        seeded_ids = {str(doc.id) for doc in store.fill(PAGE_SIZE + 1)}

        (first, second), tool_calls = _page_through(
            client,
            monkeypatch,
            user_api_key_header,
            "list_documents",
            [
                {"limit": PAGE_SIZE},
                {"limit": PAGE_SIZE, "skip": PAGE_SIZE},
            ],
        )

        first_ids = [d["id"] for d in first["data"]]
        second_ids = [d["id"] for d in second["data"]]
        assert first["metadata"] == {"has_more": True}
        assert len(first_ids) == PAGE_SIZE
        assert second["metadata"] == {"has_more": False}
        assert len(second_ids) == 1
        combined = first_ids + second_ids
        assert len(combined) == len(set(combined))
        assert set(combined) == seeded_ids
        assert tool_calls[1]["arguments"] == {"limit": PAGE_SIZE, "skip": PAGE_SIZE}


class TestAgentTenantIsolation:
    def test_other_projects_eval_run_is_not_visible(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        superuser_api_key: TestAuthContext,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        assert superuser_api_key.project_id != user_api_key.project_id
        other_run = create_test_evaluation_run(
            db,
            organization_id=superuser_api_key.organization_id,
            project_id=superuser_api_key.project_id,
        )
        fake = install_fake_anthropic(
            monkeypatch,
            [
                # Separate turns: parallel inner requests would share the test's
                # single DB session across threads, which SQLAlchemy doesn't allow.
                tool_use_message(
                    "get_evaluation_run",
                    {"evaluation_id": other_run.id},
                    tool_use_id="toolu_get",
                ),
                tool_use_message(
                    "list_evaluation_runs", {"limit": 50}, tool_use_id="toolu_list"
                ),
                text_message("Not found."),
            ],
        )

        resp = client.post(
            AGENT_URL,
            headers=user_api_key_header,
            json={"query": f"Show eval run {other_run.id}"},
        )

        assert resp.status_code == 200
        get_call, list_call = resp.json()["data"]["tool_calls"]
        assert get_call["name"] == "get_evaluation_run"
        assert get_call["is_error"] is True
        assert get_call["status_code"] == 404
        assert list_call["status_code"] == 200

        [get_result] = fake.tool_results(1)
        [list_result] = fake.tool_results(2)
        assert get_result["tool_use_id"] == "toolu_get"
        assert get_result["is_error"] is True
        assert json.loads(get_result["content"])["status_code"] == 404
        assert list_result["tool_use_id"] == "toolu_list"
        listed_ids = [r["id"] for r in json.loads(list_result["content"])["data"]]
        assert other_run.id not in listed_ids
        assert other_run.run_name not in fake.sent_payload()

    def test_config_version_scoped_to_caller_project(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        superuser_api_key: TestAuthContext,
        user_api_key: TestAuthContext,
        user_api_key_header: dict[str, str],
    ) -> None:
        own_config = create_test_config(db, project_id=user_api_key.project_id)
        other_config = create_test_config(db, project_id=superuser_api_key.project_id)
        fake = install_fake_anthropic(
            monkeypatch,
            [
                tool_use_message(
                    "get_config_version",
                    {"config_id": str(own_config.id), "version_number": 1},
                    tool_use_id="toolu_own",
                ),
                tool_use_message(
                    "get_config_version",
                    {"config_id": str(other_config.id), "version_number": 1},
                    tool_use_id="toolu_other",
                ),
                text_message("done"),
            ],
        )

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Show configs"}
        )

        assert resp.status_code == 200
        own_call, other_call = resp.json()["data"]["tool_calls"]
        assert own_call["status_code"] == 200
        assert other_call["status_code"] == 404
        [own_result] = fake.tool_results(1)
        assert json.loads(own_result["content"])["config_id"] == str(own_config.id)
        assert other_config.name not in fake.sent_payload()


class TestAgentCredentialSecrecy:
    @pytest.mark.parametrize("channel", ["api_key", "bearer", "cookie"])
    def test_credential_never_reaches_model_or_response(
        self,
        client: TestClient,
        db: Session,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key: TestAuthContext,
        channel: str,
    ) -> None:
        create_test_evaluation_run(
            db,
            organization_id=user_api_key.organization_id,
            project_id=user_api_key.project_id,
        )
        headers: dict[str, str] = {}
        if channel == "api_key":
            secret = user_api_key.key
            headers["X-API-KEY"] = secret
        elif channel == "bearer":
            secret = project_access_token(user_api_key)
            headers["Authorization"] = f"Bearer {secret}"
        else:
            secret = project_access_token(user_api_key)
            client.cookies.set("access_token", secret)
        fake = install_fake_anthropic(
            monkeypatch,
            [tool_use_message("list_evaluation_runs", {}), text_message("Done.")],
        )

        resp = client.post(AGENT_URL, headers=headers, json={"query": "List runs"})

        assert resp.status_code == 200
        # The inner call must have authenticated, or "secret absent" proves nothing.
        assert resp.json()["data"]["tool_calls"][0]["status_code"] == 200
        # Strip the "ApiKey " prefix too: the bare key must not leak either.
        bare_secret = secret.removeprefix("ApiKey ")
        assert bare_secret not in fake.sent_payload()
        assert bare_secret not in resp.text


def _anthropic_http_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        json={"type": "error", "error": {"message": "upstream said no"}},
    )


class TestAgentStopReasons:
    def test_iteration_cap_disables_tools_and_reports_limit(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
    ) -> None:
        monkeypatch.setattr(settings, "AGENT_MAX_ITERATIONS", 1)
        # A single script entry is replayed forever: a model that never stops asking.
        fake = install_fake_anthropic(
            monkeypatch, [tool_use_message("list_evaluation_runs", {})]
        )

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Loop forever"}
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["stop_reason"] == "iteration_limit"
        assert body["iterations"] == 1
        assert body["answer"] == AGENT_EMPTY_ANSWER_MESSAGE
        assert len(body["tool_calls"]) == 1
        assert [call["tool_choice"] for call in fake.calls] == [
            {"type": "auto"},
            {"type": "none"},
        ]
        # Tools stay in the request even when disabled so the cached prefix is stable.
        assert fake.calls[1]["tools"] == fake.calls[0]["tools"]

    def test_refusal_without_text_returns_fixed_decline(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
    ) -> None:
        install_fake_anthropic(monkeypatch, [anthropic_message([], "refusal")])

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Do something bad"}
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["stop_reason"] == "refusal"
        assert body["answer"] == AGENT_REFUSAL_MESSAGE
        assert body["tool_calls"] == []

    def test_max_tokens_keeps_partial_text(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
    ) -> None:
        install_fake_anthropic(
            monkeypatch, [text_message("Partial answer", stop_reason="max_tokens")]
        )

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Long question"}
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["stop_reason"] == "max_tokens"
        assert body["answer"] == "Partial answer"


class TestAgentErrors:
    def test_missing_anthropic_key_returns_503(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
    ) -> None:
        fake = install_fake_anthropic(monkeypatch, [text_message("unused")])
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Anything"}
        )

        assert resp.status_code == 503
        assert "not configured" in resp.json()["error"]
        assert fake.calls == []

    @pytest.mark.parametrize(
        "exc, expected_fragment",
        [
            (
                anthropic.APIStatusError(
                    "Error code: 500",
                    response=_anthropic_http_response(500),
                    body=None,
                ),
                "code: 500",
            ),
            (
                anthropic.RateLimitError(
                    "Error code: 429",
                    response=_anthropic_http_response(429),
                    body=None,
                ),
                "Rate limit exceeded",
            ),
        ],
        ids=["status_error", "rate_limit"],
    )
    def test_anthropic_failure_returns_502(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        user_api_key_header: dict[str, str],
        exc: anthropic.APIError,
        expected_fragment: str,
    ) -> None:
        install_fake_anthropic(monkeypatch, [exc])

        resp = client.post(
            AGENT_URL, headers=user_api_key_header, json={"query": "Anything"}
        )

        assert resp.status_code == 502
        error = resp.json()["error"]
        assert "[ANTHROPIC]" in error
        assert expected_fragment in error
        # Upstream bodies can carry request internals; only the curated message is shown.
        assert "upstream said no" not in error

    def test_requires_authentication(self, client: TestClient) -> None:
        resp = client.post(AGENT_URL, json={"query": "Anything"})

        assert resp.status_code == 401

    def test_empty_query_rejected(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        resp = client.post(AGENT_URL, headers=user_api_key_header, json={"query": ""})

        assert resp.status_code == 422


class TestExtractForwardableCredentials:
    @pytest.mark.parametrize(
        "api_key, bearer_token, cookie_token, expected",
        [
            ("ApiKey abc", None, None, {"X-API-KEY": "ApiKey abc"}),
            (None, "tok", None, {"Authorization": "Bearer tok"}),
            (None, None, "cookietok", {"Authorization": "Bearer cookietok"}),
            (
                "ApiKey abc",
                "tok",
                None,
                {"X-API-KEY": "ApiKey abc", "Authorization": "Bearer tok"},
            ),
            (None, "tok", "cookietok", {"Authorization": "Bearer tok"}),
            (None, None, None, {}),
        ],
        ids=[
            "api_key_only",
            "bearer_only",
            "cookie_only",
            "api_key_and_bearer",
            "bearer_beats_cookie",
            "nothing_forwardable",
        ],
    )
    def test_extracts_in_auth_priority_order(
        self,
        api_key: str | None,
        bearer_token: str | None,
        cookie_token: str | None,
        expected: dict[str, str],
    ) -> None:
        assert (
            _extract_forwardable_credentials(
                api_key=api_key,
                bearer_token=bearer_token,
                cookie_token=cookie_token,
            )
            == expected
        )
