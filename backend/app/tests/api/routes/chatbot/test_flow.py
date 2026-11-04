from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlmodel import Session

from app.models import Chatbot, Config
from app.tests.utils.auth import TestAuthContext
from app.tests.utils.chatbot import (
    CHATBOTS_URL,
    ONBOARDING_NODE_ID,
    UNKNOWN_CHATBOT_ID,
    add_test_config_version,
    build_flow_edge,
    build_flow_node,
    build_onboarding_node,
    create_test_anthropic_config,
    create_test_chatbot,
    create_test_flow_version,
    flow_url,
    list_stored_flow_versions,
    publish_url,
    random_contact_ref,
    sessions_url,
)
from app.tests.utils.test_data import create_test_config

INVALID_FLOW_PREFIX = "[KAAPI] Chatbot flow is invalid: "
NO_DRAFT_MESSAGE = "[KAAPI] Chatbot has no draft flow to publish; save the flow first."
MISSING_CREDENTIALS = "Anthropic credentials are not configured for this project"
INPUT_NODE_ID = "input-1"
OUTPUT_NODE_ID = "output-1"


@pytest.fixture
def save(
    client: TestClient, user_api_key_header: dict[str, str], chatbot: Chatbot
) -> Callable[..., Response]:
    def _save(
        nodes: list[dict[str, Any]], edges: list[dict[str, Any]] | None = None
    ) -> Response:
        return client.put(
            flow_url(chatbot.id),
            headers=user_api_key_header,
            json={"nodes": nodes, "edges": edges or []},
        )

    return _save


@pytest.fixture
def publish(
    client: TestClient, user_api_key_header: dict[str, str], chatbot: Chatbot
) -> Callable[[], Response]:
    return lambda: client.post(publish_url(chatbot.id), headers=user_api_key_header)


@pytest.fixture
def uncredentialed_config(db: Session, user_api_key: TestAuthContext) -> Config:
    """An Anthropic config in the caller's project, which has no Anthropic credentials."""
    return create_test_anthropic_config(
        db, project_id=user_api_key.project_id, with_anthropic_credentials=False
    )


@pytest.fixture
def live_v1(config: Config, save: Callable[..., Response]) -> dict[str, Any]:
    """A valid first save, which goes live as v1; later saves are drafts."""
    resp = save([build_onboarding_node(config_id=config.id)])
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert (body["version"], body["is_live"]) == (1, True)
    return body


class TestSaveChatbotFlow:
    def test_first_save_creates_live_version_1_with_pinned_config_version(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
    ) -> None:
        add_test_config_version(db, config)

        resp = save(
            [
                build_flow_node(INPUT_NODE_ID, "input"),
                build_onboarding_node(config_id=config.id, config_version=None),
            ],
            [build_flow_edge("e1", INPUT_NODE_ID, ONBOARDING_NODE_ID)],
        )

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["version"] == 1
        assert body["is_live"] is True
        assert body["chatbot_id"] == str(chatbot.id)
        assert body["nodes"][1]["data"]["config_version"] == 2
        [stored] = list_stored_flow_versions(db, chatbot)
        assert (stored.version, stored.is_live) == (1, True)
        assert stored.nodes[1]["data"]["config_version"] == 2
        assert stored.nodes[1]["data"]["label"] == "Beneficiary Info Extractor"
        assert "config_version" not in stored.nodes[0]["data"]

    def test_invalid_first_save_returns_422_and_stores_nothing(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
    ) -> None:
        resp = save([build_onboarding_node(config_id=config.id, prompt=None)])

        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error.startswith(INVALID_FLOW_PREFIX)
        assert f"node '{ONBOARDING_NODE_ID}' is missing data.prompt" in error
        assert list_stored_flow_versions(db, chatbot) == []
        assert (
            client.get(flow_url(chatbot.id), headers=user_api_key_header).status_code
            == 404
        )

    def test_session_can_start_right_after_first_save(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        chatbot: Chatbot,
        live_v1: dict[str, Any],
    ) -> None:
        resp = client.post(
            sessions_url(chatbot.id),
            headers=user_api_key_header,
            json={"contact_ref": random_contact_ref()},
        )

        assert resp.status_code == 201
        body = resp.json()["data"]
        assert body["flow_version"] == 1
        assert body["active_node_id"] == ONBOARDING_NODE_ID

    def test_save_overwrites_existing_draft_in_place(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
        live_v1: dict[str, Any],
    ) -> None:
        draft = save([build_onboarding_node(config_id=config.id, prompt="Draft.")])
        draft_body = draft.json()["data"]
        new_nodes = [build_onboarding_node(config_id=config.id, prompt="Ask grade.")]

        resp = save(new_nodes)

        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == draft_body["id"]
        assert resp.json()["data"]["version"] == 2
        assert resp.json()["data"]["is_live"] is False
        live, stored_draft = list_stored_flow_versions(db, chatbot)
        assert str(live.id) == live_v1["id"]
        assert live.is_live is True
        assert stored_draft.nodes == new_nodes

    def test_save_after_publish_starts_new_draft_and_keeps_live_version(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
    ) -> None:
        live_nodes = [build_onboarding_node(config_id=config.id)]
        create_test_flow_version(db, chatbot=chatbot, nodes=live_nodes, is_live=True)
        draft_nodes = [build_onboarding_node(config_id=config.id, prompt="Ask grade.")]

        resp = save(draft_nodes)

        assert resp.status_code == 200
        assert resp.json()["data"]["version"] == 2
        assert resp.json()["data"]["is_live"] is False
        live, draft = list_stored_flow_versions(db, chatbot)
        assert (live.version, live.is_live, live.nodes) == (1, True, live_nodes)
        assert (draft.version, draft.is_live, draft.nodes) == (2, False, draft_nodes)

    def test_react_flow_keys_and_extra_ui_keys_round_trip(
        self,
        client: TestClient,
        user_api_key_header: dict[str, str],
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
        live_v1: dict[str, Any],
    ) -> None:
        onboarding = build_onboarding_node(config_id=config.id)
        onboarding["measured"] = {"width": 240, "height": 80}
        onboarding["selected"] = True
        onboarding["position"] = {"x": 12.5, "y": -3, "snapped": False}
        edge = {
            **build_flow_edge("e1", INPUT_NODE_ID, ONBOARDING_NODE_ID),
            "sourceHandle": "out",
            "targetHandle": "in",
            "animated": True,
            "data": {"intent": "greet"},
        }

        save([build_flow_node(INPUT_NODE_ID, "input"), onboarding], [edge])
        resp = client.get(flow_url(chatbot.id), headers=user_api_key_header)

        body = resp.json()["data"]
        assert (body["version"], body["is_live"]) == (2, False)
        returned_onboarding = body["nodes"][1]
        assert returned_onboarding["measured"] == {"width": 240, "height": 80}
        assert returned_onboarding["selected"] is True
        assert returned_onboarding["position"] == {
            "x": 12.5,
            "y": -3,
            "snapped": False,
        }
        assert returned_onboarding["data"]["label"] == "Beneficiary Info Extractor"
        assert body["edges"] == [
            {
                "id": "e1",
                "source": INPUT_NODE_ID,
                "target": ONBOARDING_NODE_ID,
                "sourceHandle": "out",
                "targetHandle": "in",
                "animated": True,
                "data": {"intent": "greet"},
            }
        ]

    @pytest.mark.parametrize(
        "bad_node",
        [
            {"id": "n1", "type": "bogus", "position": {"x": 0, "y": 0}},
            {"id": "", "type": "input", "position": {"x": 0, "y": 0}},
            {"id": "n1", "type": "input"},
            {
                "id": "n1",
                "type": "onboarding",
                "position": {"x": 0, "y": 0},
                "data": {"config_version": 0},
            },
        ],
        ids=["unknown-type", "empty-id", "missing-position", "config-version-zero"],
    )
    def test_malformed_node_returns_422_and_saves_nothing(
        self,
        db: Session,
        chatbot: Chatbot,
        save: Callable[..., Response],
        bad_node: dict[str, Any],
    ) -> None:
        resp = save([bad_node])

        assert resp.status_code == 422
        assert list_stored_flow_versions(db, chatbot) == []

    def test_draft_save_does_not_validate_publish_rules(
        self, save: Callable[..., Response], live_v1: dict[str, Any]
    ) -> None:
        coordinator = build_flow_node("router", "coordinator", prompt="Route")

        resp = save([coordinator, build_flow_node(INPUT_NODE_ID, "input")])

        assert resp.status_code == 200
        assert resp.json()["data"]["is_live"] is False

    def test_unknown_chatbot_returns_404(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        resp = client.put(
            flow_url(UNKNOWN_CHATBOT_ID),
            headers=user_api_key_header,
            json={"nodes": []},
        )

        assert resp.status_code == 404

    def test_other_projects_chatbot_returns_404_and_saves_nothing(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        foreign_chatbot: Chatbot,
    ) -> None:
        resp = client.put(
            flow_url(foreign_chatbot.id),
            headers=user_api_key_header,
            json={"nodes": [build_flow_node(INPUT_NODE_ID, "input")]},
        )

        assert resp.status_code == 404
        assert list_stored_flow_versions(db, foreign_chatbot) == []


class TestGetChatbotFlow:
    def test_no_versions_returns_404(
        self, client: TestClient, user_api_key_header: dict[str, str], chatbot: Chatbot
    ) -> None:
        resp = client.get(flow_url(chatbot.id), headers=user_api_key_header)

        assert resp.status_code == 404
        assert resp.json()["error"] == "Chatbot has no flow yet; save a flow first"

    def test_default_returns_live_when_no_draft(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        config: Config,
        chatbot: Chatbot,
    ) -> None:
        live = create_test_flow_version(
            db,
            chatbot=chatbot,
            nodes=[build_onboarding_node(config_id=config.id)],
            is_live=True,
        )

        resp = client.get(flow_url(chatbot.id), headers=user_api_key_header)

        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == str(live.id)
        assert resp.json()["data"]["is_live"] is True

    def test_default_returns_draft_over_live(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        config: Config,
        chatbot: Chatbot,
    ) -> None:
        nodes = [build_onboarding_node(config_id=config.id)]
        create_test_flow_version(db, chatbot=chatbot, nodes=nodes, is_live=True)
        draft = create_test_flow_version(db, chatbot=chatbot, nodes=nodes)

        resp = client.get(flow_url(chatbot.id), headers=user_api_key_header)

        assert resp.json()["data"]["id"] == str(draft.id)
        assert resp.json()["data"]["version"] == 2
        assert resp.json()["data"]["is_live"] is False

    def test_version_param_returns_that_version(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        config: Config,
        chatbot: Chatbot,
    ) -> None:
        nodes = [build_onboarding_node(config_id=config.id)]
        first = create_test_flow_version(db, chatbot=chatbot, nodes=nodes, is_live=True)
        create_test_flow_version(db, chatbot=chatbot, nodes=nodes)

        resp = client.get(
            flow_url(chatbot.id), headers=user_api_key_header, params={"version": 1}
        )

        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == str(first.id)

    def test_unknown_version_returns_404(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        config: Config,
        chatbot: Chatbot,
    ) -> None:
        create_test_flow_version(
            db, chatbot=chatbot, nodes=[build_onboarding_node(config_id=config.id)]
        )

        resp = client.get(
            flow_url(chatbot.id), headers=user_api_key_header, params={"version": 7}
        )

        assert resp.status_code == 404
        assert resp.json()["error"] == "Flow version 7 not found for this chatbot"

    def test_other_projects_chatbot_returns_404(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        foreign_chatbot: Chatbot,
    ) -> None:
        create_test_flow_version(
            db, chatbot=foreign_chatbot, nodes=[build_flow_node(INPUT_NODE_ID, "input")]
        )

        resp = client.get(flow_url(foreign_chatbot.id), headers=user_api_key_header)

        assert resp.status_code == 404
        assert resp.json()["error"] == "Chatbot not found"


class TestPublishChatbotFlow:
    def test_publishes_draft_and_unsets_previous_live(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
        publish: Callable[[], Response],
    ) -> None:
        create_test_flow_version(
            db,
            chatbot=chatbot,
            nodes=[build_onboarding_node(config_id=config.id)],
            is_live=True,
        )
        save([build_onboarding_node(config_id=config.id, prompt="Ask grade.")])

        resp = publish()

        assert resp.status_code == 200
        assert resp.json()["data"]["version"] == 2
        assert resp.json()["data"]["is_live"] is True
        previous, published = list_stored_flow_versions(db, chatbot)
        assert previous.is_live is False
        assert published.is_live is True

    def test_missing_config_version_is_pinned_to_latest_and_stored(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
        publish: Callable[[], Response],
        live_v1: dict[str, Any],
    ) -> None:
        add_test_config_version(db, config)
        save(
            [
                build_flow_node(INPUT_NODE_ID, "input"),
                build_onboarding_node(config_id=config.id, config_version=None),
            ],
            [build_flow_edge("e1", INPUT_NODE_ID, ONBOARDING_NODE_ID)],
        )

        resp = publish()

        assert resp.status_code == 200
        assert resp.json()["data"]["version"] == 2
        assert resp.json()["data"]["nodes"][1]["data"]["config_version"] == 2
        _, stored = list_stored_flow_versions(db, chatbot)
        assert stored.is_live is True
        assert stored.nodes[1]["data"]["config_version"] == 2
        assert stored.nodes[1]["data"]["label"] == "Beneficiary Info Extractor"
        assert "config_version" not in stored.nodes[0]["data"]

    def test_explicit_config_version_is_kept(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
        publish: Callable[[], Response],
        live_v1: dict[str, Any],
    ) -> None:
        add_test_config_version(db, config)
        save([build_onboarding_node(config_id=config.id, config_version=1)])

        resp = publish()

        assert resp.status_code == 200
        _, stored = list_stored_flow_versions(db, chatbot)
        assert (stored.version, stored.is_live) == (2, True)
        assert stored.nodes[0]["data"]["config_version"] == 1

    def test_no_versions_returns_409(self, publish: Callable[[], Response]) -> None:
        resp = publish()

        assert resp.status_code == 409
        assert resp.json()["error"] == NO_DRAFT_MESSAGE

    def test_publish_right_after_first_save_returns_409(
        self, publish: Callable[[], Response], live_v1: dict[str, Any]
    ) -> None:
        resp = publish()

        assert resp.status_code == 409
        assert resp.json()["error"] == NO_DRAFT_MESSAGE

    def test_latest_already_live_returns_409(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        publish: Callable[[], Response],
    ) -> None:
        create_test_flow_version(
            db,
            chatbot=chatbot,
            nodes=[build_onboarding_node(config_id=config.id)],
            is_live=True,
        )

        resp = publish()

        assert resp.status_code == 409
        assert resp.json()["error"] == NO_DRAFT_MESSAGE

    def test_unknown_chatbot_returns_404(
        self, client: TestClient, user_api_key_header: dict[str, str]
    ) -> None:
        resp = client.post(publish_url(UNKNOWN_CHATBOT_ID), headers=user_api_key_header)

        assert resp.status_code == 404


# Each case builds (nodes, edges) from the project's valid Anthropic config id.
FlowBuilder = Callable[[UUID], tuple[list[dict[str, Any]], list[dict[str, Any]]]]

INVALID_FLOW_CASES: dict[str, tuple[FlowBuilder, str]] = {
    "duplicate-node-ids": (
        lambda cid: (
            [
                build_onboarding_node(config_id=cid),
                build_onboarding_node(config_id=cid),
            ],
            [],
        ),
        f"duplicate node ids: {ONBOARDING_NODE_ID}",
    ),
    "two-input-nodes": (
        lambda cid: (
            [
                build_flow_node("in-a", "input"),
                build_flow_node("in-b", "input"),
                build_onboarding_node(config_id=cid),
            ],
            [],
        ),
        "at most 1 input node allowed (found 2)",
    ),
    "two-output-nodes": (
        lambda cid: (
            [
                build_flow_node("out-a", "output"),
                build_flow_node("out-b", "output"),
                build_onboarding_node(config_id=cid),
            ],
            [],
        ),
        "at most 1 output node allowed (found 2)",
    ),
    "edge-to-unknown-node": (
        lambda cid: (
            [
                build_flow_node(INPUT_NODE_ID, "input"),
                build_onboarding_node(config_id=cid),
            ],
            [build_flow_edge("e1", INPUT_NODE_ID, "ghost")],
        ),
        "edge 'e1' target 'ghost' is not a node in the flow",
    ),
    "edge-from-unknown-node": (
        lambda cid: (
            [
                build_flow_node(INPUT_NODE_ID, "input"),
                build_onboarding_node(config_id=cid),
            ],
            [
                build_flow_edge("e1", INPUT_NODE_ID, ONBOARDING_NODE_ID),
                build_flow_edge("e2", "ghost", ONBOARDING_NODE_ID),
            ],
        ),
        "edge 'e2' source 'ghost' is not a node in the flow",
    ),
    "coordinator-node": (
        lambda cid: (
            [build_onboarding_node(config_id=cid, node_type="coordinator")],
            [],
        ),
        f"node '{ONBOARDING_NODE_ID}' has type 'coordinator', which is not supported yet",
    ),
    "rag-node": (
        lambda cid: ([build_onboarding_node(config_id=cid, node_type="rag")], []),
        f"node '{ONBOARDING_NODE_ID}' has type 'rag', which is not supported yet",
    ),
    "missing-config-id": (
        lambda cid: ([build_onboarding_node(config_id=None)], []),
        f"node '{ONBOARDING_NODE_ID}' is missing data.config_id",
    ),
    "missing-prompt": (
        lambda cid: ([build_onboarding_node(config_id=cid, prompt=None)], []),
        f"node '{ONBOARDING_NODE_ID}' is missing data.prompt",
    ),
    "blank-prompt": (
        lambda cid: ([build_onboarding_node(config_id=cid, prompt="   ")], []),
        f"node '{ONBOARDING_NODE_ID}' is missing data.prompt",
    ),
    "unknown-config-version": (
        lambda cid: ([build_onboarding_node(config_id=cid, config_version=99)], []),
        f"node '{ONBOARDING_NODE_ID}': Version with number '99' not found",
    ),
    "no-edges-two-agents": (
        lambda cid: (
            [
                build_onboarding_node(config_id=cid),
                build_onboarding_node(config_id=cid, node_id="second"),
            ],
            [],
        ),
        "a flow without edges must contain exactly one agent node (found 2)",
    ),
    "no-edges-no-agent": (
        lambda cid: ([build_flow_node(INPUT_NODE_ID, "input")], []),
        "a flow without edges must contain exactly one agent node (found 0)",
    ),
    "edges-without-input": (
        lambda cid: (
            [
                build_onboarding_node(config_id=cid),
                build_flow_node(OUTPUT_NODE_ID, "output"),
            ],
            [build_flow_edge("e1", ONBOARDING_NODE_ID, OUTPUT_NODE_ID)],
        ),
        "a flow with edges must contain exactly one input node (found 0)",
    ),
    "input-with-two-outgoing-edges": (
        lambda cid: (
            [
                build_flow_node(INPUT_NODE_ID, "input"),
                build_onboarding_node(config_id=cid),
                build_onboarding_node(config_id=cid, node_id="second"),
            ],
            [
                build_flow_edge("e1", INPUT_NODE_ID, ONBOARDING_NODE_ID),
                build_flow_edge("e2", INPUT_NODE_ID, "second"),
            ],
        ),
        f"input node '{INPUT_NODE_ID}' must have exactly one outgoing edge (found 2)",
    ),
    "input-edge-to-output": (
        lambda cid: (
            [
                build_flow_node(INPUT_NODE_ID, "input"),
                build_flow_node(OUTPUT_NODE_ID, "output"),
                build_onboarding_node(config_id=cid),
            ],
            [
                build_flow_edge("e1", INPUT_NODE_ID, OUTPUT_NODE_ID),
                build_flow_edge("e2", ONBOARDING_NODE_ID, OUTPUT_NODE_ID),
            ],
        ),
        f"the input node's edge must lead to an agent node (target '{OUTPUT_NODE_ID}')",
    ),
}


# A representative subset: the first save shares publish's validator, so the full
# matrix is covered by the publish tests below.
FIRST_SAVE_INVALID_CASES = [
    "coordinator-node",
    "duplicate-node-ids",
    "input-edge-to-output",
    "no-edges-no-agent",
    "unknown-config-version",
]


class TestFirstSaveValidation:
    @pytest.mark.parametrize(
        "case", FIRST_SAVE_INVALID_CASES, ids=FIRST_SAVE_INVALID_CASES
    )
    def test_invalid_first_save_returns_422_and_stores_nothing(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
        case: str,
    ) -> None:
        build_flow, expected_problem = INVALID_FLOW_CASES[case]
        nodes, edges = build_flow(config.id)

        resp = save(nodes, edges)

        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error.startswith(INVALID_FLOW_PREFIX)
        assert expected_problem in error
        assert list_stored_flow_versions(db, chatbot) == []


class TestPublishChatbotFlowValidation:
    @pytest.mark.parametrize(
        "case", sorted(INVALID_FLOW_CASES), ids=sorted(INVALID_FLOW_CASES)
    )
    def test_invalid_flow_returns_422_and_stays_draft(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
        publish: Callable[[], Response],
        live_v1: dict[str, Any],
        case: str,
    ) -> None:
        build_flow, expected_problem = INVALID_FLOW_CASES[case]
        nodes, edges = build_flow(config.id)
        assert save(nodes, edges).status_code == 200

        resp = publish()

        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error.startswith(INVALID_FLOW_PREFIX)
        assert expected_problem in error
        live, draft = list_stored_flow_versions(db, chatbot)
        assert (live.version, live.is_live) == (1, True)
        assert (draft.version, draft.is_live) == (2, False)

    def test_config_from_another_project_returns_422(
        self,
        db: Session,
        chatbot: Chatbot,
        save: Callable[..., Response],
        publish: Callable[[], Response],
        live_v1: dict[str, Any],
    ) -> None:
        foreign_config = create_test_anthropic_config(db)
        save([build_onboarding_node(config_id=foreign_config.id)])

        resp = publish()

        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error.startswith(INVALID_FLOW_PREFIX)
        assert f"node '{ONBOARDING_NODE_ID}': " in error
        assert str(foreign_config.id) in error
        _, draft = list_stored_flow_versions(db, chatbot)
        assert draft.is_live is False

    def test_non_anthropic_config_returns_422(
        self,
        db: Session,
        user_api_key: TestAuthContext,
        chatbot: Chatbot,
        save: Callable[..., Response],
        publish: Callable[[], Response],
        live_v1: dict[str, Any],
    ) -> None:
        openai_config = create_test_config(
            db, project_id=user_api_key.project_id, use_kaapi_schema=True
        )
        save([build_onboarding_node(config_id=openai_config.id)])

        resp = publish()

        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error.startswith(INVALID_FLOW_PREFIX)
        assert f"node '{ONBOARDING_NODE_ID}': Chatbot only supports configs" in error
        _, draft = list_stored_flow_versions(db, chatbot)
        assert draft.is_live is False

    def test_all_problems_are_reported_together(
        self,
        config: Config,
        save: Callable[..., Response],
        publish: Callable[[], Response],
        live_v1: dict[str, Any],
    ) -> None:
        save(
            [
                build_flow_node("in-a", "input"),
                build_flow_node("in-b", "input"),
                build_onboarding_node(
                    config_id=config.id, node_id="router", node_type="coordinator"
                ),
                build_onboarding_node(config_id=config.id, prompt=None),
            ]
        )

        resp = publish()

        assert resp.status_code == 422
        error = resp.json()["error"]
        assert "at most 1 input node allowed (found 2)" in error
        assert "node 'router' has type 'coordinator'" in error
        assert f"node '{ONBOARDING_NODE_ID}' is missing data.prompt" in error
        assert error.count(INVALID_FLOW_PREFIX) == 1


class TestMissingAnthropicCredentials:
    def test_first_save_returns_422_and_stores_nothing(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        uncredentialed_config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
    ) -> None:
        resp = save([build_onboarding_node(config_id=uncredentialed_config.id)])

        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error.startswith(INVALID_FLOW_PREFIX)
        assert MISSING_CREDENTIALS in error
        assert error.count("[KAAPI]") == 1
        assert list_stored_flow_versions(db, chatbot) == []
        assert (
            client.get(flow_url(chatbot.id), headers=user_api_key_header).status_code
            == 404
        )

    def test_publish_returns_422_and_stays_draft(
        self,
        db: Session,
        uncredentialed_config: Config,
        chatbot: Chatbot,
        publish: Callable[[], Response],
    ) -> None:
        # Stored directly: the credentials were removed after the flow went live.
        nodes = [build_onboarding_node(config_id=uncredentialed_config.id)]
        create_test_flow_version(db, chatbot=chatbot, nodes=nodes, is_live=True)
        create_test_flow_version(db, chatbot=chatbot, nodes=nodes)

        resp = publish()

        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error.startswith(INVALID_FLOW_PREFIX)
        assert MISSING_CREDENTIALS in error
        live, draft = list_stored_flow_versions(db, chatbot)
        assert (live.version, live.is_live) == (1, True)
        assert (draft.version, draft.is_live) == (2, False)


def _stored_updated_at(db: Session, chatbot: Chatbot) -> datetime:
    db.expire_all()
    stored = db.get(Chatbot, chatbot.id)
    assert stored is not None
    return stored.updated_at


def _newest_chatbot_id(client: TestClient, headers: dict[str, str]) -> str:
    resp = client.get(CHATBOTS_URL, headers=headers, params={"limit": 1})
    assert resp.status_code == 200
    return resp.json()["data"][0]["id"]


SaveBranchSetup = Callable[[Session, Chatbot, Config, Callable[..., Response]], None]


def _no_flow_yet(
    db: Session, chatbot: Chatbot, config: Config, save: Callable[..., Response]
) -> None:
    pass


def _live_and_draft(
    db: Session, chatbot: Chatbot, config: Config, save: Callable[..., Response]
) -> None:
    assert save([build_onboarding_node(config_id=config.id)]).status_code == 200
    assert (
        save([build_onboarding_node(config_id=config.id, prompt="Draft.")]).status_code
        == 200
    )


def _live_only(
    db: Session, chatbot: Chatbot, config: Config, save: Callable[..., Response]
) -> None:
    create_test_flow_version(
        db,
        chatbot=chatbot,
        nodes=[build_onboarding_node(config_id=config.id)],
        is_live=True,
    )


SAVE_BRANCH_SETUPS: dict[str, SaveBranchSetup] = {
    "first-live-save": _no_flow_yet,
    "draft-overwrite": _live_and_draft,
    "new-draft-after-live": _live_only,
}


class TestFlowEditsBumpChatbotUpdatedAt:
    @pytest.mark.parametrize(
        "branch", sorted(SAVE_BRANCH_SETUPS), ids=sorted(SAVE_BRANCH_SETUPS)
    )
    def test_save_moves_chatbot_to_top_of_list(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        user_api_key: TestAuthContext,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
        branch: str,
    ) -> None:
        SAVE_BRANCH_SETUPS[branch](db, chatbot, config, save)
        newer = create_test_chatbot(db, project=user_api_key.project)
        assert _newest_chatbot_id(client, user_api_key_header) == str(newer.id)
        before = _stored_updated_at(db, chatbot)

        resp = save([build_onboarding_node(config_id=config.id, prompt="Ask grade.")])

        assert resp.status_code == 200
        assert _stored_updated_at(db, chatbot) > before
        assert _newest_chatbot_id(client, user_api_key_header) == str(chatbot.id)

    def test_publish_moves_chatbot_to_top_of_list(
        self,
        client: TestClient,
        db: Session,
        user_api_key_header: dict[str, str],
        user_api_key: TestAuthContext,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
        publish: Callable[[], Response],
        live_v1: dict[str, Any],
    ) -> None:
        save([build_onboarding_node(config_id=config.id, prompt="Ask grade.")])
        newer = create_test_chatbot(db, project=user_api_key.project)
        assert _newest_chatbot_id(client, user_api_key_header) == str(newer.id)
        before = _stored_updated_at(db, chatbot)

        resp = publish()

        assert resp.status_code == 200
        assert _stored_updated_at(db, chatbot) > before
        assert _newest_chatbot_id(client, user_api_key_header) == str(chatbot.id)

    def test_rejected_first_save_leaves_updated_at_unchanged(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
    ) -> None:
        before = _stored_updated_at(db, chatbot)

        resp = save([build_onboarding_node(config_id=config.id, prompt=None)])

        assert resp.status_code == 422
        assert _stored_updated_at(db, chatbot) == before

    def test_rejected_publish_leaves_updated_at_unchanged(
        self,
        db: Session,
        config: Config,
        chatbot: Chatbot,
        save: Callable[..., Response],
        publish: Callable[[], Response],
        live_v1: dict[str, Any],
    ) -> None:
        save([build_onboarding_node(config_id=config.id, prompt=None)])
        before = _stored_updated_at(db, chatbot)

        resp = publish()

        assert resp.status_code == 422
        assert _stored_updated_at(db, chatbot) == before
