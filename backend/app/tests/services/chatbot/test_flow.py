from typing import Any
from uuid import UUID

import pytest
from sqlmodel import Session

from app.models import ChatbotFlowEdge, ChatbotFlowNode, Config
from app.services.chatbot.exceptions import ChatbotFlowValidationError
from app.services.chatbot.flow import resolve_entry_node, validate_flow_for_publish
from app.tests.utils.chatbot import (
    ONBOARDING_NODE_ID,
    add_test_config_version,
    build_flow_edge,
    build_flow_node,
    build_onboarding_node,
    create_test_anthropic_config,
)

CONFIG_ID = UUID("00000000-0000-0000-0000-00000000c0f1")


def _node(node_id: str, node_type: str, **data: Any) -> ChatbotFlowNode:
    return ChatbotFlowNode.model_validate(build_flow_node(node_id, node_type, **data))


def _onboarding(
    node_id: str = ONBOARDING_NODE_ID, config_id: UUID = CONFIG_ID, **overrides: Any
) -> ChatbotFlowNode:
    return ChatbotFlowNode.model_validate(
        build_onboarding_node(config_id=config_id, node_id=node_id, **overrides)
    )


def _edge(edge_id: str, source: str, target: str) -> ChatbotFlowEdge:
    return ChatbotFlowEdge.model_validate(build_flow_edge(edge_id, source, target))


class TestResolveEntryNode:
    def test_no_edges_returns_the_single_agent_node(self) -> None:
        nodes = [_node("in", "input"), _onboarding(), _node("out", "output")]

        entry = resolve_entry_node(nodes=nodes, edges=[])

        assert entry.id == ONBOARDING_NODE_ID

    def test_with_edges_returns_target_of_the_input_nodes_edge(self) -> None:
        nodes = [
            _node("in", "input"),
            _onboarding("first"),
            _onboarding("second"),
            _node("out", "output"),
        ]
        edges = [
            _edge("e1", "in", "second"),
            _edge("e2", "second", "first"),
            _edge("e3", "first", "out"),
        ]

        entry = resolve_entry_node(nodes=nodes, edges=edges)

        assert entry.id == "second"

    def test_input_edge_may_lead_to_any_agent_type(self) -> None:
        nodes = [_node("in", "input"), _node("router", "coordinator")]

        entry = resolve_entry_node(nodes=nodes, edges=[_edge("e1", "in", "router")])

        assert entry.id == "router"

    @pytest.mark.parametrize(
        ("nodes", "edges", "expected_problem"),
        [
            (
                [_node("in", "input")],
                [],
                "a flow without edges must contain exactly one agent node (found 0)",
            ),
            (
                [_onboarding("a"), _onboarding("b")],
                [],
                "a flow without edges must contain exactly one agent node (found 2)",
            ),
            (
                [_onboarding("a"), _node("out", "output")],
                [_edge("e1", "a", "out")],
                "a flow with edges must contain exactly one input node (found 0)",
            ),
            (
                [_node("in-a", "input"), _node("in-b", "input"), _onboarding("a")],
                [_edge("e1", "in-a", "a")],
                "a flow with edges must contain exactly one input node (found 2)",
            ),
            (
                [_node("in", "input"), _onboarding("a"), _node("out", "output")],
                [_edge("e1", "a", "out")],
                "input node 'in' must have exactly one outgoing edge (found 0)",
            ),
            (
                [_node("in", "input"), _onboarding("a"), _onboarding("b")],
                [_edge("e1", "in", "a"), _edge("e2", "in", "b")],
                "input node 'in' must have exactly one outgoing edge (found 2)",
            ),
            (
                [_node("in", "input"), _node("out", "output")],
                [_edge("e1", "in", "out")],
                "the input node's edge must lead to an agent node (target 'out')",
            ),
            (
                [_node("in", "input"), _onboarding("a")],
                [_edge("e1", "in", "ghost")],
                "the input node's edge must lead to an agent node (target 'ghost')",
            ),
        ],
        ids=[
            "no-edges-no-agent",
            "no-edges-two-agents",
            "edges-without-input",
            "edges-with-two-inputs",
            "input-without-outgoing-edge",
            "input-with-two-outgoing-edges",
            "input-edge-to-output",
            "input-edge-to-missing-node",
        ],
    )
    def test_unresolvable_entry_raises(
        self,
        nodes: list[ChatbotFlowNode],
        edges: list[ChatbotFlowEdge],
        expected_problem: str,
    ) -> None:
        with pytest.raises(ChatbotFlowValidationError) as exc_info:
            resolve_entry_node(nodes=nodes, edges=edges)

        assert str(exc_info.value) == (
            f"[KAAPI] Chatbot flow is invalid: {expected_problem}."
        )


class TestValidateFlowForPublish:
    def test_returns_nodes_with_missing_config_version_pinned_to_latest(
        self, db: Session, config: Config
    ) -> None:
        add_test_config_version(db, config)
        input_node = _node("in", "input", label="Start")
        draft_node = _onboarding(config_id=config.id, config_version=None)

        pinned = validate_flow_for_publish(
            session=db,
            project_id=config.project_id,
            nodes=[input_node, draft_node],
            edges=[_edge("e1", "in", ONBOARDING_NODE_ID)],
        )

        assert pinned[0] == input_node
        assert pinned[1].data == {
            "label": "Beneficiary Info Extractor",
            "config_id": str(config.id),
            "prompt": "Collect the user's name and city.",
            "config_version": 2,
        }
        assert "config_version" not in draft_node.data

    def test_explicit_config_version_is_kept(self, db: Session, config: Config) -> None:
        add_test_config_version(db, config)

        [pinned] = validate_flow_for_publish(
            session=db,
            project_id=config.project_id,
            nodes=[_onboarding(config_id=config.id, config_version=1)],
            edges=[],
        )

        assert pinned.data["config_version"] == 1

    def test_problems_are_joined_into_one_error(
        self, db: Session, config: Config
    ) -> None:
        nodes = [
            _onboarding(config_id=config.id),
            _onboarding(config_id=config.id),
            _node("out-a", "output"),
            _node("out-b", "output"),
        ]

        with pytest.raises(ChatbotFlowValidationError) as exc_info:
            validate_flow_for_publish(
                session=db, project_id=config.project_id, nodes=nodes, edges=[]
            )

        assert str(exc_info.value) == (
            "[KAAPI] Chatbot flow is invalid: "
            f"duplicate node ids: {ONBOARDING_NODE_ID}; "
            "at most 1 output node allowed (found 2)."
        )

    def test_config_error_is_reported_per_node_without_nested_kaapi_tag(
        self, db: Session
    ) -> None:
        bad_thinking_config = create_test_anthropic_config(
            db, thinking={"type": "enabled", "budget_tokens": 999_999}
        )

        with pytest.raises(ChatbotFlowValidationError) as exc_info:
            validate_flow_for_publish(
                session=db,
                project_id=bad_thinking_config.project_id,
                nodes=[_onboarding(config_id=bad_thinking_config.id)],
                edges=[],
            )

        message = str(exc_info.value)
        assert f"node '{ONBOARDING_NODE_ID}': Config thinking.budget_tokens" in message
        assert message.count("[KAAPI]") == 1

    def test_config_of_another_project_is_rejected(
        self, db: Session, config: Config
    ) -> None:
        other_project_config = create_test_anthropic_config(db)

        with pytest.raises(ChatbotFlowValidationError) as exc_info:
            validate_flow_for_publish(
                session=db,
                project_id=config.project_id,
                nodes=[_onboarding(config_id=other_project_config.id)],
                edges=[],
            )

        assert f"node '{ONBOARDING_NODE_ID}': " in str(exc_info.value)

    def test_structurally_valid_flow_without_entry_raises(
        self, db: Session, config: Config
    ) -> None:
        nodes = [_onboarding("a", config.id), _onboarding("b", config.id)]

        with pytest.raises(ChatbotFlowValidationError) as exc_info:
            validate_flow_for_publish(
                session=db, project_id=config.project_id, nodes=nodes, edges=[]
            )

        assert "exactly one agent node (found 2)" in str(exc_info.value)
