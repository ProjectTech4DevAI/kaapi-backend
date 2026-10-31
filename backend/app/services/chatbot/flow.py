from typing import Any, NoReturn
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from app.core.util import now

from app.crud.chatbot import (
    create_chatbot_flow_version,
    get_chatbot,
    get_chatbot_flow_version,
    get_latest_chatbot_flow_version,
    publish_chatbot_flow_version,
    update_chatbot_flow_version_graph,
)
from app.models import (
    AGENT_NODE_TYPES,
    Chatbot,
    ChatbotFlowEdge,
    ChatbotFlowNode,
    ChatbotFlowNodeTypeEnum,
    ChatbotFlowUpdate,
    ChatbotFlowVersion,
    ChatbotFlowVersionPublic,
)
from app.services.chatbot.config import (
    build_chatbot_model_settings,
    get_anthropic_api_key,
    get_chatbot_config_version,
    prepare_rag_search_client,
)
from app.services.chatbot.exceptions import (
    ChatbotFlowConflictError,
    ChatbotFlowNotFoundError,
    ChatbotFlowValidationError,
    ChatbotInputError,
    ChatbotNotFoundError,
)

KAAPI_ERROR_TAG = "[KAAPI]"
FIRST_FLOW_VERSION = 1
MAX_INPUT_NODES = 1
MAX_OUTPUT_NODES = 1
# The runtime executes onboarding and rag agents; coordinator is stored but not publishable yet.
SUPPORTED_AGENT_NODE_TYPES: frozenset[ChatbotFlowNodeTypeEnum] = frozenset(
    {ChatbotFlowNodeTypeEnum.ONBOARDING, ChatbotFlowNodeTypeEnum.RAG}
)
_SUPPORTED_AGENT_NODE_TYPES_LABEL = ", ".join(
    sorted(node_type.value for node_type in SUPPORTED_AGENT_NODE_TYPES)
)
CHATBOT_NOT_FOUND_MESSAGE = "Chatbot not found"
NO_DRAFT_TO_PUBLISH_MESSAGE = (
    "[KAAPI] Chatbot has no draft flow to publish; save the flow first."
)


def dump_flow_nodes(nodes: list[ChatbotFlowNode]) -> list[dict[str, Any]]:
    """Serialize with React Flow (camelCase) keys so the stored JSON matches what the UI sent."""
    dumped: list[dict[str, Any]] = []
    for node in nodes:
        dumped.append(node.model_dump(mode="json", by_alias=True))
    return dumped


def dump_flow_edges(edges: list[ChatbotFlowEdge]) -> list[dict[str, Any]]:
    dumped: list[dict[str, Any]] = []
    for edge in edges:
        dumped.append(edge.model_dump(mode="json", by_alias=True))
    return dumped


def parse_flow_nodes(raw_nodes: list[dict[str, Any]]) -> list[ChatbotFlowNode]:
    nodes: list[ChatbotFlowNode] = []
    for raw_node in raw_nodes:
        nodes.append(ChatbotFlowNode.model_validate(raw_node))
    return nodes


def parse_flow_edges(raw_edges: list[dict[str, Any]]) -> list[ChatbotFlowEdge]:
    edges: list[ChatbotFlowEdge] = []
    for raw_edge in raw_edges:
        edges.append(ChatbotFlowEdge.model_validate(raw_edge))
    return edges


def find_flow_node(
    nodes: list[ChatbotFlowNode], node_id: str
) -> ChatbotFlowNode | None:
    for node in nodes:
        if node.id == node_id:
            return node
    return None


def build_chatbot_flow_version_public(
    flow_version: ChatbotFlowVersion,
) -> ChatbotFlowVersionPublic:
    return ChatbotFlowVersionPublic(
        id=flow_version.id,
        chatbot_id=flow_version.chatbot_id,
        version=flow_version.version,
        is_live=flow_version.is_live,
        nodes=parse_flow_nodes(flow_version.nodes),
        edges=parse_flow_edges(flow_version.edges),
        inserted_at=flow_version.inserted_at,
        updated_at=flow_version.updated_at,
    )


def _raise_flow_validation_error(function_name: str, problems: list[str]) -> NoReturn:
    error_message = f"{KAAPI_ERROR_TAG} Chatbot flow is invalid: {'; '.join(problems)}."
    raise ChatbotFlowValidationError(error_message)


def resolve_entry_node(
    *, nodes: list[ChatbotFlowNode], edges: list[ChatbotFlowEdge]
) -> ChatbotFlowNode:
    """The agent node a new session starts on.

    Without edges the flow must be a single agent node; otherwise the entry is
    the target of the one edge leaving the input node.
    """
    if not edges:
        agent_nodes = [node for node in nodes if node.type in AGENT_NODE_TYPES]
        if len(agent_nodes) != 1:
            _raise_flow_validation_error(
                "resolve_entry_node",
                [
                    "a flow without edges must contain exactly one agent node "
                    f"(found {len(agent_nodes)})"
                ],
            )
        return agent_nodes[0]

    input_nodes = [node for node in nodes if node.type == ChatbotFlowNodeTypeEnum.INPUT]
    if len(input_nodes) != 1:
        _raise_flow_validation_error(
            "resolve_entry_node",
            [
                "a flow with edges must contain exactly one input node "
                f"(found {len(input_nodes)})"
            ],
        )
    input_node_id = input_nodes[0].id

    entry_edges = [edge for edge in edges if edge.source == input_node_id]
    if len(entry_edges) != 1:
        _raise_flow_validation_error(
            "resolve_entry_node",
            [
                f"input node '{input_node_id}' must have exactly one outgoing edge "
                f"(found {len(entry_edges)})"
            ],
        )

    entry_node = find_flow_node(nodes, entry_edges[0].target)
    if entry_node is None or entry_node.type not in AGENT_NODE_TYPES:
        _raise_flow_validation_error(
            "resolve_entry_node",
            [
                f"the input node's edge must lead to an agent node "
                f"(target '{entry_edges[0].target}')"
            ],
        )
    return entry_node


def _list_structural_problems(
    *, nodes: list[ChatbotFlowNode], edges: list[ChatbotFlowEdge]
) -> list[str]:
    problems: list[str] = []

    seen_node_ids: set[str] = set()
    duplicate_node_ids: set[str] = set()
    input_count = 0
    output_count = 0
    for node in nodes:
        if node.id in seen_node_ids:
            duplicate_node_ids.add(node.id)
        seen_node_ids.add(node.id)
        if node.type == ChatbotFlowNodeTypeEnum.INPUT:
            input_count += 1
        if node.type == ChatbotFlowNodeTypeEnum.OUTPUT:
            output_count += 1

    if duplicate_node_ids:
        problems.append(f"duplicate node ids: {', '.join(sorted(duplicate_node_ids))}")
    if input_count > MAX_INPUT_NODES:
        problems.append(
            f"at most {MAX_INPUT_NODES} input node allowed (found {input_count})"
        )
    if output_count > MAX_OUTPUT_NODES:
        problems.append(
            f"at most {MAX_OUTPUT_NODES} output node allowed (found {output_count})"
        )

    for edge in edges:
        if edge.source not in seen_node_ids:
            problems.append(
                f"edge '{edge.id}' source '{edge.source}' is not a node in the flow"
            )
        if edge.target not in seen_node_ids:
            problems.append(
                f"edge '{edge.id}' target '{edge.target}' is not a node in the flow"
            )
    return problems


def _pin_agent_node_config(
    *, session: Session, organization_id: int, project_id: int, node: ChatbotFlowNode
) -> tuple[ChatbotFlowNode, list[str]]:
    """Check the agent node is runnable; returns the node with its config_version pinned, plus any problems."""
    if node.type not in SUPPORTED_AGENT_NODE_TYPES:
        return node, [
            f"node '{node.id}' has type '{node.type.value}', which is not supported yet "
            f"(supported agent types: {_SUPPORTED_AGENT_NODE_TYPES_LABEL})"
        ]

    node_data = node.agent_data()
    problems: list[str] = []
    if node_data.prompt is None or not node_data.prompt.strip():
        problems.append(f"node '{node.id}' is missing data.prompt")
    if node_data.config_id is None:
        problems.append(f"node '{node.id}' is missing data.config_id")
        return node, problems

    try:
        version_row = get_chatbot_config_version(
            session=session,
            project_id=project_id,
            config_id=node_data.config_id,
            config_version=node_data.config_version,
        )
    except HTTPException as e:
        problems.append(f"node '{node.id}': {e.detail}")
        return node, problems

    try:
        model_settings = build_chatbot_model_settings(
            version_row, max_tokens=None, history_token_limit=None
        )
    except ChatbotInputError as e:
        reason = str(e).removeprefix(KAAPI_ERROR_TAG).strip()
        problems.append(f"node '{node.id}': {reason}")
        return node, problems

    if node.type == ChatbotFlowNodeTypeEnum.RAG:
        try:
            prepare_rag_search_client(
                session=session,
                organization_id=organization_id,
                project_id=project_id,
                model_settings=model_settings,
            )
        except ChatbotInputError as e:
            reason = str(e).removeprefix(KAAPI_ERROR_TAG).strip()
            problems.append(f"node '{node.id}': {reason}")
            return node, problems

    # Pinned at publish so later config edits never change a published flow.
    pinned_data = {**node.data, "config_version": version_row.version}
    return node.model_copy(update={"data": pinned_data}), problems


def validate_flow_for_publish(
    *,
    session: Session,
    organization_id: int,
    project_id: int,
    nodes: list[ChatbotFlowNode],
    edges: list[ChatbotFlowEdge],
) -> list[ChatbotFlowNode]:
    """Strict publish-time checks; returns the nodes with every agent config_version pinned."""
    problems = _list_structural_problems(nodes=nodes, edges=edges)

    # Agent nodes call Anthropic with the project's own key, so a live flow needs one.
    if any(node.type in AGENT_NODE_TYPES for node in nodes):
        try:
            get_anthropic_api_key(
                session=session, organization_id=organization_id, project_id=project_id
            )
        except ChatbotInputError as e:
            problems.append(str(e).removeprefix(KAAPI_ERROR_TAG).strip())

    pinned_nodes: list[ChatbotFlowNode] = []
    for node in nodes:
        if node.type not in AGENT_NODE_TYPES:
            pinned_nodes.append(node)
            continue
        pinned_node, node_problems = _pin_agent_node_config(
            session=session,
            organization_id=organization_id,
            project_id=project_id,
            node=node,
        )
        pinned_nodes.append(pinned_node)
        problems.extend(node_problems)

    if problems:
        _raise_flow_validation_error("validate_flow_for_publish", problems)

    resolve_entry_node(nodes=pinned_nodes, edges=edges)
    return pinned_nodes


def get_chatbot_or_raise(
    *,
    session: Session,
    chatbot_id: UUID,
    organization_id: int,
    project_id: int,
    for_update: bool = False,
) -> Chatbot:
    """The non-deleted chatbot in the project, else ChatbotNotFoundError; `for_update` row-locks it."""
    chatbot = get_chatbot(
        session,
        chatbot_id=chatbot_id,
        organization_id=organization_id,
        project_id=project_id,
        for_update=for_update,
    )
    if chatbot is None:
        raise ChatbotNotFoundError(CHATBOT_NOT_FOUND_MESSAGE)
    return chatbot


def _mark_chatbot_edited(session: Session, chatbot: Chatbot) -> None:
    chatbot.updated_at = now()
    session.add(chatbot)


def get_chatbot_flow(
    *,
    session: Session,
    chatbot_id: UUID,
    organization_id: int,
    project_id: int,
    version: int | None,
) -> ChatbotFlowVersion:
    """The requested version, else the draft if one exists, else the live version."""
    get_chatbot_or_raise(
        session=session,
        chatbot_id=chatbot_id,
        organization_id=organization_id,
        project_id=project_id,
        for_update=False,
    )
    if version is not None:
        flow_version = get_chatbot_flow_version(
            session, chatbot_id=chatbot_id, version=version
        )
        if flow_version is None:
            raise ChatbotFlowNotFoundError(
                f"Flow version {version} not found for this chatbot"
            )
        return flow_version

    # The highest version is always the draft when one exists, otherwise the live one.
    flow_version = get_latest_chatbot_flow_version(session, chatbot_id=chatbot_id)
    if flow_version is None:
        raise ChatbotFlowNotFoundError("Chatbot has no flow yet; save a flow first")
    return flow_version


def save_chatbot_flow_draft(
    *,
    session: Session,
    chatbot_id: UUID,
    organization_id: int,
    project_id: int,
    flow: ChatbotFlowUpdate,
) -> ChatbotFlowVersion:
    """Save the flow: the first version goes live at once; later saves overwrite the
    draft, or start a new draft when the highest version is live (live versions are immutable).
    """
    # The row lock serializes draft saves and publishes of the same chatbot.
    chatbot = get_chatbot_or_raise(
        session=session,
        chatbot_id=chatbot_id,
        organization_id=organization_id,
        project_id=project_id,
        for_update=True,
    )
    nodes = dump_flow_nodes(flow.nodes)
    edges = dump_flow_edges(flow.edges)

    latest = get_latest_chatbot_flow_version(session, chatbot_id=chatbot_id)
    if latest is None:
        # A new chatbot has nothing to run until its first flow exists, so that save
        # goes live directly; it is validated like a publish so live is always runnable.
        pinned_nodes = validate_flow_for_publish(
            session=session,
            organization_id=organization_id,
            project_id=project_id,
            nodes=flow.nodes,
            edges=flow.edges,
        )
        _mark_chatbot_edited(session, chatbot)
        return create_chatbot_flow_version(
            session,
            chatbot_id=chatbot_id,
            version=FIRST_FLOW_VERSION,
            nodes=dump_flow_nodes(pinned_nodes),
            edges=edges,
            is_live=True,
        )

    _mark_chatbot_edited(session, chatbot)
    if not latest.is_live:
        return update_chatbot_flow_version_graph(
            session, flow_version=latest, nodes=nodes, edges=edges
        )

    return create_chatbot_flow_version(
        session,
        chatbot_id=chatbot_id,
        version=latest.version + 1,
        nodes=nodes,
        edges=edges,
    )


def publish_chatbot_flow(
    *,
    session: Session,
    chatbot_id: UUID,
    organization_id: int,
    project_id: int,
) -> ChatbotFlowVersion:
    chatbot = get_chatbot_or_raise(
        session=session,
        chatbot_id=chatbot_id,
        organization_id=organization_id,
        project_id=project_id,
        for_update=True,
    )
    draft = get_latest_chatbot_flow_version(session, chatbot_id=chatbot_id)
    if draft is None or draft.is_live:
        raise ChatbotFlowConflictError(NO_DRAFT_TO_PUBLISH_MESSAGE)

    pinned_nodes = validate_flow_for_publish(
        session=session,
        organization_id=organization_id,
        project_id=project_id,
        nodes=parse_flow_nodes(draft.nodes),
        edges=parse_flow_edges(draft.edges),
    )
    _mark_chatbot_edited(session, chatbot)
    return publish_chatbot_flow_version(
        session, flow_version=draft, nodes=dump_flow_nodes(pinned_nodes)
    )
