import asyncio
import logging
from typing import NamedTuple
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from app.core.db import engine
from app.crud.chatbot import (
    create_chatbot_session,
    create_chatbot_session_message,
    get_chatbot_flow_version_by_id,
    get_chatbot_session,
    get_latest_chatbot_session,
    get_live_chatbot_flow_version,
    list_chatbot_session_messages,
    update_chatbot_session_progress,
)
from app.models import (
    AGENT_NODE_TYPES,
    ChatbotFlowNodeTypeEnum,
    ChatbotFlowVersion,
    ChatbotMessageRoleEnum,
    ChatbotSession,
    ChatbotSessionMessage,
    ChatbotSessionStatusEnum,
    ChatbotSessionSummary,
)
from app.services.chatbot.agents.coordinator.graph import run_coordinator_turn
from app.services.chatbot.agents.coordinator.prompts import (
    FALLBACK_CLARIFYING_QUESTION,
)
from app.services.chatbot.agents.onboarding.graph import run_chatbot_turn
from app.services.chatbot.config import resolve_chatbot_model_settings
from app.services.chatbot.exceptions import (
    ChatbotFlowConflictError,
    ChatbotInputError,
    ChatbotProviderError,
)
from app.services.chatbot.flow import (
    EDGE_DESCRIPTION_KEY,
    EDGE_INTENT_KEY,
    SUPPORTED_AGENT_NODE_TYPES,
    find_flow_node,
    get_chatbot_or_raise,
    parse_flow_edges,
    parse_flow_nodes,
    resolve_entry_node,
)
from app.services.chatbot.types import (
    ChatbotMessageDict,
    ChatbotTurnResult,
    CoordinatorRoute,
    TurnContext,
    TurnNode,
)
from app.services.chatbot.utils import ChatbotModelSettings
from app.utils import APIResponse, get_webhook_secret, send_callback

logger = logging.getLogger(__name__)

SUPERSEDED_SESSION_ERROR = (
    "[KAAPI] Chatbot session was superseded by a newer session; start a new session."
)
NO_LIVE_FLOW_MESSAGE = "[KAAPI] Chatbot has no published flow; publish the flow first."


_RUNNABLE_NODE_TYPES_LABEL = ", ".join(
    sorted(f"'{node_type.value}'" for node_type in SUPPORTED_AGENT_NODE_TYPES)
)
ACTIVE_NODE_LABEL = "Active node"
ROUTE_TARGET_LABEL = "Route target"

_RouteTargets = dict[str, tuple[TurnNode, ChatbotModelSettings]]


class _TurnOutcome(NamedTuple):
    result: ChatbotTurnResult
    reply_node_id: str
    # None leaves the session's active node unchanged.
    next_active_node_id: str | None


def start_chatbot_session(
    *,
    session: Session,
    chatbot_id: UUID,
    organization_id: int,
    project_id: int,
    contact_ref: str,
) -> tuple[ChatbotSession, ChatbotFlowVersion]:
    """Create a session pinned to the live flow version and its entry node."""
    get_chatbot_or_raise(
        session=session,
        chatbot_id=chatbot_id,
        organization_id=organization_id,
        project_id=project_id,
    )

    live_version = get_live_chatbot_flow_version(session, chatbot_id=chatbot_id)
    if live_version is None:
        raise ChatbotFlowConflictError(NO_LIVE_FLOW_MESSAGE)

    entry_node = resolve_entry_node(
        nodes=parse_flow_nodes(live_version.nodes),
        edges=parse_flow_edges(live_version.edges),
    )
    chatbot_session = create_chatbot_session(
        session,
        chatbot_id=chatbot_id,
        flow_version_id=live_version.id,
        contact_ref=contact_ref,
        active_node_id=entry_node.id,
    )
    return chatbot_session, live_version


def get_latest_chatbot_session_summary(
    *,
    session: Session,
    organization_id: int,
    project_id: int,
    contact_ref: str,
    chatbot_id: UUID | None,
) -> ChatbotSessionSummary | None:
    row = get_latest_chatbot_session(
        session,
        organization_id=organization_id,
        project_id=project_id,
        contact_ref=contact_ref,
        chatbot_id=chatbot_id,
    )
    if row is None:
        return None
    chatbot_session, chatbot, flow_version = row

    prompt: str | None = None
    active_node = find_flow_node(
        parse_flow_nodes(flow_version.nodes), chatbot_session.active_node_id
    )
    if active_node is not None and active_node.type in AGENT_NODE_TYPES:
        prompt = active_node.agent_data().prompt

    return ChatbotSessionSummary(
        session_id=chatbot_session.id,
        chatbot_id=chatbot.id,
        chatbot_name=chatbot.name,
        flow_version=flow_version.version,
        active_node_id=chatbot_session.active_node_id,
        prompt=prompt,
        contact_ref=chatbot_session.contact_ref,
        status=chatbot_session.status,
        completion_status=chatbot_session.completion_status,
        collected_fields=chatbot_session.collected_fields,
        inserted_at=chatbot_session.inserted_at,
        updated_at=chatbot_session.updated_at,
    )


def _node_not_in_flow_message(label: str, node_id: str) -> str:
    return (
        f"[KAAPI] {label} '{node_id}' is not in the session's flow version; "
        "start a new session."
    )


def _resolve_turn_node(
    *,
    flow_version: ChatbotFlowVersion,
    active_node_id: str,
    label: str = ACTIVE_NODE_LABEL,
) -> TurnNode:
    """The runnable agent node `active_node_id` names; ChatbotInputError when it can't run.

    `label` names the node's role in error messages (the active node or a coordinator's route target).
    """
    active_node = find_flow_node(parse_flow_nodes(flow_version.nodes), active_node_id)
    if active_node is None:
        raise ChatbotInputError(_node_not_in_flow_message(label, active_node_id))

    if active_node.type not in SUPPORTED_AGENT_NODE_TYPES:
        error_message = (
            f"[KAAPI] {label} '{active_node_id}' has type '{active_node.type.value}', "
            f"which the chatbot runtime does not support yet; only "
            f"{_RUNNABLE_NODE_TYPES_LABEL} nodes can run."
        )
        raise ChatbotInputError(error_message)

    node_data = active_node.agent_data()
    if node_data.prompt is None or node_data.config_id is None:
        error_message = (
            f"[KAAPI] {label} '{active_node_id}' is missing data.prompt or "
            "data.config_id; republish the flow and start a new session."
        )
        raise ChatbotInputError(error_message)

    return TurnNode(
        node_id=active_node.id,
        node_type=active_node.type,
        prompt=node_data.prompt,
        config_id=node_data.config_id,
        config_version=node_data.config_version,
    )


def _resolve_coordinator_routes(
    *,
    session: Session,
    flow_version: ChatbotFlowVersion,
    coordinator_node_id: str,
    organization_id: int,
    project_id: int,
) -> tuple[list[CoordinatorRoute], _RouteTargets]:
    """Every outgoing route of the coordinator, plus each target's node and model settings.

    Intents and target types are enforced at publish, the only way a flow goes live.
    """
    routes: list[CoordinatorRoute] = []
    targets: _RouteTargets = {}
    for edge in parse_flow_edges(flow_version.edges):
        if edge.source != coordinator_node_id:
            continue

        target_node = _resolve_turn_node(
            flow_version=flow_version,
            active_node_id=edge.target,
            label=ROUTE_TARGET_LABEL,
        )
        description = edge.data.get(EDGE_DESCRIPTION_KEY)
        targets[target_node.node_id] = (
            target_node,
            resolve_chatbot_model_settings(
                session=session,
                organization_id=organization_id,
                project_id=project_id,
                config_id=target_node.config_id,
                config_version=target_node.config_version,
                max_tokens=None,
                history_token_limit=None,
            ),
        )
        routes.append(
            CoordinatorRoute(
                intent=edge.data[EDGE_INTENT_KEY].strip(),
                description=description if isinstance(description, str) else None,
                target_node_id=target_node.node_id,
            )
        )

    if not routes:
        error_message = (
            f"[KAAPI] Coordinator '{coordinator_node_id}' has no outgoing routes; "
            "republish the flow and start a new session."
        )
        raise ChatbotInputError(error_message)
    return routes, targets


def _find_return_coordinator(
    *, flow_version: ChatbotFlowVersion, node_id: str
) -> str | None:
    """The coordinator that routes to `node_id`, or None when the node isn't behind one."""
    nodes = parse_flow_nodes(flow_version.nodes)
    for edge in parse_flow_edges(flow_version.edges):
        if edge.target != node_id:
            continue
        source_node = find_flow_node(nodes, edge.source)
        if (
            source_node is not None
            and source_node.type == ChatbotFlowNodeTypeEnum.COORDINATOR
        ):
            return source_node.id
    return None


def _split_pending_turn(
    stored_messages: list[ChatbotSessionMessage],
) -> tuple[list[ChatbotMessageDict], str | None]:
    """Split the stored transcript into (prior history, pending user message)."""
    history: list[ChatbotMessageDict] = []
    for stored in stored_messages:
        # The system prompt lives on the flow node and is passed separately.
        if stored.role == ChatbotMessageRoleEnum.SYSTEM:
            continue
        history.append({"role": stored.role.value, "content": stored.content})

    if not history:
        return history, None
    if history[-1]["role"] != ChatbotMessageRoleEnum.USER:
        # No pending user turn; the agent runner rejects this with ChatbotInputError.
        return history, None
    pending = history.pop()
    return history, pending["content"]


def _resolve_turn_context(
    *,
    session: Session,
    chatbot_session: ChatbotSession,
    stored_messages: list[ChatbotSessionMessage],
    organization_id: int,
    project_id: int,
) -> TurnContext:
    """The active node and its settings, plus (coordinator) its routes or (agent) the coordinator it returns to.

    Every route target's settings are resolved here too, so no DB access happens once the LLM calls start.
    """
    flow_version = get_chatbot_flow_version_by_id(
        session, flow_version_id=chatbot_session.flow_version_id
    )
    if flow_version is None:
        raise ChatbotInputError(
            _node_not_in_flow_message(ACTIVE_NODE_LABEL, chatbot_session.active_node_id)
        )
    turn_node = _resolve_turn_node(
        flow_version=flow_version, active_node_id=chatbot_session.active_node_id
    )

    routes: list[CoordinatorRoute] = []
    route_targets: _RouteTargets = {}
    return_node_id: str | None = None
    try:
        model_settings = resolve_chatbot_model_settings(
            session=session,
            organization_id=organization_id,
            project_id=project_id,
            config_id=turn_node.config_id,
            config_version=turn_node.config_version,
            max_tokens=None,
            history_token_limit=None,
        )
        if turn_node.node_type == ChatbotFlowNodeTypeEnum.COORDINATOR:
            routes, route_targets = _resolve_coordinator_routes(
                session=session,
                flow_version=flow_version,
                coordinator_node_id=turn_node.node_id,
                organization_id=organization_id,
                project_id=project_id,
            )
        else:
            return_node_id = _find_return_coordinator(
                flow_version=flow_version, node_id=turn_node.node_id
            )
    except HTTPException as e:
        # Config lookups raise HTTP-shaped 404s; the worker reports them as a failure webhook.
        raise ChatbotInputError(str(e.detail)) from e
    return TurnContext(
        chatbot_id=chatbot_session.chatbot_id,
        session_status=chatbot_session.status.value,
        collected_fields=dict(chatbot_session.collected_fields),
        stored_messages=stored_messages,
        turn_node=turn_node,
        model_settings=model_settings,
        routes=routes,
        route_targets=route_targets,
        return_node_id=return_node_id,
    )


def _load_turn_context(
    *, session_id: UUID, organization_id: int, project_id: int
) -> TurnContext | None:
    """Everything the turn needs, read in one short DB session; None when there is nothing to answer."""
    # The DB session is closed before the LLM call so a slow provider doesn't
    # pin a pooled connection for the whole turn.
    with Session(engine) as session:
        chatbot_session = get_chatbot_session(
            session,
            session_id=session_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        if chatbot_session is None:
            return None
        # The turn may have been queued before a newer session (or a chatbot delete) deactivated this one.
        if chatbot_session.status == ChatbotSessionStatusEnum.INACTIVE:
            raise ChatbotInputError(SUPERSEDED_SESSION_ERROR)

        stored_messages = list_chatbot_session_messages(session, session_id=session_id)
        if (
            stored_messages
            and stored_messages[-1].role == ChatbotMessageRoleEnum.ASSISTANT
        ):
            return None

        return _resolve_turn_context(
            session=session,
            chatbot_session=chatbot_session,
            stored_messages=stored_messages,
            organization_id=organization_id,
            project_id=project_id,
        )


async def _run_coordinator_turn_and_handoff(
    *,
    context: TurnContext,
    history: list[ChatbotMessageDict],
    user_message: str | None,
) -> _TurnOutcome:
    """Route, then let the chosen agent reply in the same turn (silent handoff)."""
    coordinator_node = context.turn_node
    decision = await run_coordinator_turn(
        system_prompt=coordinator_node.prompt,
        history=history,
        user_message=user_message,
        routes=context.routes,
        model_settings=context.model_settings,
    )
    if decision.target_node_id is None:
        question = decision.clarifying_question or FALLBACK_CLARIFYING_QUESTION
        return _TurnOutcome(
            result=ChatbotTurnResult(
                reply=question,
                collected_fields=context.collected_fields,
                is_complete=False,
            ),
            reply_node_id=coordinator_node.node_id,
            next_active_node_id=None,
        )

    target_node, target_settings = context.route_targets[decision.target_node_id]
    turn_result = await run_chatbot_turn(
        system_prompt=target_node.prompt,
        history=history,
        user_message=user_message,
        model_settings=target_settings,
        collected_fields=context.collected_fields,
    )
    # A finished sub-flow hands the session back so the next message is routed afresh.
    return _TurnOutcome(
        result=turn_result,
        reply_node_id=target_node.node_id,
        next_active_node_id=(
            coordinator_node.node_id if turn_result.is_complete else target_node.node_id
        ),
    )


def _run_turn(context: TurnContext) -> _TurnOutcome:
    history, user_message = _split_pending_turn(context.stored_messages)
    if context.turn_node.node_type == ChatbotFlowNodeTypeEnum.COORDINATOR:
        return asyncio.run(
            _run_coordinator_turn_and_handoff(
                context=context, history=history, user_message=user_message
            )
        )

    turn_result = asyncio.run(
        run_chatbot_turn(
            system_prompt=context.turn_node.prompt,
            history=history,
            user_message=user_message,
            model_settings=context.model_settings,
            collected_fields=context.collected_fields,
        )
    )
    return _TurnOutcome(
        result=turn_result,
        reply_node_id=context.turn_node.node_id,
        next_active_node_id=context.return_node_id if turn_result.is_complete else None,
    )


def _persist_turn(*, session_id: UUID, outcome: _TurnOutcome) -> None:
    turn_result = outcome.result
    with Session(engine) as session:
        create_chatbot_session_message(
            session,
            session_id=session_id,
            role=ChatbotMessageRoleEnum.ASSISTANT,
            content=turn_result.reply,
            node_id=outcome.reply_node_id,
        )
        update_chatbot_session_progress(
            session,
            session_id=session_id,
            collected_fields=turn_result.collected_fields,
            completion_status=turn_result.completion_status,
            active_node_id=outcome.next_active_node_id,
        )


def _send_failure_webhook(
    *,
    session_id: UUID,
    webhook_url: str,
    webhook_secret: str | None,
    error: str,
) -> None:
    envelope = APIResponse.failure_response(
        error=error, data={"session_id": str(session_id)}
    )
    send_callback(webhook_url, envelope.model_dump(exclude_none=True), webhook_secret)


def process_chatbot_session_message(
    *,
    session_id: UUID,
    organization_id: int,
    project_id: int,
    webhook_url: str,
) -> None:
    """Run the pending chatbot turn (load -> run -> persist -> notify) and POST the reply to `webhook_url`."""
    webhook_secret = get_webhook_secret(project_id, organization_id)

    try:
        context = _load_turn_context(
            session_id=session_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        if context is None:
            return
        outcome = _run_turn(context)
    except (ChatbotInputError, ChatbotProviderError) as e:
        _send_failure_webhook(
            session_id=session_id,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            error=str(e),
        )
        return

    _persist_turn(session_id=session_id, outcome=outcome)
    logger.info(
        f"[process_chatbot_session_message] Turn stored | session_id: {session_id}, "
        f"reply_node_id: {outcome.reply_node_id}, "
        f"next_active_node_id: {outcome.next_active_node_id}"
    )
    turn_result = outcome.result
    envelope = APIResponse.success_response(
        data={
            "session_id": str(session_id),
            "chatbot_id": str(context.chatbot_id),
            "node_id": outcome.reply_node_id,
            "reply": turn_result.reply,
            "status": context.session_status,
            "completion_status": turn_result.completion_status.value,
            "collected_fields": turn_result.collected_fields,
        }
    )
    send_callback(webhook_url, envelope.model_dump(exclude_none=True), webhook_secret)
