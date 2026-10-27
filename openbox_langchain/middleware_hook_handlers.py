"""``before_agent``/``after_agent`` hook implementations for OpenBoxLangChainMiddleware.

``wrap_model_call``/``awrap_model_call`` live in ``middleware_model_call.py``;
tool governance lives in ``middleware_tool_hook.py`` (split to stay under 200
lines per file).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from openbox_langchain.lifecycle_events import build_activity_started
from openbox_langchain.middleware_hooks import (
    build_signal_received_envelope,
    build_workflow_completed_envelope,
    build_workflow_started_envelope,
    enforce_start_verdict_async,
    enforce_start_verdict_sync,
    evaluate_lifecycle_async,
    evaluate_lifecycle_sync,
    extract_last_user_message,
)
from openbox_langchain.middleware_llm_logic import llm_activity_context

if TYPE_CHECKING:
    from openbox_langchain.middleware import OpenBoxLangChainMiddleware
    from openbox_langchain.middleware_turn_state import MiddlewareTurnState

_logger = logging.getLogger(__name__)

__all__ = [
    "handle_after_agent",
    "handle_after_agent_sync",
    "handle_before_agent",
    "handle_before_agent_sync",
]


def _state_messages(state: Any) -> list[Any]:
    return state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])


def _last_message_content(messages: list[Any]) -> Any:
    if not messages:
        return None
    last_msg = messages[-1]
    if hasattr(last_msg, "content"):
        return last_msg.content
    if isinstance(last_msg, dict):
        return last_msg.get("content")
    return None


def _build_pre_screen_envelope(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, user_prompt: str
) -> Any:
    return build_activity_started(
        workflow_id=turn.workflow_id, run_id=turn.run_id, workflow_type=mw._workflow_type,
        activity_id=f"{turn.run_id}-pre", activity_type="llm_call",
        task_queue=mw._options.task_queue, activity_input=[{"prompt": user_prompt}],
        session_id=mw._options.session_id, agent_name=mw._options.agent_name,
    )


# ═══════════════════════════════════════════════════════════════════
# Hook: before_agent → WorkflowStarted + SignalReceived + pre-screen
# ═══════════════════════════════════════════════════════════════════


async def handle_before_agent(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, state: Any, runtime: Any,
) -> MiddlewareTurnState:
    """Async before_agent: SignalReceived + WorkflowStarted + pre-screen guardrails.

    Returns the (possibly pre-screen-carrying) ``turn`` — ``MiddlewareTurnState``
    is frozen, so the caller (``middleware.py``) returns THIS value as the
    ``ob_turn`` state update rather than relying on in-place mutation.
    """
    user_prompt = extract_last_user_message(_state_messages(state))

    if user_prompt:
        await evaluate_lifecycle_async(mw, build_signal_received_envelope(mw, turn, user_prompt))
    if mw._options.send_chain_start_event:
        await evaluate_lifecycle_async(mw, build_workflow_started_envelope(mw, turn, state))

    if mw._options.send_llm_start_event and user_prompt and user_prompt.strip():
        pre_screen = _build_pre_screen_envelope(mw, turn, user_prompt)
        response = await evaluate_lifecycle_async(mw, pre_screen)
        try:
            await enforce_start_verdict_async(
                mw, response, llm_activity_context(mw, turn, pre_screen.activity_id)
            )
        except Exception as enforcement_error:
            if mw._options.send_chain_end_event:
                failed = build_workflow_completed_envelope(mw, turn, error=str(enforcement_error))
                # Best effort: a HALT has already closed the session on Core,
                # which rejects this event — that must not replace the verdict.
                try:
                    await evaluate_lifecycle_async(mw, failed)
                except Exception:
                    _logger.debug("closing event after pre-screen stop failed", exc_info=True)
            raise
        turn = turn.with_pre_screen(response)
    return turn


def handle_before_agent_sync(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, state: Any, runtime: Any,
) -> MiddlewareTurnState:
    """Sync before_agent: same sequence as :func:`handle_before_agent`, ``gate.evaluate``.

    Returns the (possibly pre-screen-carrying) ``turn`` — see
    :func:`handle_before_agent` for why this is a return value, not a mutation.
    """
    user_prompt = extract_last_user_message(_state_messages(state))

    if user_prompt:
        evaluate_lifecycle_sync(mw, build_signal_received_envelope(mw, turn, user_prompt))
    if mw._options.send_chain_start_event:
        evaluate_lifecycle_sync(mw, build_workflow_started_envelope(mw, turn, state))

    if mw._options.send_llm_start_event and user_prompt and user_prompt.strip():
        pre_screen = _build_pre_screen_envelope(mw, turn, user_prompt)
        response = evaluate_lifecycle_sync(mw, pre_screen)
        try:
            enforce_start_verdict_sync(mw, response)
        except Exception as enforcement_error:
            if mw._options.send_chain_end_event:
                failed = build_workflow_completed_envelope(mw, turn, error=str(enforcement_error))
                # Best effort: a HALT has already closed the session on Core,
                # which rejects this event — that must not replace the verdict.
                try:
                    evaluate_lifecycle_sync(mw, failed)
                except Exception:
                    _logger.debug("closing event after pre-screen stop failed", exc_info=True)
            raise
        turn = turn.with_pre_screen(response)
    return turn


# ═══════════════════════════════════════════════════════════════════
# Hook: after_agent → WorkflowCompleted
# ═══════════════════════════════════════════════════════════════════


async def handle_after_agent(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, state: Any, runtime: Any,
) -> dict[str, Any] | None:
    """Async after_agent: WorkflowCompleted."""
    if mw._options.send_chain_end_event:
        last_content = _last_message_content(_state_messages(state))
        envelope = build_workflow_completed_envelope(
            mw, turn, workflow_output={"result": last_content}
        )
        await evaluate_lifecycle_async(mw, envelope)
    return None


def handle_after_agent_sync(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, state: Any, runtime: Any,
) -> dict[str, Any] | None:
    """Sync after_agent: WorkflowCompleted."""
    if mw._options.send_chain_end_event:
        last_content = _last_message_content(_state_messages(state))
        envelope = build_workflow_completed_envelope(
            mw, turn, workflow_output={"result": last_content}
        )
        evaluate_lifecycle_sync(mw, envelope)
    return None
