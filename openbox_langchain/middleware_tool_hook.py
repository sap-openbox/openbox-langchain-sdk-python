"""Tool governance hook for OpenBoxLangChainMiddleware.

Separated from middleware_hook_handlers.py to stay under 200 lines per file.
Wraps tool execution in ``activity_scope(...)`` (bound to the middleware's
own ``ContextStore``) so Layer-2 HTTP/DB/file instrumentation resolves the
right ``ActivityContext`` for the duration of the tool body — replacing the
removed ``WorkflowSpanProcessor`` wiring.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from openbox_core.context import activity_scope
from openbox_core.contracts.context import ActivityContext
from openbox_core.serialization import to_json_safe

from openbox_langchain.lifecycle_events import build_activity_completed, build_activity_started
from openbox_langchain.middleware_hooks import (
    enforce_start_verdict_async,
    enforce_start_verdict_sync,
    evaluate_lifecycle_async,
    evaluate_lifecycle_sync,
)

if TYPE_CHECKING:
    from openbox_core.contracts.events import EventEnvelope

    from openbox_langchain.middleware import OpenBoxLangChainMiddleware
    from openbox_langchain.middleware_turn_state import MiddlewareTurnState

__all__ = ["handle_wrap_tool_call", "handle_wrap_tool_call_sync"]


def _tool_call_context(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, activity_id: str, tool_name: str,
) -> ActivityContext:
    return ActivityContext(
        workflow_id=turn.workflow_id, run_id=turn.run_id, workflow_type=mw._workflow_type,
        task_queue=mw._options.task_queue, activity_id=activity_id, activity_type=tool_name,
        agent_name=mw._options.agent_name, session_id=mw._options.session_id,
    )


def _build_tool_started(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, activity_id: str,
    tool_name: str, tool_args: Any,
) -> EventEnvelope:
    return build_activity_started(
        workflow_id=turn.workflow_id, run_id=turn.run_id, workflow_type=mw._workflow_type,
        activity_id=activity_id, activity_type=tool_name, task_queue=mw._options.task_queue,
        activity_input=[to_json_safe(tool_args)],
        session_id=mw._options.session_id, agent_name=mw._options.agent_name,
    )


def _build_tool_completed(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, activity_id: str, tool_name: str,
    *, result: Any = None, error: str | None = None,
) -> EventEnvelope:
    return build_activity_completed(
        workflow_id=turn.workflow_id, run_id=turn.run_id, workflow_type=mw._workflow_type,
        activity_id=activity_id, activity_type=tool_name, task_queue=mw._options.task_queue,
        result=to_json_safe(result) if result is not None else None, error=error,
        session_id=mw._options.session_id, agent_name=mw._options.agent_name,
    )


async def handle_wrap_tool_call(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, request: Any, handler: Any,
) -> Any:
    """Tool governance: ToolStarted → Tool (activity_scope) → ToolCompleted (async)."""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})

    if tool_name in mw._options.skip_tool_types:
        return await handler(request)

    activity_id = str(uuid.uuid4())
    ctx = _tool_call_context(mw, turn, activity_id, tool_name)

    if mw._options.send_tool_start_event:
        response = await evaluate_lifecycle_async(
            mw, _build_tool_started(mw, turn, activity_id, tool_name, tool_args)
        )
        try:
            await enforce_start_verdict_async(mw, response, ctx)
        except Exception as exc:
            await _close_orphan_start_async(mw, turn, activity_id, tool_name, str(exc))
            raise

    try:
        with activity_scope(ctx, store=mw._runtime.context_store):
            tool_result = await handler(request)
    except Exception as exc:
        await _send_tool_failed_async(mw, turn, activity_id, tool_name, exc)
        raise

    if mw._options.send_tool_end_event:
        await evaluate_lifecycle_async(
            mw, _build_tool_completed(mw, turn, activity_id, tool_name, result=tool_result)
        )
    return tool_result


def handle_wrap_tool_call_sync(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, request: Any, handler: Any,
) -> Any:
    """Tool governance: ToolStarted → Tool (activity_scope) → ToolCompleted (sync)."""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})

    if tool_name in mw._options.skip_tool_types:
        return handler(request)

    activity_id = str(uuid.uuid4())
    ctx = _tool_call_context(mw, turn, activity_id, tool_name)

    if mw._options.send_tool_start_event:
        response = evaluate_lifecycle_sync(
            mw, _build_tool_started(mw, turn, activity_id, tool_name, tool_args)
        )
        try:
            enforce_start_verdict_sync(mw, response, ctx)
        except Exception as exc:
            _close_orphan_start_sync(mw, turn, activity_id, tool_name, str(exc))
            raise

    try:
        with activity_scope(ctx, store=mw._runtime.context_store):
            tool_result = handler(request)
    except Exception as exc:
        _send_tool_failed_sync(mw, turn, activity_id, tool_name, exc)
        raise

    if mw._options.send_tool_end_event:
        evaluate_lifecycle_sync(
            mw, _build_tool_completed(mw, turn, activity_id, tool_name, result=tool_result)
        )
    return tool_result


# ─── Failure-path completions (orphan close + tool-body error) ─────────────


async def _close_orphan_start_async(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, activity_id: str,
    tool_name: str, error: str,
) -> None:
    """Close a stop-shaped ToolStarted with a failed ToolCompleted (same id, C6)."""
    if not mw._options.send_tool_end_event:
        return
    await evaluate_lifecycle_async(
        mw, _build_tool_completed(mw, turn, activity_id, tool_name, error=error)
    )


def _close_orphan_start_sync(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, activity_id: str,
    tool_name: str, error: str,
) -> None:
    if not mw._options.send_tool_end_event:
        return
    evaluate_lifecycle_sync(
        mw, _build_tool_completed(mw, turn, activity_id, tool_name, error=error)
    )


async def _send_tool_failed_async(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, activity_id: str,
    tool_name: str, error: Exception,
) -> None:
    if not mw._options.send_tool_end_event:
        return
    await evaluate_lifecycle_async(
        mw, _build_tool_completed(mw, turn, activity_id, tool_name, error=str(error))
    )


def _send_tool_failed_sync(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, activity_id: str,
    tool_name: str, error: Exception,
) -> None:
    if not mw._options.send_tool_end_event:
        return
    evaluate_lifecycle_sync(
        mw, _build_tool_completed(mw, turn, activity_id, tool_name, error=str(error))
    )
