"""Shared helpers for OpenBoxLangChainMiddleware hooks.

Envelope building and prompt/redaction logic are NOT duplicated here — they
live in ``lifecycle_events.py`` (shared with the core callback, Phase 2).
This module holds middleware-specific concerns: WorkflowStarted/
WorkflowCompleted/SignalReceived envelope builders (the core callback never
sends these — they are middleware-only events) and inline verdict enforcement
(BLOCK/HALT raise, REQUIRE_APPROVAL awaits the poller on async / fails-shut on
sync — M15). Message/response extraction lives in
``middleware_message_extraction.py`` (split to stay under 200 lines/file;
re-exported here for a single hook-helpers import surface).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from openbox_core.adapters.base import adapter_accepts_context
from openbox_core.contracts.events import (
    EventEnvelope,
    signal_received,
    workflow_completed,
    workflow_started,
)
from openbox_core.errors import ApprovalRejectedError
from openbox_core.serialization import to_json_safe

from openbox_langchain.middleware_message_extraction import (
    extract_last_user_message,
    extract_response_metadata,
)

if TYPE_CHECKING:
    from openbox_core.contracts.context import ActivityContext
    from openbox_core.contracts.results import EvaluationResult

    from openbox_langchain.middleware import OpenBoxLangChainMiddleware
    from openbox_langchain.middleware_turn_state import MiddlewareTurnState

_logger = logging.getLogger(__name__)

__all__ = [
    "build_signal_received_envelope",
    "build_workflow_completed_envelope",
    "build_workflow_started_envelope",
    "enforce_start_verdict_async",
    "enforce_start_verdict_sync",
    "evaluate_lifecycle_async",
    "evaluate_lifecycle_sync",
    "extract_last_user_message",
    "extract_response_metadata",
    "note_completion_verdict",
]


# ─── Envelope builders (workflow/signal — middleware-only events) ──────────


def build_workflow_started_envelope(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, state: Any
) -> EventEnvelope:
    """Build the WorkflowStarted envelope for ``before_agent``."""
    return workflow_started(
        workflow_id=turn.workflow_id,
        run_id=turn.run_id,
        workflow_type=mw._workflow_type,
        task_queue=mw._options.task_queue,
        extra=_session_extra(mw, {"activity_input": to_json_safe(state)}),
    )


def build_workflow_completed_envelope(
    mw: OpenBoxLangChainMiddleware,
    turn: MiddlewareTurnState,
    *,
    workflow_output: Any = None,
    error: str | None = None,
) -> EventEnvelope:
    """Build the WorkflowCompleted envelope for ``after_agent`` (or an early
    enforcement-error exit from ``before_agent``)."""
    extra: dict[str, Any] = {}
    if workflow_output is not None:
        extra["workflow_output"] = to_json_safe(workflow_output)
    if error is not None:
        extra["error"] = error
        extra["status"] = "failed"
    else:
        extra["status"] = "completed"
    return workflow_completed(
        workflow_id=turn.workflow_id,
        run_id=turn.run_id,
        workflow_type=mw._workflow_type,
        task_queue=mw._options.task_queue,
        extra=_session_extra(mw, extra),
    )


def build_signal_received_envelope(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, user_prompt: str
) -> EventEnvelope:
    """Build the SignalReceived envelope for the initial user prompt."""
    return signal_received(
        workflow_id=turn.workflow_id,
        run_id=turn.run_id,
        workflow_type=mw._workflow_type,
        signal_name="user_prompt",
        task_queue=mw._options.task_queue,
        extra=_session_extra(mw, {"signal_args": [user_prompt]}),
    )


def _session_extra(mw: OpenBoxLangChainMiddleware, base: dict[str, Any]) -> dict[str, Any]:
    """Merge session_id/agent_name wire passthrough (same rule as lifecycle_events)."""
    if mw._options.session_id is not None:
        base.setdefault("session_id", mw._options.session_id)
    if mw._options.agent_name is not None:
        base.setdefault("agent_name", mw._options.agent_name)
    return base


# ─── Lifecycle evaluation (sync/async gate dispatch) ───────────────────────


async def evaluate_lifecycle_async(
    mw: OpenBoxLangChainMiddleware, event: EventEnvelope
) -> EvaluationResult:
    """Send a lifecycle event via ``gate.aevaluate`` (telemetry-only — no
    enforcement here; callers that must enforce use
    :func:`enforce_start_verdict_async`)."""
    return await mw._runtime.gate.aevaluate(event)


def note_completion_verdict(result: EvaluationResult, activity_type: str) -> None:
    """Log a REQUIRE_APPROVAL that arrives on ActivityCompleted.

    The tool/LLM call already ran, so there is nothing left to gate: the run
    continues without waiting, as in the LangGraph and Temporal SDKs.
    """
    if result.verdict.requires_approval():
        _logger.warning(
            "REQUIRE_APPROVAL returned after %s completed; continuing without "
            "waiting. Match the rule on ActivityStarted to gate the operation.%s",
            activity_type,
            f" Reason: {result.reason}" if result.reason else "",
        )


def evaluate_lifecycle_sync(
    mw: OpenBoxLangChainMiddleware, event: EventEnvelope
) -> EvaluationResult:
    """Sync counterpart of :func:`evaluate_lifecycle_async` (``gate.evaluate``)."""
    return mw._runtime.gate.evaluate(event)


# ─── Inline verdict enforcement (wrap hooks enforce BEFORE the wrapped call) ──


async def enforce_start_verdict_async(
    mw: OpenBoxLangChainMiddleware,
    result: EvaluationResult,
    context: ActivityContext | None = None,
) -> None:
    """Enforce a start-stage verdict on the ASYNC path.

    BLOCK/HALT raise via ``adapter.raise_lifecycle_blocked`` (native core
    error types). REQUIRE_APPROVAL drives ``adapter.handle_approval`` — a real
    wait when an ``ApprovalPoller`` is configured, else fail-safe rejection.
    ``context`` is the activity awaiting approval: Core's verdict does not echo
    the workflow/run/activity ids the approval is polled by.
    """
    if result.verdict.should_stop():
        mw._runtime.adapter.raise_lifecycle_blocked(result)
        return
    if result.verdict.requires_approval():
        handle_approval = mw._runtime.adapter.handle_approval
        if context is not None and adapter_accepts_context(handle_approval):
            await handle_approval(result, context=context)
        else:
            await handle_approval(result)


def enforce_start_verdict_sync(
    mw: OpenBoxLangChainMiddleware,
    result: EvaluationResult,
    context: ActivityContext | None = None,
) -> None:
    """Enforce a start-stage verdict on the SYNC path.

    BLOCK/HALT raise the same way as the async path. REQUIRE_APPROVAL blocks
    until the approval is decided (``LangChainCoreAdapter``), so a decision on
    the dashboard takes effect for ``agent.invoke`` too. It fails safe — the
    operation does not run — without an approval poller, without ``context``
    (the ids the approval is polled by), or on a running event loop.
    """
    if result.verdict.should_stop():
        mw._runtime.adapter.raise_lifecycle_blocked(result)
        return
    if result.verdict.requires_approval():
        handle_approval_sync = getattr(mw._runtime.adapter, "handle_approval_sync", None)
        if callable(handle_approval_sync):
            handle_approval_sync(result)
            return
        wait_for_approval_sync = getattr(mw._runtime.adapter, "wait_for_approval_sync", None)
        if callable(wait_for_approval_sync):
            wait_for_approval_sync(result, context)
            return
        raise ApprovalRejectedError(
            "REQUIRE_APPROVAL verdict on the sync middleware path but the adapter "
            "defines no handle_approval_sync — failing safe (operation not run)"
        )
