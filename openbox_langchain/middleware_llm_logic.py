"""LLM-lifecycle helpers for ``wrap_model_call``/``awrap_model_call``.

Split out of ``middleware_hook_handlers.py`` to stay under 200 lines per file.
Resolves the (activity_id, verdict) pair for an LLM call — reusing the
before_agent pre-screen verdict for the FIRST call in a turn — and builds the
LLMCompleted envelope plus OTel trace register/unregister for Layer-2 hook
correlation (mirrors ``core_callback_llm_logic.py``'s trace-registration shape).

"First LLM call" is DERIVED from ``request.messages`` (no prior ``AIMessage``)
rather than a mutable flag on ``MiddlewareTurnState`` — the turn state is now
carried by value through LangGraph's graph state (see ``middleware.py``'s
``OpenBoxAgentState``), so there is no shared mutable object across hook calls
left to flip a flag on; this detection is stateless and concurrency-safe by
construction (each invocation's messages list is its own).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from openbox_core.contracts.context import ActivityContext
from openbox_core.otel.trace_context import raw_trace_id
from opentelemetry import trace as otel_trace

from openbox_langchain.lifecycle_events import build_activity_completed, build_activity_started
from openbox_langchain.middleware_hooks import evaluate_lifecycle_async, evaluate_lifecycle_sync

if TYPE_CHECKING:
    from openbox_core.contracts.events import EventEnvelope
    from openbox_core.contracts.results import EvaluationResult

    from openbox_langchain.middleware import OpenBoxLangChainMiddleware
    from openbox_langchain.middleware_turn_state import MiddlewareTurnState

__all__ = [
    "build_llm_completed_envelope",
    "is_first_llm_call",
    "llm_activity_context",
    "register_llm_trace",
    "resolve_llm_started_verdict_async",
    "resolve_llm_started_verdict_sync",
    "unregister_llm_trace",
]


def _build_llm_started_envelope(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, activity_id: str, prompt_text: str,
) -> EventEnvelope:
    return build_activity_started(
        workflow_id=turn.workflow_id, run_id=turn.run_id, workflow_type=mw._workflow_type,
        activity_id=activity_id, activity_type="llm_call", task_queue=mw._options.task_queue,
        activity_input=[{"prompt": prompt_text}],
        session_id=mw._options.session_id, agent_name=mw._options.agent_name,
    )


def is_first_llm_call(messages: Any) -> bool:
    """True when ``messages`` contains no prior ``AIMessage`` (assistant turn).

    A model request's message list accumulates every prior turn of the SAME
    invocation, so the first call in a run is exactly the one with no
    assistant-authored message yet — this holds for both a flat
    ``list[BaseMessage]`` and dict-shaped messages (``{"role": ..., ...}``).
    """
    if not isinstance(messages, (list, tuple)):
        return True
    for msg in messages:
        role = getattr(msg, "type", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role") or msg.get("type")
        if role in ("ai", "assistant"):
            return False
    return True


def _reuse_pre_screen(
    turn: MiddlewareTurnState, messages: Any
) -> tuple[str, EvaluationResult] | None:
    """Reuse the pre-screen verdict for the FIRST LLM call in a turn, if any."""
    if is_first_llm_call(messages) and turn.pre_screen_response is not None:
        return f"{turn.run_id}-pre", turn.pre_screen_response
    return None


async def resolve_llm_started_verdict_async(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, prompt_text: str, messages: Any,
) -> tuple[str, EvaluationResult | None]:
    """Resolve (activity_id, verdict) for an LLM call — async gate path."""
    reused = _reuse_pre_screen(turn, messages)
    if reused is not None:
        return reused
    activity_id = str(uuid.uuid4())
    if not mw._options.send_llm_start_event:
        return activity_id, None
    envelope = _build_llm_started_envelope(mw, turn, activity_id, prompt_text)
    response = await evaluate_lifecycle_async(mw, envelope)
    return activity_id, response


def resolve_llm_started_verdict_sync(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, prompt_text: str, messages: Any,
) -> tuple[str, EvaluationResult | None]:
    """Sync counterpart of :func:`resolve_llm_started_verdict_async`."""
    reused = _reuse_pre_screen(turn, messages)
    if reused is not None:
        return reused
    activity_id = str(uuid.uuid4())
    if not mw._options.send_llm_start_event:
        return activity_id, None
    envelope = _build_llm_started_envelope(mw, turn, activity_id, prompt_text)
    response = evaluate_lifecycle_sync(mw, envelope)
    return activity_id, response


def build_llm_completed_envelope(
    mw: OpenBoxLangChainMiddleware,
    turn: MiddlewareTurnState,
    activity_id: str,
    response_metadata: dict[str, Any],
) -> EventEnvelope:
    """Build the LLMCompleted envelope (SAME activity_id as LLMStarted — no ``-c``)."""
    return build_activity_completed(
        workflow_id=turn.workflow_id, run_id=turn.run_id, workflow_type=mw._workflow_type,
        activity_id=activity_id, activity_type="llm_call", task_queue=mw._options.task_queue,
        result=response_metadata,
        session_id=mw._options.session_id, agent_name=mw._options.agent_name,
    )


def llm_activity_context(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, activity_id: str
) -> ActivityContext:
    """The ``llm_call`` ActivityContext the model call runs under."""
    return ActivityContext(
        workflow_id=turn.workflow_id, run_id=turn.run_id, workflow_type=mw._workflow_type,
        task_queue=mw._options.task_queue, activity_id=activity_id, activity_type="llm_call",
        agent_name=mw._options.agent_name, session_id=mw._options.session_id,
    )


def register_llm_trace(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, activity_id: str
) -> None:
    """Register the current OTel trace id against this LLM's ActivityContext
    (Layer-2 HTTP-hook correlation). No-op when there is no active span."""
    trace_id = raw_trace_id(otel_trace.get_current_span())
    if not trace_id:
        return
    mw._runtime.context_store.register_trace(
        trace_id, llm_activity_context(mw, turn, activity_id)
    )


def unregister_llm_trace(mw: OpenBoxLangChainMiddleware) -> None:
    """Mandatory cleanup counterpart of :func:`register_llm_trace`."""
    trace_id = raw_trace_id(otel_trace.get_current_span())
    if not trace_id:
        return
    mw._runtime.context_store.unregister_trace(trace_id)
