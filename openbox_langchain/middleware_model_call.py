"""``wrap_model_call``/``awrap_model_call`` hook entrypoints.

Split out of ``middleware_hook_handlers.py`` to stay under 200 lines per file.
Verdict-resolution and trace plumbing are in ``middleware_llm_logic.py``; this
module composes them into the LLMStarted → redaction → Model → LLMCompleted
sequence for both the sync and async hook paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbox_core.context import activity_scope

from openbox_langchain.lifecycle_events import (
    apply_redaction_to_messages,
    extract_human_turn_prompt,
)
from openbox_langchain.middleware_hooks import (
    enforce_start_verdict_async,
    enforce_start_verdict_sync,
    evaluate_lifecycle_async,
    evaluate_lifecycle_sync,
    extract_response_metadata,
    note_completion_verdict,
)
from openbox_langchain.middleware_llm_logic import (
    build_llm_completed_envelope,
    llm_activity_context,
    register_llm_trace,
    resolve_llm_started_verdict_async,
    resolve_llm_started_verdict_sync,
    unregister_llm_trace,
)

if TYPE_CHECKING:
    from openbox_langchain.middleware import OpenBoxLangChainMiddleware
    from openbox_langchain.middleware_turn_state import MiddlewareTurnState

__all__ = ["handle_wrap_model_call", "handle_wrap_model_call_sync"]


async def handle_wrap_model_call(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, request: Any, handler: Any,
) -> Any:
    """LLM governance: LLMStarted → redaction → Model → LLMCompleted (async)."""
    prompt_text = extract_human_turn_prompt(request.messages)
    if not prompt_text.strip():
        return await handler(request)

    activity_id, response = await resolve_llm_started_verdict_async(
        mw, turn, prompt_text, request.messages
    )
    if response is not None:
        await enforce_start_verdict_async(mw, response, llm_activity_context(mw, turn, activity_id))
        if response.guardrails and response.guardrails.redacted_input:
            apply_redaction_to_messages(request.messages, response.guardrails.redacted_input)

    # Bound like a tool body, so Layer-2 HTTP hooks resolve the llm_call
    # activity even when no OTel span is active (trace registration alone is a
    # no-op then, and the model's HTTP calls went unattributed).
    register_llm_trace(mw, turn, activity_id)
    try:
        with activity_scope(
            llm_activity_context(mw, turn, activity_id), store=mw._runtime.context_store
        ):
            model_response = await handler(request)
    finally:
        unregister_llm_trace(mw)

    if mw._options.send_llm_end_event:
        meta = extract_response_metadata(model_response)
        envelope = build_llm_completed_envelope(mw, turn, activity_id, meta)
        note_completion_verdict(await evaluate_lifecycle_async(mw, envelope), "llm_call")
    return model_response


def handle_wrap_model_call_sync(
    mw: OpenBoxLangChainMiddleware, turn: MiddlewareTurnState, request: Any, handler: Any,
) -> Any:
    """LLM governance: LLMStarted → redaction → Model → LLMCompleted (sync)."""
    prompt_text = extract_human_turn_prompt(request.messages)
    if not prompt_text.strip():
        return handler(request)

    activity_id, response = resolve_llm_started_verdict_sync(
        mw, turn, prompt_text, request.messages
    )
    if response is not None:
        enforce_start_verdict_sync(mw, response, llm_activity_context(mw, turn, activity_id))
        if response.guardrails and response.guardrails.redacted_input:
            apply_redaction_to_messages(request.messages, response.guardrails.redacted_input)

    register_llm_trace(mw, turn, activity_id)
    try:
        with activity_scope(
            llm_activity_context(mw, turn, activity_id), store=mw._runtime.context_store
        ):
            model_response = handler(request)
    finally:
        unregister_llm_trace(mw)

    if mw._options.send_llm_end_event:
        meta = extract_response_metadata(model_response)
        completed = evaluate_lifecycle_sync(
            mw, build_llm_completed_envelope(mw, turn, activity_id, meta)
        )
        note_completion_verdict(completed, "llm_call")
    return model_response
