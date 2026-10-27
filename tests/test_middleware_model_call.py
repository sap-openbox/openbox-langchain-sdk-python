"""Tests for middleware_model_call.py (LLMStarted -> Model -> LLMCompleted)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from openbox_core.contracts.results import EvaluationResult, GuardrailsResult, Verdict
from openbox_core.errors import ApprovalRejectedError, GovernanceBlockedError

from openbox_langchain.middleware_model_call import (
    handle_wrap_model_call,
    handle_wrap_model_call_sync,
)
from openbox_langchain.middleware_turn_state import MiddlewareTurnState
from tests.fakes_core_callback import FakeAdapter, FakeGate, FakeRuntime, block_result

WORKFLOW_TYPE = "TestAgent"


def make_mw(*, gate: FakeGate | None = None, adapter: FakeAdapter | None = None) -> MagicMock:
    mw = MagicMock()
    mw._workflow_type = WORKFLOW_TYPE
    mw._options = MagicMock()
    mw._options.task_queue = "langchain"
    mw._options.session_id = None
    mw._options.agent_name = None
    mw._options.send_llm_start_event = True
    mw._options.send_llm_end_event = True
    mw._runtime = FakeRuntime(gate=gate or FakeGate(), adapter=adapter or FakeAdapter())
    return mw


def make_turn(
    *, sync_mode: bool = False, pre_screen: EvaluationResult | None = None
) -> MiddlewareTurnState:
    turn = MiddlewareTurnState.new(sync_mode=sync_mode)
    if pre_screen is not None:
        # MiddlewareTurnState is frozen (M17: carried by value through graph
        # state, no shared mutable instance) — use with_pre_screen() to derive
        # a new instance rather than assigning the field directly.
        turn = turn.with_pre_screen(pre_screen)
    return turn


def make_request(text: str = "Hello") -> MagicMock:
    request = MagicMock()
    request.messages = [{"role": "user", "content": text}]
    return request


def block_first_gate() -> FakeGate:
    class BlockFirstGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = block_result()
            return await super().aevaluate(event)

        def evaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = block_result()
            return super().evaluate(event)

    return BlockFirstGate()



async def test_handle_wrap_model_call_binds_llm_activity_during_model_call():
    """The model's HTTP calls must resolve the llm_call activity even with no
    active OTel span — trace registration alone is a no-op then."""
    gate = FakeGate()
    mw = make_mw(gate=gate)
    seen = []

    async def handler(_request):
        seen.append(mw._runtime.context_store.current_activity_context())
        return MagicMock(content="Response")

    await handle_wrap_model_call(mw, make_turn(), make_request(), handler)

    (ctx,) = seen
    started = next(e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted")
    assert ctx is not None
    assert ctx.activity_type == "llm_call"
    assert ctx.activity_id == started.activity_id
    assert mw._runtime.context_store.current_activity_context() is None


def test_handle_wrap_model_call_sync_binds_llm_activity_during_model_call():
    mw = make_mw()
    seen = []

    def handler(_request):
        seen.append(mw._runtime.context_store.current_activity_context())
        return MagicMock(content="Response")

    handle_wrap_model_call_sync(mw, make_turn(sync_mode=True), make_request(), handler)

    (ctx,) = seen
    assert ctx is not None and ctx.activity_type == "llm_call"
    assert mw._runtime.context_store.current_activity_context() is None


# ─── handle_wrap_model_call (async) ────────────────────────────────────


async def test_handle_wrap_model_call_sends_llm_started_and_completed():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()
    handler = AsyncMock(return_value=MagicMock(content="Response"))

    await handle_wrap_model_call(mw, turn, make_request(), handler)

    event_types = [e.payload.get("event_type") for e in gate.sent]
    assert "ActivityStarted" in event_types
    assert "ActivityCompleted" in event_types
    handler.assert_awaited_once()


async def test_handle_wrap_model_call_completion_reuses_start_id_no_suffix():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()
    handler = AsyncMock(return_value=MagicMock(content="Response"))

    await handle_wrap_model_call(mw, turn, make_request(), handler)

    started = next(e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted")
    completed = next(e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted")
    assert started.activity_id == completed.activity_id
    assert not (completed.activity_id or "").endswith("-c")


async def test_handle_wrap_model_call_skips_empty_prompt():
    mw = make_mw()
    turn = make_turn()
    request = make_request(text="")
    handler = AsyncMock(return_value=MagicMock(content="Response"))

    result = await handle_wrap_model_call(mw, turn, request, handler)

    handler.assert_awaited_once()
    assert mw._runtime.gate.sent == []
    assert result.content == "Response"


async def test_handle_wrap_model_call_reuses_pre_screen_response():
    """The FIRST LLM call (no prior AIMessage in request.messages) reuses the
    before_agent pre-screen verdict — detected statelessly from the message
    history rather than a mutable flag on turn (M17: turn is frozen/carried by
    value through graph state, not a shared mutable instance)."""
    pre_screen = EvaluationResult(verdict=Verdict.ALLOW)
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn(pre_screen=pre_screen)
    handler = AsyncMock(return_value=MagicMock(content="Response"))

    await handle_wrap_model_call(mw, turn, make_request(), handler)

    # No fresh ActivityStarted gate call — the pre-screen verdict was reused.
    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    assert len(started) == 0
    completed = next(e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted")
    assert completed.activity_id == f"{turn.run_id}-pre"


async def test_handle_wrap_model_call_second_call_does_not_reuse_pre_screen():
    """A SECOND LLM call in the same turn (request.messages already contains
    an AIMessage from the first call) re-evaluates fresh rather than reusing
    the same turn's pre-screen verdict again."""
    from langchain_core.messages import AIMessage

    pre_screen = EvaluationResult(verdict=Verdict.ALLOW)
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn(pre_screen=pre_screen)
    handler = AsyncMock(return_value=MagicMock(content="Response"))

    request = make_request()
    request.messages = [*request.messages, AIMessage(content="prior turn reply")]

    await handle_wrap_model_call(mw, turn, request, handler)

    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    assert len(started) == 1
    completed = next(e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted")
    assert completed.activity_id != f"{turn.run_id}-pre"


async def test_handle_wrap_model_call_applies_pii_redaction():
    guardrails = GuardrailsResult(redacted_input="Redacted", input_type="activity_input")
    pre_screen = EvaluationResult(verdict=Verdict.ALLOW, guardrails=guardrails)
    mw = make_mw()
    turn = make_turn(pre_screen=pre_screen)
    request = MagicMock()
    msg = MagicMock(type="human", content="Original")
    request.messages = [msg]
    handler = AsyncMock(return_value=MagicMock(content="Response"))

    await handle_wrap_model_call(mw, turn, request, handler)

    assert request.messages[0].content == "Redacted"


async def test_handle_wrap_model_call_block_raises_before_handler():
    gate = block_first_gate()
    mw = make_mw(gate=gate)
    turn = make_turn()
    handler = AsyncMock(return_value=MagicMock(content="Response"))

    with pytest.raises(GovernanceBlockedError):
        await handle_wrap_model_call(mw, turn, make_request(), handler)

    handler.assert_not_awaited()


async def test_handle_wrap_model_call_skips_llm_start_when_flag_false():
    mw = make_mw()
    mw._options.send_llm_start_event = False
    turn = make_turn()
    handler = AsyncMock(return_value=MagicMock(content="Response"))

    await handle_wrap_model_call(mw, turn, make_request(), handler)

    event_types = [e.payload.get("event_type") for e in mw._runtime.gate.sent]
    assert "ActivityStarted" not in event_types


# ─── handle_wrap_model_call_sync ────────────────────────────────────────


def test_handle_wrap_model_call_sync_sends_llm_started_and_completed():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn(sync_mode=True)
    handler = MagicMock(return_value=MagicMock(content="Response"))

    handle_wrap_model_call_sync(mw, turn, make_request(), handler)

    event_types = [e.payload.get("event_type") for e in gate.sent]
    assert "ActivityStarted" in event_types
    assert "ActivityCompleted" in event_types
    handler.assert_called_once()


def test_handle_wrap_model_call_sync_block_raises_before_handler():
    gate = block_first_gate()
    mw = make_mw(gate=gate)
    turn = make_turn(sync_mode=True)
    handler = MagicMock(return_value=MagicMock(content="Response"))

    with pytest.raises(GovernanceBlockedError):
        handle_wrap_model_call_sync(mw, turn, make_request(), handler)

    handler.assert_not_called()


def test_handle_wrap_model_call_sync_require_approval_fails_shut():
    class ApprovalFirstGate(FakeGate):
        def evaluate(self, event):  # type: ignore[override]
            from tests.fakes_core_callback import require_approval_result

            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = require_approval_result()
            return super().evaluate(event)

    mw = make_mw(gate=ApprovalFirstGate())
    turn = make_turn(sync_mode=True)
    handler = MagicMock(return_value=MagicMock(content="Response"))

    with pytest.raises(ApprovalRejectedError):
        handle_wrap_model_call_sync(mw, turn, make_request(), handler)
    handler.assert_not_called()
