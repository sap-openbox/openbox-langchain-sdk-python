"""Tests for middleware_tool_hook.py (ToolStarted -> Tool -> ToolCompleted)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from openbox_core.errors import GovernanceBlockedError, GovernanceHaltError

from openbox_langchain.middleware_tool_hook import handle_wrap_tool_call, handle_wrap_tool_call_sync
from openbox_langchain.middleware_turn_state import MiddlewareTurnState
from tests.fakes_core_callback import (
    FakeAdapter,
    FakeGate,
    FakeRuntime,
    block_result,
    halt_result,
    require_approval_result,
)

WORKFLOW_TYPE = "TestAgent"


def make_mw(*, gate: FakeGate | None = None, adapter: FakeAdapter | None = None) -> MagicMock:
    mw = MagicMock()
    mw._workflow_type = WORKFLOW_TYPE
    mw._options = MagicMock()
    mw._options.task_queue = "langchain"
    mw._options.session_id = None
    mw._options.agent_name = None
    mw._options.skip_tool_types = set()
    mw._options.send_tool_start_event = True
    mw._options.send_tool_end_event = True
    mw._runtime = FakeRuntime(gate=gate or FakeGate(), adapter=adapter or FakeAdapter())
    return mw


def make_turn(*, sync_mode: bool = False) -> MiddlewareTurnState:
    return MiddlewareTurnState.new(sync_mode=sync_mode)


def make_request(name: str = "search_web", args: dict | None = None) -> MagicMock:
    request = MagicMock()
    request.tool_call = {"name": name, "args": args or {"query": "test"}}
    return request


def block_first_gate(*, halt: bool = False) -> FakeGate:
    verdict_fn = halt_result if halt else block_result

    class BlockFirstGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = verdict_fn()
            return await super().aevaluate(event)

        def evaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = verdict_fn()
            return super().evaluate(event)

    return BlockFirstGate()


def approval_first_gate() -> FakeGate:
    class ApprovalFirstGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = require_approval_result()
            return await super().aevaluate(event)

        def evaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = require_approval_result()
            return super().evaluate(event)

    return ApprovalFirstGate()


# ─── handle_wrap_tool_call (async) ─────────────────────────────────────


async def test_handle_wrap_tool_call_sends_tool_started_and_completed():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()
    handler = AsyncMock(return_value="tool result")

    result = await handle_wrap_tool_call(mw, turn, make_request(), handler)

    event_types = [e.payload.get("event_type") for e in gate.sent]
    assert "ActivityStarted" in event_types
    assert "ActivityCompleted" in event_types
    assert result == "tool result"
    handler.assert_awaited_once()


async def test_handle_wrap_tool_call_completion_reuses_start_id_no_suffix():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()
    handler = AsyncMock(return_value="tool result")

    await handle_wrap_tool_call(mw, turn, make_request(), handler)

    started = next(e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted")
    completed = next(e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted")
    assert started.activity_id == completed.activity_id
    assert not (completed.activity_id or "").endswith("-c")


async def test_handle_wrap_tool_call_block_prevents_body_and_raises():
    gate = block_first_gate()
    mw = make_mw(gate=gate)
    turn = make_turn()
    handler = AsyncMock(return_value="tool result")

    with pytest.raises(GovernanceBlockedError):
        await handle_wrap_tool_call(mw, turn, make_request(), handler)

    handler.assert_not_awaited()


async def test_handle_wrap_tool_call_halt_prevents_body_and_raises():
    gate = block_first_gate(halt=True)
    mw = make_mw(gate=gate)
    turn = make_turn()
    handler = AsyncMock(return_value="tool result")

    with pytest.raises(GovernanceHaltError):
        await handle_wrap_tool_call(mw, turn, make_request(), handler)

    handler.assert_not_awaited()


async def test_handle_wrap_tool_call_block_closes_orphan_start_same_id():
    """C6: a stop-shaped ToolStarted closes its own row (same id) before raising."""
    gate = block_first_gate()
    mw = make_mw(gate=gate)
    turn = make_turn()
    handler = AsyncMock(return_value="tool result")

    with pytest.raises(GovernanceBlockedError):
        await handle_wrap_tool_call(mw, turn, make_request(), handler)

    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0].activity_id == completed[0].activity_id
    assert completed[0].payload.get("error")


async def test_handle_wrap_tool_call_require_approval_awaits_adapter():
    gate = approval_first_gate()
    adapter = FakeAdapter()
    adapter.auto_approve = True
    mw = make_mw(gate=gate, adapter=adapter)
    turn = make_turn()
    handler = AsyncMock(return_value="tool result")

    result = await handle_wrap_tool_call(mw, turn, make_request(), handler)

    assert result == "tool result"
    assert len(adapter.approval_calls) == 1
    handler.assert_awaited_once()


async def test_handle_wrap_tool_call_approval_gets_the_tool_activity_ids():
    """Core's verdict does not echo the ids the approval is polled by, so the
    adapter must be handed the tool's own activity context."""

    class ContextAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.contexts = []

        async def handle_approval(self, result, context=None):  # type: ignore[override]
            self.contexts.append(context)

    gate = approval_first_gate()
    adapter = ContextAdapter()
    mw = make_mw(gate=gate, adapter=adapter)
    turn = make_turn()

    handler = AsyncMock(return_value="ok")
    await handle_wrap_tool_call(mw, turn, make_request("refund_order"), handler)

    (ctx,) = adapter.contexts
    started = next(e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted")
    assert (ctx.workflow_id, ctx.run_id, ctx.activity_id) == (
        turn.workflow_id, turn.run_id, started.activity_id
    )
    assert ctx.activity_type == "refund_order"


async def test_handle_wrap_tool_call_require_approval_rejected_never_runs_body():
    gate = approval_first_gate()
    mw = make_mw(gate=gate)  # default adapter rejects (no auto_approve)
    turn = make_turn()
    handler = AsyncMock(return_value="tool result")

    from openbox_core.errors import ApprovalRejectedError

    with pytest.raises(ApprovalRejectedError):
        await handle_wrap_tool_call(mw, turn, make_request(), handler)
    handler.assert_not_awaited()


async def test_handle_wrap_tool_call_skips_governance_for_excluded_tools():
    mw = make_mw()
    mw._options.skip_tool_types = {"internal_tool"}
    turn = make_turn()
    handler = AsyncMock(return_value="result")

    result = await handle_wrap_tool_call(mw, turn, make_request("internal_tool"), handler)

    assert mw._runtime.gate.sent == []
    handler.assert_awaited_once()
    assert result == "result"


async def test_handle_wrap_tool_call_tool_body_error_sends_failed_completion():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()
    handler = AsyncMock(side_effect=RuntimeError("tool crashed"))

    with pytest.raises(RuntimeError, match="tool crashed"):
        await handle_wrap_tool_call(mw, turn, make_request(), handler)

    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    assert len(completed) == 1
    assert completed[0].payload["error"]["message"] == "tool crashed"


async def test_handle_wrap_tool_call_skips_tool_start_when_flag_false():
    mw = make_mw()
    mw._options.send_tool_start_event = False
    turn = make_turn()
    handler = AsyncMock(return_value="result")

    await handle_wrap_tool_call(mw, turn, make_request(), handler)

    event_types = [e.payload.get("event_type") for e in mw._runtime.gate.sent]
    assert "ActivityStarted" not in event_types


async def test_handle_wrap_tool_call_skips_tool_end_when_flag_false():
    mw = make_mw()
    mw._options.send_tool_end_event = False
    turn = make_turn()
    handler = AsyncMock(return_value="result")

    await handle_wrap_tool_call(mw, turn, make_request(), handler)

    event_types = [e.payload.get("event_type") for e in mw._runtime.gate.sent]
    assert "ActivityCompleted" not in event_types


async def test_handle_wrap_tool_call_includes_tool_args_in_started_input():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()
    handler = AsyncMock(return_value="result")

    await handle_wrap_tool_call(mw, turn, make_request(args={"query": "abc"}), handler)

    started = next(e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted")
    assert started.payload.get("activity_input") == [{"query": "abc"}]


# ─── handle_wrap_tool_call_sync — sync BLOCK never runs the tool (regression) ──


def test_handle_wrap_tool_call_sync_block_never_runs_tool():
    """Sync agent.invoke with BLOCK on ToolStarted never runs the tool body."""
    gate = block_first_gate()
    mw = make_mw(gate=gate)
    turn = make_turn(sync_mode=True)
    handler = MagicMock(return_value="tool result")

    with pytest.raises(GovernanceBlockedError):
        handle_wrap_tool_call_sync(mw, turn, make_request(), handler)

    handler.assert_not_called()


def test_handle_wrap_tool_call_sync_allow_runs_tool_and_completes():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn(sync_mode=True)
    handler = MagicMock(return_value="tool result")

    result = handle_wrap_tool_call_sync(mw, turn, make_request(), handler)

    assert result == "tool result"
    handler.assert_called_once()
    event_types = [e.payload.get("event_type") for e in gate.sent]
    assert "ActivityStarted" in event_types
    assert "ActivityCompleted" in event_types


def test_handle_wrap_tool_call_sync_waits_for_approval_with_tool_ids():
    """agent.invoke waits for the decision instead of refusing, so the
    dashboard request is not left dangling."""

    class WaitingAdapter(FakeAdapter):
        handle_approval_sync = None  # the LangChainCoreAdapter shape

        def __init__(self) -> None:
            super().__init__()
            self.waited = []

        def wait_for_approval_sync(self, result, context):
            self.waited.append(context)

    gate = approval_first_gate()
    adapter = WaitingAdapter()
    mw = make_mw(gate=gate, adapter=adapter)
    turn = make_turn(sync_mode=True)
    handler = MagicMock(return_value="ok")

    assert handle_wrap_tool_call_sync(mw, turn, make_request("refund_order"), handler) == "ok"

    (ctx,) = adapter.waited
    started = next(e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted")
    assert (ctx.run_id, ctx.activity_id, ctx.activity_type) == (
        turn.run_id, started.activity_id, "refund_order"
    )
    handler.assert_called_once()


def test_handle_wrap_tool_call_sync_require_approval_fails_shut():
    """M15: sync REQUIRE_APPROVAL with no handle_approval_sync fails safe."""
    from openbox_core.errors import ApprovalRejectedError

    gate = approval_first_gate()
    mw = make_mw(gate=gate)
    turn = make_turn(sync_mode=True)
    handler = MagicMock(return_value="tool result")

    with pytest.raises(ApprovalRejectedError):
        handle_wrap_tool_call_sync(mw, turn, make_request(), handler)
    handler.assert_not_called()


def test_handle_wrap_tool_call_sync_tool_body_error_sends_failed_completion():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn(sync_mode=True)
    handler = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        handle_wrap_tool_call_sync(mw, turn, make_request(), handler)

    completed = [e for e in gate.sent if e.payload.get("event_type") == "ActivityCompleted"]
    assert len(completed) == 1
    assert completed[0].payload["error"]["message"] == "boom"


def test_handle_wrap_tool_call_sync_skips_governance_for_excluded_tools():
    mw = make_mw()
    mw._options.skip_tool_types = {"internal_tool"}
    turn = make_turn(sync_mode=True)
    handler = MagicMock(return_value="result")

    result = handle_wrap_tool_call_sync(mw, turn, make_request("internal_tool"), handler)

    assert mw._runtime.gate.sent == []
    handler.assert_called_once()
    assert result == "result"
