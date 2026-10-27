"""Tests for middleware_hook_handlers.py (before_agent/after_agent, sync + async)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from openbox_core.errors import GovernanceAPIError, GovernanceBlockedError

from openbox_langchain.middleware_hook_handlers import (
    handle_after_agent,
    handle_after_agent_sync,
    handle_before_agent,
    handle_before_agent_sync,
)
from openbox_langchain.middleware_turn_state import MiddlewareTurnState
from tests.fakes_core_callback import (
    FakeAdapter,
    FakeGate,
    FakeRuntime,
    block_result,
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
    mw._options.send_chain_start_event = True
    mw._options.send_chain_end_event = True
    mw._options.send_llm_start_event = True
    mw._runtime = FakeRuntime(gate=gate or FakeGate(), adapter=adapter or FakeAdapter())
    return mw


def make_turn(*, sync_mode: bool = False) -> MiddlewareTurnState:
    return MiddlewareTurnState.new(sync_mode=sync_mode)


STATE_WITH_MESSAGES = {"messages": [{"role": "user", "content": "Hello"}]}
STATE_NO_USER_MESSAGE = {"messages": [{"role": "assistant", "content": "Only AI"}]}


# ─── handle_before_agent (async) ─────────────────────────────────────────


async def test_handle_before_agent_sends_signal_received():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()

    await handle_before_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    event_types = [e.payload.get("event_type") for e in gate.sent]
    assert "SignalReceived" in event_types


async def test_handle_before_agent_sends_workflow_started():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()

    await handle_before_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    event_types = [e.payload.get("event_type") for e in gate.sent]
    assert "WorkflowStarted" in event_types


async def test_handle_before_agent_sends_pre_screen_llm_started():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()

    await handle_before_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    started = [e for e in gate.sent if e.payload.get("event_type") == "ActivityStarted"]
    assert len(started) == 1
    assert started[0].activity_id == f"{turn.run_id}-pre"


async def test_handle_before_agent_caches_pre_screen_response():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()

    result = await handle_before_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    # MiddlewareTurnState is frozen (no shared mutable turn state, M17 fix) —
    # the pre-screen verdict comes back on the RETURNED turn, not the input.
    assert result.pre_screen_response is not None
    assert result.pre_screen_response.verdict.value == "allow"


async def test_handle_before_agent_no_user_message_skips_pre_screen():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()

    result = await handle_before_agent(mw, turn, STATE_NO_USER_MESSAGE, MagicMock())

    event_types = [e.payload.get("event_type") for e in gate.sent]
    assert "WorkflowStarted" in event_types
    assert "ActivityStarted" not in event_types
    assert result.pre_screen_response is None


async def test_handle_before_agent_skips_events_when_flags_false():
    mw = make_mw()
    mw._options.send_chain_start_event = False
    mw._options.send_llm_start_event = False
    turn = make_turn()

    await handle_before_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    event_types = [e.payload.get("event_type") for e in mw._runtime.gate.sent]
    assert "WorkflowStarted" not in event_types
    assert "ActivityStarted" not in event_types


async def test_handle_before_agent_sends_workflow_started_before_signal():
    """Core holds every other event for a run until WorkflowStarted has
    created its session (SessionNotReady), so it must go out first."""
    gate = FakeGate()
    await handle_before_agent(make_mw(gate=gate), make_turn(), STATE_WITH_MESSAGES, MagicMock())

    event_types = [e.payload.get("event_type") for e in gate.sent]
    assert event_types[:2] == ["WorkflowStarted", "SignalReceived"]


def test_handle_before_agent_sync_sends_workflow_started_before_signal():
    gate = FakeGate()
    handle_before_agent_sync(
        make_mw(gate=gate), make_turn(sync_mode=True), STATE_WITH_MESSAGES, MagicMock()
    )

    event_types = [e.payload.get("event_type") for e in gate.sent]
    assert event_types[:2] == ["WorkflowStarted", "SignalReceived"]


async def test_handle_before_agent_block_raises_and_sends_workflow_completed_failed():
    class BlockFirstGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = block_result("no")
            return await super().aevaluate(event)

    gate = BlockFirstGate()
    mw = make_mw(gate=gate)
    turn = make_turn()

    with pytest.raises(GovernanceBlockedError):
        await handle_before_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    completed = [e for e in gate.sent if e.payload.get("event_type") == "WorkflowCompleted"]
    assert len(completed) == 1
    assert completed[0].payload.get("status") == "failed"


class _BlockThenRejectCloseGate(FakeGate):
    """Blocks the pre-screen, then fails the closing event the way Core does
    once a stop has already closed the session (HTTP 400, fail-closed)."""

    def _stop_or_reject(self, event):
        if event.event_type.value == "ActivityStarted":
            self.verdicts[event.activity_id] = block_result("banned words")
        if event.event_type.value == "WorkflowCompleted":
            raise GovernanceAPIError("Governance API error: HTTP 400")

    async def aevaluate(self, event):  # type: ignore[override]
        self._stop_or_reject(event)
        return await super().aevaluate(event)

    def evaluate(self, event):  # type: ignore[override]
        self._stop_or_reject(event)
        return super().evaluate(event)


async def test_handle_before_agent_stop_survives_rejected_closing_event():
    mw = make_mw(gate=_BlockThenRejectCloseGate())

    with pytest.raises(GovernanceBlockedError):
        await handle_before_agent(mw, make_turn(), STATE_WITH_MESSAGES, MagicMock())


def test_handle_before_agent_sync_stop_survives_rejected_closing_event():
    mw = make_mw(gate=_BlockThenRejectCloseGate())

    with pytest.raises(GovernanceBlockedError):
        handle_before_agent_sync(mw, make_turn(sync_mode=True), STATE_WITH_MESSAGES, MagicMock())


async def test_handle_before_agent_require_approval_awaits_adapter():
    class ApprovalFirstGate(FakeGate):
        async def aevaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = require_approval_result()
            return await super().aevaluate(event)

    adapter = FakeAdapter()
    adapter.auto_approve = True
    mw = make_mw(gate=ApprovalFirstGate(), adapter=adapter)
    turn = make_turn()

    result = await handle_before_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    assert len(adapter.approval_calls) == 1
    assert result.pre_screen_response is not None


# ─── handle_before_agent_sync ─────────────────────────────────────────────


def test_handle_before_agent_sync_sends_same_sequence():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn(sync_mode=True)

    result = handle_before_agent_sync(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    event_types = [e.payload.get("event_type") for e in gate.sent]
    assert event_types.count("SignalReceived") == 1
    assert event_types.count("WorkflowStarted") == 1
    assert event_types.count("ActivityStarted") == 1
    assert result.pre_screen_response is not None


def test_handle_before_agent_sync_block_raises_via_sync_gate():
    class BlockFirstGate(FakeGate):
        def evaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = block_result()
            return super().evaluate(event)

    mw = make_mw(gate=BlockFirstGate())
    turn = make_turn(sync_mode=True)

    with pytest.raises(GovernanceBlockedError):
        handle_before_agent_sync(mw, turn, STATE_WITH_MESSAGES, MagicMock())


def test_handle_before_agent_sync_require_approval_fails_shut():
    """M15: sync REQUIRE_APPROVAL with the default FakeAdapter (no auto_approve,
    no handle_approval_sync) fails safe via handle_approval_sync's ApprovalRejectedError."""
    from openbox_core.errors import ApprovalRejectedError

    class ApprovalFirstGate(FakeGate):
        def evaluate(self, event):  # type: ignore[override]
            if event.event_type.value == "ActivityStarted":
                self.verdicts[event.activity_id] = require_approval_result()
            return super().evaluate(event)

    mw = make_mw(gate=ApprovalFirstGate())
    turn = make_turn(sync_mode=True)

    with pytest.raises(ApprovalRejectedError):
        handle_before_agent_sync(mw, turn, STATE_WITH_MESSAGES, MagicMock())


# ─── handle_after_agent ────────────────────────────────────────────────


async def test_handle_after_agent_sends_workflow_completed():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()

    result = await handle_after_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    assert result is None
    completed = [e for e in gate.sent if e.payload.get("event_type") == "WorkflowCompleted"]
    assert len(completed) == 1
    assert completed[0].payload["workflow_id"] == turn.workflow_id
    assert completed[0].payload["run_id"] == turn.run_id


async def test_handle_after_agent_skips_when_flag_false():
    mw = make_mw()
    mw._options.send_chain_end_event = False
    turn = make_turn()

    await handle_after_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    assert mw._runtime.gate.sent == []


async def test_handle_after_agent_includes_last_message_content():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()
    state = {"messages": [{"role": "assistant", "content": "Final response"}]}

    await handle_after_agent(mw, turn, state, MagicMock())

    completed = next(e for e in gate.sent if e.payload.get("event_type") == "WorkflowCompleted")
    assert completed.payload.get("workflow_output") == {"result": "Final response"}


def test_handle_after_agent_sync_sends_workflow_completed():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn(sync_mode=True)

    result = handle_after_agent_sync(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    assert result is None
    completed = [e for e in gate.sent if e.payload.get("event_type") == "WorkflowCompleted"]
    assert len(completed) == 1


def test_handle_after_agent_sync_skips_when_flag_false():
    mw = make_mw()
    mw._options.send_chain_end_event = False
    turn = make_turn(sync_mode=True)

    handle_after_agent_sync(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    assert mw._runtime.gate.sent == []


# ─── Workflow/run id reuse across before/after (same turn) ────────────────


async def test_before_and_after_agent_share_workflow_and_run_id():
    gate = FakeGate()
    mw = make_mw(gate=gate)
    turn = make_turn()

    await handle_before_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())
    await handle_after_agent(mw, turn, STATE_WITH_MESSAGES, MagicMock())

    started = next(e for e in gate.sent if e.payload.get("event_type") == "WorkflowStarted")
    completed = next(e for e in gate.sent if e.payload.get("event_type") == "WorkflowCompleted")
    assert started.payload["workflow_id"] == completed.payload["workflow_id"] == turn.workflow_id
    assert started.payload["run_id"] == completed.payload["run_id"] == turn.run_id
