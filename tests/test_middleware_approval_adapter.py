"""Sync approval wait used by the middleware's sync (agent.invoke) path."""

from __future__ import annotations

import pytest
from openbox_core.approvals import ApprovalPoller
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.results import ApprovalResult, EvaluationResult, Verdict
from openbox_core.errors import ApprovalExpiredError, ApprovalRejectedError

from openbox_langchain.middleware_approval_adapter import LangChainCoreAdapter

RESULT = EvaluationResult(verdict=Verdict.REQUIRE_APPROVAL)
CTX = ActivityContext(
    workflow_id="wf-1", run_id="run-1", activity_id="act-1", activity_type="refund_order"
)


class _PollClient:
    def __init__(self, *answers: dict) -> None:
        self.answers = list(answers)
        self.polled: list[tuple[str, str, str]] = []

    def poll_approval(self, workflow_id, run_id, activity_id):
        self.polled.append((workflow_id, run_id, activity_id))
        return ApprovalResult.from_dict(self.answers.pop(0))


def _adapter(*answers: dict) -> tuple[LangChainCoreAdapter, _PollClient]:
    client = _PollClient(*answers)
    poller = ApprovalPoller(client, poll_interval_seconds=0)  # type: ignore[arg-type]
    return LangChainCoreAdapter(approval_poller=poller), client


def test_waits_until_approved():
    adapter, client = _adapter({"action": "require_approval"}, {"action": "allow"})

    adapter.wait_for_approval_sync(RESULT, CTX)

    assert client.polled == [("wf-1", "run-1", "act-1")] * 2


def test_rejection_raises():
    adapter, _ = _adapter({"action": "block", "reason": "not today"})

    with pytest.raises(ApprovalRejectedError, match="not today"):
        adapter.wait_for_approval_sync(RESULT, CTX)


def test_expiry_raises():
    adapter, _ = _adapter({"action": "block", "expired": True})

    with pytest.raises(ApprovalExpiredError):
        adapter.wait_for_approval_sync(RESULT, CTX)


def test_fails_safe_without_a_poller():
    with pytest.raises(ApprovalRejectedError, match="no approval flow"):
        LangChainCoreAdapter().wait_for_approval_sync(RESULT, CTX)


def test_fails_safe_without_activity_ids():
    adapter, client = _adapter()

    with pytest.raises(ApprovalRejectedError, match="ids to poll"):
        adapter.wait_for_approval_sync(RESULT, None)
    assert client.polled == []


async def test_fails_safe_on_a_running_event_loop():
    adapter, client = _adapter()

    with pytest.raises(ApprovalRejectedError, match="ainvoke"):
        adapter.wait_for_approval_sync(RESULT, CTX)
    assert client.polled == []
