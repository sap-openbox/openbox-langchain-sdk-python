"""Core adapter with a blocking approval wait for the sync middleware path.

``agent.invoke`` runs the middleware's sync hooks, which cannot ``await`` the
base ``ApprovalPoller``. Refusing outright left the approval request pending on
the dashboard with nothing waiting on it, so a decision there had no effect.
This adapter waits on the poller synchronously instead, keyed by the activity
the verdict was given for.

The method is deliberately not named ``handle_approval_sync``: the base SDK's
HTTP preflight delegates to an adapter's ``handle_approval_sync`` when present,
and that path keeps its own poller and loop-liveness rules.
"""

from __future__ import annotations

import asyncio

from openbox_core.adapters.base import CoreAdapter
from openbox_core.contracts.context import ActivityContext
from openbox_core.contracts.results import EvaluationResult
from openbox_core.errors import ApprovalExpiredError, ApprovalRejectedError

__all__ = ["LangChainCoreAdapter"]


def _on_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class LangChainCoreAdapter(CoreAdapter):
    """``CoreAdapter`` plus :meth:`wait_for_approval_sync`."""

    def wait_for_approval_sync(
        self, result: EvaluationResult, context: ActivityContext | None
    ) -> None:
        """Block until the approval is decided. Returns when approved.

        Fails safe (the operation does not run) without an approval poller,
        without the activity ids to poll by, or when called on a running event
        loop, where a blocking wait would stall every other task.
        """
        if self._poller is None:
            raise ApprovalRejectedError(
                "REQUIRE_APPROVAL verdict but no approval flow is configured — "
                "failing safe (operation not run)"
            )
        if context is None or not (context.workflow_id and context.run_id and context.activity_id):
            raise ApprovalRejectedError(
                "REQUIRE_APPROVAL verdict without the workflow/run/activity ids to "
                "poll — failing safe (operation not run)"
            )
        if _on_running_loop():
            raise ApprovalRejectedError(
                "REQUIRE_APPROVAL verdict on a sync call made from a running event "
                "loop — use agent.ainvoke to wait for the decision (operation not run)"
            )
        approval = self._poller.wait_for_decision(
            context.workflow_id, context.run_id, context.activity_id
        )
        if approval.allow_shaped:
            return
        if approval.expired:
            raise ApprovalExpiredError(approval.reason or "Approval window expired")
        raise ApprovalRejectedError(approval.reason or "Approval rejected")
