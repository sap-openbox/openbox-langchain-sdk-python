"""OpenBox governance middleware for LangChain agents (``openbox_core`` based).

Subclasses ``AgentMiddleware`` to intercept agent lifecycle and enforce
governance INLINE (wrap hooks run in the execution path):

    before_agent    → WorkflowStarted + SignalReceived + pre-screen guardrails
    wrap_model_call → LLMStarted (redaction) → Model → LLMCompleted
    wrap_tool_call  → ToolStarted → Tool (activity_scope) → ToolCompleted
    after_agent     → WorkflowCompleted + cleanup

Requires the optional ``agent`` extra (``pip install
openbox-langchain-sdk-python[agent]``) — this module imports ``langchain``
(``langchain.agents.middleware.types``), never ``langgraph``/``openbox_langgraph``.

Usage:
    from openbox_langchain import create_openbox_langchain_middleware
    middleware = create_openbox_langchain_middleware(api_url=..., api_key=...)
    agent = create_agent(model=..., tools=[...], middleware=[middleware])
    result = agent.invoke({"messages": [("user", "Hello")]})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

try:
    from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
except ImportError as exc:  # pragma: no cover - surfaced to caller
    raise ImportError(
        "OpenBoxLangChainMiddleware requires the optional 'agent' extra: "
        "pip install openbox-langchain-sdk-python[agent]"
    ) from exc

from openbox_langchain.middleware_runtime_builder import build_middleware_runtime
from openbox_langchain.middleware_state import OpenBoxAgentState
from openbox_langchain.middleware_tool_call_protocol import ToolCallRequestLike
from openbox_langchain.middleware_turn_state import (
    MiddlewareTurnState,
    extract_thread_id,
    require_turn_state,
    require_turn_state_from_request,
)

_logger = logging.getLogger("openbox_langchain")

__all__ = [
    "OpenBoxAgentState",
    "OpenBoxLangChainMiddleware",
    "OpenBoxLangChainMiddlewareOptions",
    "ToolCallRequestLike",
]


@dataclass
class OpenBoxLangChainMiddlewareOptions:
    """Configuration for ``OpenBoxLangChainMiddleware``.

    ``api_url``/``api_key`` default to None (falls through to
    ``OPENBOX_LANGCHAIN_*``/``OPENBOX_*`` env resolution — see
    ``middleware_runtime_builder.py``); the factory
    (``create_openbox_langchain_middleware``) always threads its explicit
    values through here.
    """

    api_url: str | None = None
    api_key: str | None = None
    agent_name: str | None = None
    agent_did: str | None = None
    agent_private_key: str | None = None
    workload_private_key: str | None = field(default=None, repr=False)
    session_id: str | None = None
    task_queue: str = "langchain"
    on_api_error: str = "fail_open"
    governance_timeout: float = 30.0
    tool_type_map: dict[str, str] = field(default_factory=dict)
    skip_tool_types: set[str] = field(default_factory=set)
    send_chain_start_event: bool = True
    send_chain_end_event: bool = True
    send_llm_start_event: bool = True
    send_llm_end_event: bool = True
    send_tool_start_event: bool = True
    send_tool_end_event: bool = True
    # HITL: when set, the async path gets a real approval wait via ApprovalPoller.
    # The sync path always fails-shut on REQUIRE_APPROVAL regardless (M15) — see
    # middleware_hooks.enforce_start_verdict_sync.
    approval_poll_interval_seconds: float = 5.0
    approval_max_wait_seconds: float | None = None


class OpenBoxLangChainMiddleware(AgentMiddleware):
    """AgentMiddleware implementing OpenBox governance for LangChain agents.

    Owns exactly ONE ``OpenBoxRuntime`` per middleware instance (private
    ``ContextStore``), installs base instrumentation, and wraps tool execution
    in ``activity_scope(...)``. Turn identity (workflow_id/run_id/pre-screen
    verdict/sync-mode) is carried in a per-run ``MiddlewareTurnState`` object
    threaded through LangGraph's own GRAPH STATE (``OpenBoxAgentState.ob_turn``)
    — NOT on ``self`` and NOT in a ``ContextVar`` — so concurrent
    ``agent.ainvoke`` calls on one middleware instance never cross-contaminate
    turn state (M17). See ``OpenBoxAgentState`` above for why a ContextVar does
    not work through the real graph dispatch.
    """

    state_schema = OpenBoxAgentState

    def __init__(self, options: OpenBoxLangChainMiddlewareOptions | None = None) -> None:
        super().__init__()
        self._options = options or OpenBoxLangChainMiddlewareOptions()
        self._runtime = build_middleware_runtime(self._options)
        self._workflow_type = self._options.agent_name or "LangChainRun"

    def close(self) -> None:
        """Release runtime resources (HTTP clients, instrumentation). Idempotent."""
        self._runtime.close()

    async def aclose(self) -> None:
        """Async :meth:`close`."""
        await self._runtime.aclose()

    # ─── Sync hooks (for invoke/stream) ────────────────────────────

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        from openbox_langchain.middleware_hook_handlers import handle_before_agent_sync

        turn_state = MiddlewareTurnState.new(sync_mode=True, thread_id=extract_thread_id(runtime))
        turn_state = handle_before_agent_sync(self, turn_state, state, runtime)
        return {"ob_turn": turn_state}

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        from openbox_langchain.middleware_hook_handlers import handle_after_agent_sync

        turn_state = require_turn_state(state)
        return handle_after_agent_sync(self, turn_state, state, runtime)

    def wrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        from openbox_langchain.middleware_model_call import handle_wrap_model_call_sync

        turn_state = require_turn_state_from_request(request)
        return handle_wrap_model_call_sync(self, turn_state, request, handler)

    def wrap_tool_call(  # type: ignore[override]
        self, request: ToolCallRequestLike, handler: Any
    ) -> Any:
        # Declared as ToolCallRequestLike (a local Protocol, not the real
        # langgraph.prebuilt.tool_node.ToolCallRequest) to avoid a langgraph
        # import in this package's boundary (P2-new-C) — mypy flags this as an
        # LSP-narrowing override, but the real request object LangChain passes
        # in still satisfies the Protocol structurally at runtime.
        from openbox_langchain.middleware_tool_hook import handle_wrap_tool_call_sync

        turn_state = require_turn_state_from_request(request)
        return handle_wrap_tool_call_sync(self, turn_state, request, handler)

    # ─── Async hooks (for ainvoke/astream) ─────────────────────────

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        from openbox_langchain.middleware_hook_handlers import handle_before_agent

        turn_state = MiddlewareTurnState.new(sync_mode=False, thread_id=extract_thread_id(runtime))
        turn_state = await handle_before_agent(self, turn_state, state, runtime)
        return {"ob_turn": turn_state}

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        from openbox_langchain.middleware_hook_handlers import handle_after_agent

        turn_state = require_turn_state(state)
        return await handle_after_agent(self, turn_state, state, runtime)

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        from openbox_langchain.middleware_model_call import handle_wrap_model_call

        turn_state = require_turn_state_from_request(request)
        return await handle_wrap_model_call(self, turn_state, request, handler)

    async def awrap_tool_call(  # type: ignore[override]
        self, request: ToolCallRequestLike, handler: Any
    ) -> Any:
        # See wrap_tool_call above — same ToolCallRequestLike Protocol rationale.
        from openbox_langchain.middleware_tool_hook import handle_wrap_tool_call

        turn_state = require_turn_state_from_request(request)
        return await handle_wrap_tool_call(self, turn_state, request, handler)
