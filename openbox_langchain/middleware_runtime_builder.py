"""Builds the ``OpenBoxRuntime`` + adapter for one ``OpenBoxLangChainMiddleware``
instance. Split out of ``middleware.py`` to stay under 200 lines per file.

Each middleware instance gets its OWN runtime with a private ``ContextStore``
(never a module-level singleton) — multiple concurrent agent instances must
not share correlation state or instrumentation installs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openbox_core.approvals import ApprovalPoller
from openbox_core.client import EvaluationClient
from openbox_core.config import OpenBoxConfig
from openbox_core.context import ContextStore
from openbox_core.runtime import OpenBoxRuntime

from openbox_langchain.middleware_approval_adapter import LangChainCoreAdapter
from openbox_langchain.sdk_metadata import SDK_ENGINE, SDK_LANGUAGE, SDK_PACKAGE_VERSION

if TYPE_CHECKING:
    from openbox_langchain.middleware import OpenBoxLangChainMiddlewareOptions

__all__ = ["build_middleware_runtime"]


def build_middleware_runtime(options: OpenBoxLangChainMiddlewareOptions) -> OpenBoxRuntime:
    """Resolve config, build the (optional) HITL approval poller, and construct
    a fully-installed ``OpenBoxRuntime`` for one middleware instance.

    Config precedence: explicit ``options`` fields > ``OPENBOX_LANGCHAIN_*``
    env > global ``OPENBOX_*`` env > defaults (``OpenBoxConfig.resolve``).
    """
    config = OpenBoxConfig.resolve(
        env_prefix="OPENBOX_LANGCHAIN",
        api_url=options.api_url,
        api_key=options.api_key,
        on_api_error=options.on_api_error,
        timeout_seconds=options.governance_timeout,
        agent_name=options.agent_name,
        agent_did=options.agent_did,
        agent_private_key=options.agent_private_key,
        workload_private_key=options.workload_private_key,
        sdk_version=SDK_PACKAGE_VERSION,
        sdk_engine=SDK_ENGINE,
        sdk_language=SDK_LANGUAGE,
    )

    approval_poller: ApprovalPoller | None = None
    if config.hitl.enabled:
        # Drives the approval wait on both paths: awaited on the async hooks,
        # blocking (LangChainCoreAdapter.wait_for_approval_sync) on the sync ones.
        approval_client = EvaluationClient(
            config.api_url,
            config.api_key,
            timeout_seconds=config.timeout_seconds,
            on_api_error=config.on_api_error,
            identity=config.load_identity(),
            workload_private_key=config.keycloak_workload_private_key(),
            sdk_version=config.sdk_version,
            sdk_engine=config.sdk_engine,
            sdk_language=config.sdk_language,
        )
        approval_poller = ApprovalPoller(
            approval_client,
            poll_interval_seconds=options.approval_poll_interval_seconds,
            max_wait_seconds=options.approval_max_wait_seconds,
        )

    adapter = LangChainCoreAdapter(approval_poller=approval_poller)
    runtime = OpenBoxRuntime(config, adapter, context_store=ContextStore())
    runtime.install_instrumentation()
    return runtime
