"""Factory for creating configured OpenBoxLangChainMiddleware instances.

Config resolution goes through ``OpenBoxConfig.resolve`` (base SDK) directly —
no ``openbox_langgraph.config.initialize`` call, no process-wide global config.
Precedence: explicit args (``api_url``/``api_key``/...) > ``OPENBOX_LANGCHAIN_*``
env > global ``OPENBOX_*`` env > defaults. ``validate=True`` (default) checks
the API key against the server via the base SDK client before returning.

Usage:
    from openbox_langchain import create_openbox_langchain_middleware
    middleware = create_openbox_langchain_middleware(
        api_url=os.environ["OPENBOX_URL"],
        api_key=os.environ["OPENBOX_API_KEY"],
        agent_name="MyAgent",
    )
    agent = create_agent(model=..., tools=[...], middleware=[middleware])
    result = agent.invoke({"messages": [("user", "Hello")]})
"""

from __future__ import annotations

import dataclasses
from typing import Any

from openbox_core.client import EvaluationClient
from openbox_core.config import OpenBoxConfig

from openbox_langchain.middleware import (
    OpenBoxLangChainMiddleware,
    OpenBoxLangChainMiddlewareOptions,
)
from openbox_langchain.sdk_metadata import SDK_ENGINE, SDK_LANGUAGE, SDK_PACKAGE_VERSION


def create_openbox_langchain_middleware(
    *,
    api_url: str,
    api_key: str,
    agent_name: str | None = None,
    agent_did: str | None = None,
    agent_private_key: str | None = None,
    workload_private_key: str | None = None,
    governance_timeout: float = 30.0,
    validate: bool = True,
    **kwargs: Any,
) -> OpenBoxLangChainMiddleware:
    """Create a configured OpenBoxLangChainMiddleware for create_agent(middleware=[...]).

    Validates the API key against the server (``validate=True``, default)
    before constructing the middleware — a fast-failing config error beats a
    silent misconfiguration surfacing only on the first governed call.

    Args:
        api_url: Base URL of your OpenBox Core instance.
        api_key: API key in ``obx_live_*`` or ``obx_test_*`` format.
        agent_name: Agent name as configured in the dashboard.
        agent_did: Optional OpenBox agent DID. Falls back to ``OPENBOX_AGENT_DID``.
        agent_private_key: Optional raw Ed25519 private key seed. Falls back to
            ``OPENBOX_AGENT_PRIVATE_KEY``.
        workload_private_key: Optional PKCS8 PEM RSA key for an IAM v3 Keycloak
            workload agent (the PEM contents, not a path). Falls back to
            ``OPENBOX_LANGCHAIN_WORKLOAD_PRIVATE_KEY``, then
            ``OPENBOX_WORKLOAD_PRIVATE_KEY``.
        governance_timeout: HTTP timeout in seconds (default 30.0).
        validate: If True, validates the API key against the server on startup.
        **kwargs: Additional kwargs forwarded to OpenBoxLangChainMiddlewareOptions
            (e.g. ``session_id``, ``task_queue``, ``on_api_error``, ``tool_type_map``,
            ``skip_tool_types``, ``send_*_event`` flags).

    Returns:
        A configured ``OpenBoxLangChainMiddleware`` ready for create_agent().
    """
    config = OpenBoxConfig.resolve(
        env_prefix="OPENBOX_LANGCHAIN",
        api_url=api_url,
        api_key=api_key,
        timeout_seconds=governance_timeout,
        agent_name=agent_name,
        agent_did=agent_did,
        agent_private_key=agent_private_key,
        workload_private_key=workload_private_key,
        sdk_version=SDK_PACKAGE_VERSION,
        sdk_engine=SDK_ENGINE,
        sdk_language=SDK_LANGUAGE,
    )

    if validate:
        client = EvaluationClient(
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
        try:
            client.validate_api_key()
        finally:
            client.close()

    valid_fields = {f.name for f in dataclasses.fields(OpenBoxLangChainMiddlewareOptions)}
    options = OpenBoxLangChainMiddlewareOptions(
        api_url=config.api_url,
        api_key=config.api_key,
        agent_name=config.agent_name,
        agent_did=config.agent_did,
        agent_private_key=config.agent_private_key,
        workload_private_key=config.workload_private_key,
        governance_timeout=config.timeout_seconds,
        **{k: v for k, v in kwargs.items() if k in valid_fields},
    )
    return OpenBoxLangChainMiddleware(options)
