# OpenBox LangChain SDK — Python

LangChain-Core adapter for OpenBox governance. Provides callback handlers that emit tool and LLM lifecycle events to OpenBox Core's policy engine, enabling real-time governance, guardrails, HITL approval flows, and base-SDK hook governance (HTTP/DB/File I/O) for LangChain agents.

## Installation

```bash
pip install openbox-langchain-sdk-python
```

## Quick Start

```python
from langchain.agents import create_agent
from openbox_langchain import create_openbox_langchain_middleware

# 1. Create middleware
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    agent_did="did:aip:...",
    agent_private_key="...",
    agent_name="MyAgent",
)

# 2. Create agent with middleware
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[middleware],
)

# 3. Invoke — governance applied automatically
result = agent.invoke({"messages": [("user", "your query")]})
```

## How It Works

Two-layer governance architecture:

| Layer | Mechanism | Governs |
|-------|-----------|---------|
| 1 | LangChain-Core Callbacks | Tool and LLM lifecycle emission via `ActivityBridge` |
| 2 | Base SDK Hook Governance | HTTP requests, DB queries, file I/O at process boundary |

**Core components:**
- `ActivityBridge` — Ownership channel for tool and LLM events; prevents duplicate governance evaluation
- `OpenBoxLangChainCoreAsyncCallbackHandler` / `...SyncCallbackHandler` — LangChain-Core callbacks emitting tool and LLM lifecycle events to OpenBox Core
- AgentMiddleware (for `create_agent`) — Wraps model and tool calls with additional governance context

## Configuration

```python
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",  # OpenBox Core URL
    api_key="obx_live_...",              # API key (obx_live_* or obx_test_*)
    agent_did="did:aip:...",             # Required by default; can use OPENBOX_AGENT_DID
    agent_private_key="...",             # Required by default; can use OPENBOX_AGENT_PRIVATE_KEY
    workload_private_key="...",          # IAM v3 agents instead of the DID pair
    agent_name="MyAgent",                # Agent name (from dashboard)
    governance_timeout=30.0,             # HTTP timeout in seconds
    validate=True,                       # Validate API key on startup
    session_id="session-123",            # Optional session tracking
    tool_type_map={                      # Optional tool classification
        "search_web": "http",
        "query_db": "database",
    },
)
```

## Agent Identity and DID Signing

OpenBox issues each registered agent a decentralized identifier (DID) and
private key. DID signing is enabled by default for newly registered agents. The
OpenBox UI returns both values when the agent is created. Pass them to the
middleware so governance events are signed and attributable to that agent.

You can provide them directly:

```python
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    agent_did="did:aip:...",
    agent_private_key="...",
)
```

Or set them through the environment:

```bash
export OPENBOX_AGENT_DID="did:aip:..."
export OPENBOX_AGENT_PRIVATE_KEY="..."
```

If DID signing is explicitly disabled for the agent in OpenBox, these values can
be omitted. Otherwise, provide both values together.

### IAM v3 workload identity

An agent registered with a Keycloak workload identity authenticates with its
PKCS8 PEM private key (the key contents, not a path). The base SDK handles the
bootstrap, token exchange and refresh against Core's `/api/v3/*` endpoints:

```python
middleware = create_openbox_langchain_middleware(
    api_url="https://core.openbox.ai",
    api_key="obx_live_...",
    workload_private_key=os.environ["OPENBOX_WORKLOAD_PRIVATE_KEY"],
)
```

When the argument is omitted, `OPENBOX_LANGCHAIN_WORKLOAD_PRIVATE_KEY` and then
`OPENBOX_WORKLOAD_PRIVATE_KEY` are used. The key is also used for the startup
validation and for approval polling.

## OpenRouter routing provenance

When the model is served through OpenRouter (for example
`ChatOpenAI(base_url="https://openrouter.ai/api/v1", ...)`), each model call
also reports which upstream provider served it, at what cost, and whether the
requested model and routing were honoured. These feed the agent's Provenance tab
in OpenBox. No configuration is needed; see the base SDK's README for details
and `OPENBOX_OPENROUTER_PROVENANCE=false` to turn it off.

## Supported Agent Types

- `create_agent(model, tools, middleware=[...])` — recommended
- Any LangChain agent builder that accepts `middleware`

## Verdict Enforcement

5-tier verdict system:
- **ALLOW** — Request permitted
- **CONSTRAIN** — Request constrained (e.g., rate limit)
- **REQUIRE_APPROVAL** — Human approval required. The async entry points
  (`agent.ainvoke` / `astream`) wait for the decision; the sync ones fail safe
  and do not run the operation.
- **BLOCK** — Request blocked with error
- **HALT** — Entire workflow halted (unrecoverable error)

## Requirements

- Python 3.11+
- openbox-sdk-python[http] >= 1.5.0
- langchain-core >= 1.3.3
- LangChain >= 1.0.0 (required only for `[agent]` extra, which enables `create_agent` middleware)

## API Reference

**LangChain-Core callbacks:**
- `OpenBoxLangChainCoreAsyncCallbackHandler` — Async callback handler for tool/LLM lifecycle
- `OpenBoxLangChainCoreSyncCallbackHandler` — Sync callback handler for tool/LLM lifecycle
- `ActivityBridge` — Ownership channel for lifecycle event deduplication

**AgentMiddleware (optional):**
- `create_openbox_langchain_middleware()` — Creates configured middleware for `create_agent`

See `openbox_langchain.__init__.py` for full API export list.

## License

MIT
