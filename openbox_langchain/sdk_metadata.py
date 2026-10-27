"""Static SDK identity used for OpenBox request headers."""

from __future__ import annotations

SDK_ENGINE = "langchain"
SDK_LANGUAGE = "python"

# Static on purpose: outbound headers should not call importlib.metadata at
# request time. Keep in sync with pyproject.toml on release.
SDK_PACKAGE_VERSION = "1.1.0"

__all__ = ["SDK_ENGINE", "SDK_LANGUAGE", "SDK_PACKAGE_VERSION"]
