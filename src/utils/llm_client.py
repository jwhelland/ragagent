"""LLM client creation factory.

This module provides a centralized way to create LLM clients (OpenAI, etc.)
to ensure consistent configuration of API keys, base URLs, and timeouts.
"""

import os
from typing import Any, Optional

from loguru import logger
from openai import OpenAI


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> OpenAI:
    """Create and configure an OpenAI client.

    Args:
        api_key: The API key. If None, tries env var or defaults.
        base_url: The base URL. If None, tries env var.
        timeout: Request timeout in seconds.
        max_retries: Number of retries.
        **kwargs: Additional arguments to pass to the OpenAI constructor.

    Returns:
        Configured OpenAI client.
    """
    # Resolve API Key
    # 1. Argument
    # 2. Env var OPENAI_API_KEY (handled by library, but we can log checks)
    final_api_key = api_key or os.getenv("OPENAI_API_KEY")

    # Resolve Base URL
    final_base_url = base_url or os.getenv("OPENAI_BASE_URL")

    # Log configuration (masking key)
    masked_key = (
        f"{final_api_key[:4]}...{final_api_key[-4:]}"
        if final_api_key and len(final_api_key) > 8
        else "None"
    )
    logger.debug(
        f"Creating OpenAI client: base_url={final_base_url}, "
        f"api_key={masked_key}, timeout={timeout}"
    )

    return OpenAI(
        api_key=final_api_key,
        base_url=final_base_url,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )
