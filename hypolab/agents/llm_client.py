"""LiteLLM wrapper for structured JSON completions and availability checks."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from hypolab.config import HypoLabConfig


def is_llm_configured(config: HypoLabConfig) -> bool:
    """Return True if an API key or local Ollama model is configured."""
    model_lower = config.model.lower()
    if model_lower.startswith("ollama/") or model_lower.startswith("ollama_chat/"):
        return True
    if config.api_key:
        return True
    env_keys = (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
    )
    for k in env_keys:
        val = os.getenv(k, "")
        if val and not val.startswith("your_"):
            return True
    return False


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    config: HypoLabConfig,
) -> dict[str, Any] | list[Any]:
    """Call LiteLLM and parse the JSON object or array from the response."""
    import litellm

    litellm.suppress_debug_info = True

    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base and config.model.lower().startswith("ollama"):
        kwargs["api_base"] = config.api_base

    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content or ""
    return extract_json_from_text(content)


def extract_json_from_text(text: str) -> dict[str, Any] | list[Any]:
    """Extract JSON block from raw LLM markdown or plain text output."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try finding outermost {...} or [...]
        start_brace = cleaned.find("{")
        start_bracket = cleaned.find("[")
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            end_brace = cleaned.rfind("}")
            if end_brace > start_brace:
                return json.loads(cleaned[start_brace : end_brace + 1])
        if start_bracket != -1:
            end_bracket = cleaned.rfind("]")
            if end_bracket > start_bracket:
                return json.loads(cleaned[start_bracket : end_bracket + 1])
        raise
