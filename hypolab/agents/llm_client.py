"""LiteLLM wrapper for structured JSON completions and availability checks."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from hypolab.config import HypoLabConfig

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _clean_key(val: str | None) -> str:
    if not val:
        return ""
    cleaned = val.strip().strip('"').strip("'")
    if cleaned.startswith("your_") or cleaned.endswith("_here"):
        return ""
    return cleaned


def is_llm_configured(config: HypoLabConfig) -> bool:
    """Return True if an API key or local Ollama model is configured."""
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    model_lower = config.model.lower()
    if model_lower.startswith("ollama/") or model_lower.startswith("ollama_chat/"):
        return True
    if _clean_key(config.api_key):
        return True
    env_keys = (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
    )
    for k in env_keys:
        if _clean_key(os.getenv(k)):
            return True
    return False


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    config: HypoLabConfig,
) -> dict[str, Any] | list[Any]:
    """Call LiteLLM and parse the JSON object or array from the response."""
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True

    model_to_use = config.model.strip()
    kwargs: dict[str, Any] = {
        "model": model_to_use,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
    }

    explicit_key = _clean_key(config.api_key)
    if explicit_key:
        kwargs["api_key"] = explicit_key
    elif model_to_use.lower().startswith("gemini/"):
        env_gemini = _clean_key(os.getenv("GEMINI_API_KEY"))
        if env_gemini:
            kwargs["api_key"] = env_gemini

    if config.api_base and model_to_use.lower().startswith("ollama"):
        kwargs["api_base"] = config.api_base

    try:
        response = litellm.completion(**kwargs)
    except Exception as exc:
        err_str = str(exc)
        if model_to_use.lower().startswith("gemini/") and (
            "404" in err_str or "503" in err_str or "no longer available" in err_str or "UNAVAILABLE" in err_str
        ):
            fallback_models = [
                "gemini/gemini-3.8-flash",
                "gemini/gemini-2.5-flash-lite",
                "gemini/gemini-3.1-flash-lite",
            ]
            last_exc = exc
            response = None
            for fb_model in fallback_models:
                if fb_model.lower() == model_to_use.lower():
                    continue
                try:
                    kwargs["model"] = fb_model
                    response = litellm.completion(**kwargs)
                    config.model = fb_model
                    break
                except Exception as fb_exc:
                    last_exc = fb_exc
            if response is None:
                raise last_exc
        else:
            raise

    content = response.choices[0].message.content or ""
    return extract_json_from_text(content)


def _loads_lenient(raw_json: str) -> dict[str, Any] | list[Any]:
    """Parse JSON while repairing common LLM escaping issues (e.g., LaTeX \\alpha, \\le, \\beta)."""
    try:
        return json.loads(raw_json, strict=False)
    except json.JSONDecodeError:
        def _fix_escape(match: re.Match[str]) -> str:
            token = match.group(0)
            if token in ('\\"', "\\\\", "\\/", "\\n", "\\r", "\\t") or token.startswith("\\u"):
                return token
            return "\\" + token

        repaired = re.sub(r"\\(?:u[0-9a-fA-F]{4}|.)", _fix_escape, raw_json)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return json.loads(repaired, strict=False)


def extract_json_from_text(text: str) -> dict[str, Any] | list[Any]:
    """Extract JSON block from raw LLM markdown or plain text output."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return _loads_lenient(cleaned)
    except json.JSONDecodeError:
        start_brace = cleaned.find("{")
        start_bracket = cleaned.find("[")
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            end_brace = cleaned.rfind("}")
            if end_brace > start_brace:
                return _loads_lenient(cleaned[start_brace : end_brace + 1])
        if start_bracket != -1:
            end_bracket = cleaned.rfind("]")
            if end_bracket > start_bracket:
                return _loads_lenient(cleaned[start_bracket : end_bracket + 1])
        raise
