"""Configuration management for HypoLab."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


@dataclass
class HypoLabConfig:
    """Global configuration for HypoLab analysis runs."""

    model: str = field(
        default_factory=lambda: os.getenv("HYPOLAB_MODEL", "gemini/gemini-2.5-flash")
    )
    alpha: float = field(
        default_factory=lambda: float(os.getenv("HYPOLAB_ALPHA", "0.05"))
    )
    min_effect_size: float = field(
        default_factory=lambda: float(os.getenv("HYPOLAB_MIN_EFFECT_SIZE", "0.10"))
    )
    max_hypotheses: int = field(
        default_factory=lambda: int(os.getenv("HYPOLAB_MAX_HYPOTHESES", "6"))
    )
    sandbox_timeout_sec: int = field(
        default_factory=lambda: int(os.getenv("HYPOLAB_SANDBOX_TIMEOUT", "15"))
    )
    max_sandbox_retries: int = 2
    temperature: float = 0.2
    api_key: str | None = None
    api_base: str | None = field(
        default_factory=lambda: os.getenv("OLLAMA_API_BASE", None)
    )
    seed: int = 42
