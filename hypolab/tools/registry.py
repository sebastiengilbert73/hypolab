"""Tool registry and metadata for structured statistical tests."""

from __future__ import annotations

from typing import Any, Callable

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(name: str, description: str, parameters: dict[str, Any]) -> Callable:
    """Decorator to register a statistical tool with its schema."""
    def decorator(func: Callable) -> Callable:
        func.__test__ = False  # Prevent pytest from collecting imported tool functions as tests
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "callable": func,
        }
        return func
    return decorator
