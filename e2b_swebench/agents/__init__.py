"""Coding agents that generate SWE-bench predictions in E2B sandboxes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import muse_spark
from .muse_spark import (
    GenerationResult,
    MissingTemplates,
    ModelUnavailable,
    SandboxWorkspace,
    check_templates,
    empty_agent_prediction,
    generate_prediction,
    is_permanent_model_error,
    run_agent,
)


@dataclass(frozen=True)
class AgentSpec:
    """Functions and metadata needed by the prediction runner."""

    name: str
    description: str
    default_model: str
    create_client: Callable[..., Any]
    check_model: Callable[..., None]
    generate_prediction: Callable[..., tuple[dict[str, Any], GenerationResult]]
    empty_prediction: Callable[[str, str], dict[str, Any]]
    requires_env: tuple[str, ...] = ()


REGISTRY: dict[str, AgentSpec] = {}


def register(spec: AgentSpec) -> AgentSpec:
    if spec.name in REGISTRY:
        raise ValueError(f"agent {spec.name!r} is already registered")
    REGISTRY[spec.name] = spec
    return spec


def agent_names() -> list[str]:
    return sorted(REGISTRY)


def get_agent(name: str) -> AgentSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown agent {name!r}; available: {', '.join(agent_names())}"
        ) from None


register(
    AgentSpec(
        name="muse-spark",
        description="Meta's Muse Spark over the Meta Model API (OpenAI-compatible)",
        default_model=muse_spark.DEFAULT_MODEL,
        create_client=muse_spark.create_model_client,
        check_model=muse_spark.check_model,
        generate_prediction=muse_spark.generate_prediction,
        empty_prediction=muse_spark.empty_agent_prediction,
        requires_env=("META_API_KEY",),
    )
)

__all__ = [
    "REGISTRY",
    "AgentSpec",
    "GenerationResult",
    "MissingTemplates",
    "ModelUnavailable",
    "SandboxWorkspace",
    "agent_names",
    "check_templates",
    "empty_agent_prediction",
    "generate_prediction",
    "get_agent",
    "is_permanent_model_error",
    "register",
    "run_agent",
]
