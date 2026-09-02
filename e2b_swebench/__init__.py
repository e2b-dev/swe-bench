"""Run the SWE-bench benchmark on E2B sandboxes (Strategy A: one E2B template
per instance, built FROM the prebuilt swebench/sweb.eval.x86_64.* Docker image).

The swebench package supplies all the grading logic (Docker-free); E2B only
replaces the per-instance *execution environment*.
"""

from .dataset import (
    empty_prediction,
    gold_prediction,
    load_instances,
    parse_tests,
    select_per_repo,
)
from .driver import run_instance, run_instance_async
from .logs import quiet_logs
from .metrics import summarize_metrics
from .runner import evaluation_identity, resolve_template_specs, run_many
from .templates import (
    ImmutableTemplateIdentityRequired,
    TemplateSpec,
    build_many,
    content_key,
    ensure_template,
    ensure_template_from_spec,
    immutable_template_name,
    instance_image,
    resolve_image,
    resolve_template_spec,
    resolve_template_specs_sync,
    template_identity,
    template_name,
    template_name_from_spec,
    template_ready,
    template_spec,
)

__all__ = [
    "ImmutableTemplateIdentityRequired",
    "TemplateSpec",
    "build_many",
    "content_key",
    "empty_prediction",
    "ensure_template",
    "ensure_template_from_spec",
    "evaluation_identity",
    "gold_prediction",
    "immutable_template_name",
    "instance_image",
    "load_instances",
    "parse_tests",
    "quiet_logs",
    "resolve_image",
    "resolve_template_spec",
    "resolve_template_specs",
    "resolve_template_specs_sync",
    "run_instance",
    "run_instance_async",
    "run_many",
    "select_per_repo",
    "summarize_metrics",
    "template_identity",
    "template_name",
    "template_name_from_spec",
    "template_ready",
    "template_spec",
]
