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
from .runner import run_many
from .templates import (
    build_many,
    ensure_template,
    instance_image,
    template_name,
    template_ready,
)

__all__ = [
    "build_many",
    "empty_prediction",
    "ensure_template",
    "gold_prediction",
    "instance_image",
    "load_instances",
    "parse_tests",
    "quiet_logs",
    "run_instance",
    "run_instance_async",
    "run_many",
    "select_per_repo",
    "summarize_metrics",
    "template_name",
    "template_ready",
]
