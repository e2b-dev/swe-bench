"""Run the SWE-bench benchmark on E2B sandboxes (Strategy A: one E2B template
per instance, built FROM the prebuilt swebench/sweb.eval.x86_64.* Docker image).

The swebench package supplies all the grading logic (Docker-free); E2B only
replaces the per-instance *execution environment*.
"""

from .dataset import (
    load_instances,
    parse_tests,
    gold_prediction,
    empty_prediction,
    select_per_repo,
)
from .templates import template_name, instance_image, ensure_template, build_many
from .driver import run_instance, run_instance_async
from .runner import run_many
from .logs import quiet_logs

__all__ = [
    "load_instances",
    "parse_tests",
    "gold_prediction",
    "empty_prediction",
    "select_per_repo",
    "template_name",
    "instance_image",
    "ensure_template",
    "build_many",
    "run_instance",
    "run_instance_async",
    "run_many",
    "quiet_logs",
]
