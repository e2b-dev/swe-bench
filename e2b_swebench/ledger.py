"""Persistent build+verify ledger for the batched full-dataset run.

Tracks, per instance, whether its template built and whether the gold patch
verifies. Saved to disk after every batch so a long run can be stopped and
resumed across sessions.

verify categories:
  pass              - gold resolved; template is good
  grader_artifact   - a specifically verified test-ID mismatch in canonical grading
  ordering_artifact - a specifically verified full-suite order-pollution failure
  collection_error  - a test module failed to collect
  warning_error     - a dependency warning was promoted to a test error
  fail              - gold did not apply, or a FAIL_TO_PASS test genuinely failed -> investigate
  error             - sandbox/transient error -> retried on the next run
Only pass and the two evidence-backed, instance-specific artifacts are DONE.
All other non-resolutions are reprocessed on resume.
"""

import datetime
import json
import os
from collections import Counter
from collections.abc import Mapping, Sized

DONE = ("pass", "grader_artifact", "ordering_artifact")
_IDENTITY_FIELDS = ("template", "content_key", "source_image")

# These exceptions were reproduced under the final evaluator and independently
# audited. Do not generalize all PASS_TO_PASS-only failures into artifacts: a
# newly observed regression must remain a failure until it is investigated.
_KNOWN_GOLD_ARTIFACTS = {
    "astropy__astropy-7606": (
        "grader_artifact",
        "pytest emits test_compose_roundtrip[unit0], dataset expects []",
        {
            "astropy/units/tests/test_units.py::test_compose_roundtrip[]",
        },
    ),
    "django__django-10097": (
        "ordering_artifact",
        "five generic_inline_admin tests pass alone but fail after the full suite",
        {
            "test_add (generic_inline_admin.tests.GenericInlineAdminWithUniqueTogetherTest)",
            "test_delete (generic_inline_admin.tests.GenericInlineAdminWithUniqueTogetherTest)",
            "test_no_param (generic_inline_admin.tests.GenericInlineAdminParametersTest)",
            "test_basic_add_GET (generic_inline_admin.tests.GenericAdminViewTest)",
            "test_basic_edit_GET (generic_inline_admin.tests.GenericAdminViewTest)",
        },
    ),
}


def _safe_failure_count(failures: object) -> int:
    return len(failures) if isinstance(failures, Sized) else 0


def _is_failure_list(failures: object) -> bool:
    return isinstance(failures, list) and all(
        isinstance(failure, str) for failure in failures
    )


def categorize_verdict(v: dict) -> tuple[str, dict]:
    """Map a driver verdict to a (category, detail) pair for the ledger."""
    ts = v.get("tests_status") or {}
    f2p = ts.get("FAIL_TO_PASS", {}) or {}
    p2p = ts.get("PASS_TO_PASS", {}) or {}
    actual_f2p_failures = f2p.get("failure", [])
    actual_p2p_failures = p2p.get("failure", [])
    detail = {
        "resolved": bool(v.get("resolved")),
        "patch_applied": bool(v.get("patch_successfully_applied")),
        "f2p_fail": _safe_failure_count(actual_f2p_failures),
        "p2p_fail": _safe_failure_count(actual_p2p_failures),
        "error": v.get("error"),
        "collection_error": bool(v.get("collection_error")),
        "warning_error": bool(v.get("warning_error")),
        "resource_exhausted": bool(v.get("resource_exhausted")),
        "runtime_seconds": v.get("runtime_seconds"),
        "metrics": v.get("metrics"),
    }
    if v.get("error"):
        return "error", detail
    if v.get("resolved"):
        return "pass", detail
    # a test module failed to import/collect -> upstream environment/image issue
    if v.get("collection_error"):
        return "collection_error", detail
    # a drifted-dependency warning was promoted to a test error -> same env family
    if v.get("warning_error"):
        return "warning_error", detail
    known_artifact = _KNOWN_GOLD_ARTIFACTS.get(v.get("instance_id"))
    if known_artifact:
        category, reason, expected_p2p_failures = known_artifact
        if (
            v.get("patch_successfully_applied") is True
            and _is_failure_list(actual_f2p_failures)
            and _is_failure_list(actual_p2p_failures)
            and detail["f2p_fail"] == 0
            and len(actual_p2p_failures) == len(expected_p2p_failures)
            and set(actual_p2p_failures) == expected_p2p_failures
        ):
            detail["artifact_reason"] = reason
            return category, detail
    return "fail", detail


class Ledger:
    def __init__(self, path: str):
        self.path = path
        self.data: dict = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def get(self, iid: str) -> dict:
        return self.data.get(iid, {})

    def verify_status(self, iid: str):
        return self.get(iid).get("verify")

    def is_done(
        self,
        iid: str,
        cpu_count: int | None = None,
        memory_mb: int | None = None,
        *,
        identity: Mapping[str, str] | None = None,
    ) -> bool:
        record = self.get(iid)
        if identity is None or any(field not in identity for field in _IDENTITY_FIELDS):
            return False
        if any(record.get(field) != identity[field] for field in _IDENTITY_FIELDS):
            return False
        if cpu_count is not None and record.get("cpu_count") != cpu_count:
            return False
        if memory_mb is not None and record.get("memory_mb") != memory_mb:
            return False
        return record.get("verify") in DONE

    def update(self, iid: str, **fields) -> None:
        rec = self.data.setdefault(iid, {"instance_id": iid})
        rec.update(fields)
        rec["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )

    def save(self) -> None:
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)  # atomic

    def instances_with_verify(self, *statuses) -> list:
        return sorted(i for i, r in self.data.items() if r.get("verify") in statuses)

    def summary(self) -> dict:
        return {
            "total": len(self.data),
            "build": dict(
                Counter(r.get("build") or "pending" for r in self.data.values())
            ),
            "verify": dict(
                Counter(r.get("verify") or "pending" for r in self.data.values())
            ),
        }
