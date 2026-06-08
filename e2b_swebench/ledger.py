"""Persistent build+verify ledger for the batched full-dataset run.

Tracks, per instance, whether its template built and whether the gold patch
verifies. Saved to disk after every batch so a long run can be stopped and
resumed across sessions (finish some tonight, continue tomorrow).

verify categories:
  pass              - gold resolved; template is good
  ordering_artifact - gold fix applied & all FAIL_TO_PASS pass, only PASS_TO_PASS
                      regressed (the known non-Docker test-ordering effect; the
                      template itself is fine, see django__django-10097)
  collection_error  - a test module failed to COLLECT (import-time error), almost
                      always upstream image drift (e.g. setuptools 68's distutils
                      DeprecationWarning vs astropy's warnings-as-errors). NOT a
                      template defect, affects the official Docker harness too;
                      see astropy__astropy-8872.
  warning_error     - same image-drift family, but at TEST time: a drifted-dependency
                      warning (e.g. pytest's nose-deprecation) is promoted to an error
                      by the repo's warnings-as-errors config; see astropy__astropy-8707.
  fail              - gold did not apply, or a FAIL_TO_PASS test genuinely failed -> investigate
  error             - sandbox/transient error -> retried on the next run
pass / ordering_artifact / collection_error / warning_error are DONE (skipped on
resume) — they are environment facts, not template defects. fail / error are reprocessed.
"""

import datetime
import json
import os
from collections import Counter

DONE = ("pass", "ordering_artifact", "collection_error", "warning_error")


def categorize_verdict(v: dict) -> tuple[str, dict]:
    """Map a driver verdict to a (category, detail) pair for the ledger."""
    ts = v.get("tests_status") or {}
    f2p = ts.get("FAIL_TO_PASS", {}) or {}
    p2p = ts.get("PASS_TO_PASS", {}) or {}
    detail = {
        "resolved": bool(v.get("resolved")),
        "patch_applied": bool(v.get("patch_successfully_applied")),
        "f2p_fail": len(f2p.get("failure", [])),
        "p2p_fail": len(p2p.get("failure", [])),
        "error": v.get("error"),
        "collection_error": bool(v.get("collection_error")),
        "warning_error": bool(v.get("warning_error")),
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
    # gold fix applied, every FAIL_TO_PASS passes, only PASS_TO_PASS regressed:
    # test-ordering artifact, not a build defect.
    if detail["patch_applied"] and detail["f2p_fail"] == 0 and detail["p2p_fail"] > 0:
        return "ordering_artifact", detail
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

    def is_done(self, iid: str) -> bool:
        return self.verify_status(iid) in DONE

    def update(self, iid: str, **fields) -> None:
        rec = self.data.setdefault(iid, {"instance_id": iid})
        rec.update(fields)
        rec["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")

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
            "build": dict(Counter(r.get("build") or "pending" for r in self.data.values())),
            "verify": dict(Counter(r.get("verify") or "pending" for r in self.data.values())),
        }
