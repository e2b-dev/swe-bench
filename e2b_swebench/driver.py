"""The execution driver: the SWE-bench Docker harness's per-instance flow,
reimplemented on E2B primitives.

Native harness            ->  E2B
------------------------------------------------------------
build_container           ->  Sandbox.create(template)        (template == prebuilt image)
copy_to_container(patch)  ->  sandbox.files.write(...)
container.exec_run(apply) ->  sandbox.commands.run("git apply ...")
container.exec_run(eval)  ->  sandbox.commands.run("/bin/bash /eval.sh ...")
parse logs / grade        ->  swebench.harness.grading.get_eval_report  (unchanged, pure Python)
"""

import asyncio
import os
import random
import re
import tempfile
import time

from e2b import CommandExitException, RateLimitException, Sandbox
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

from .config import ARCH, CMD_TIMEOUT, NAMESPACE, SANDBOX_TIMEOUT

# Sandbox creation is the only call that can hit the account's concurrent-sandbox
# cap (RateLimitException / 429). Our semaphore keeps us under the cap, but build
# provisioning or stray sandboxes can briefly fill it — so back off and retry
# rather than failing the instance. Backoff: ~5,10,20,40,60,60,60s (+jitter).
_RL_RETRIES = 8


def _rl_backoff(attempt: int) -> float:
    return min(5 * (2 ** attempt), 60) + random.uniform(0, 2)

# Same order/commands the native harness uses to apply a prediction patch.
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

# eval.sh is run as root, redirecting combined stdout+stderr to a file we read
# back. We MUST capture both streams: some repos' test runners (e.g. Django)
# print results to stderr. Reading from a file also survives a non-zero exit.
_EVAL_CMD = "/bin/bash /eval.sh > /tmp/test_output.txt 2>&1"


def _detect_collection_error(output: str) -> bool:
    """A pytest *collection* error (the test module fails to import) — almost always
    an environment/image issue, not a model/template defect. Canonical example:
    SWE-bench ':latest' images now ship setuptools 68, whose vendored distutils
    emits a DeprecationWarning that astropy's (programmatic) warnings-as-errors
    turns into a collection error. We surface these transparently rather than
    suppressing them (env/CLI warning filters can't override astropy's config, and
    patching it would diverge from the image more than the warning itself)."""
    o = output.lower()
    return "errors during collection" in o or "error collecting" in o


# A drifted-dependency warning promoted to a *test* error (pytest 'E <Warning>:' line) —
# e.g. astropy's warnings-as-errors tripping on pytest's nose-deprecation. Same upstream
# image-drift family as a collection error, just at test-run time instead of import time.
_WARNING_E_LINE = re.compile(
    r"^\s*E\s+.*?(DeprecationWarning|PendingDeprecationWarning|FutureWarning|"
    r"PytestRemovedIn\d+Warning|PytestDeprecationWarning|PytestUnraisableExceptionWarning)",
    re.MULTILINE,
)


def _detect_warning_error(output: str) -> bool:
    return bool(_WARNING_E_LINE.search(output)) or "is using nose-specific method" in output


def _create_sandbox(template: str, timeout: int):
    for attempt in range(_RL_RETRIES):
        try:
            return Sandbox.create(template, timeout=timeout)
        except RateLimitException:
            if attempt == _RL_RETRIES - 1:
                raise
            time.sleep(_rl_backoff(attempt))


async def _create_sandbox_async(template: str, timeout: int):
    from e2b import AsyncSandbox

    for attempt in range(_RL_RETRIES):
        try:
            return await AsyncSandbox.create(template, timeout=timeout)
        except RateLimitException:
            if attempt == _RL_RETRIES - 1:
                raise
            await asyncio.sleep(_rl_backoff(attempt))


def _grade(test_spec, prediction: dict, output: str) -> dict:
    """Feed captured output to swebench's grader and return this instance's verdict."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(output)
        log_path = f.name
    try:
        report = get_eval_report(test_spec, prediction, log_path, True)
    finally:
        os.unlink(log_path)
    return report[prediction["instance_id"]]


def run_instance(
    instance: dict,
    prediction: dict,
    template: str,
    sandbox_timeout: int = SANDBOX_TIMEOUT,
    cmd_timeout: int = CMD_TIMEOUT,
    keep_output: bool = False,
) -> dict:
    """Evaluate one prediction in a fresh sandbox. Returns the swebench verdict
    dict ({'resolved': bool, 'patch_successfully_applied': bool, ...})."""
    ts = make_test_spec(instance, namespace=NAMESPACE, arch=ARCH)
    patch = prediction.get("model_patch") or ""
    sbx = _create_sandbox(template, sandbox_timeout)
    try:
        # 1. apply the prediction patch (empty patch = no-op, still graded)
        applied = not patch.strip()
        if patch.strip():
            sbx.files.write("/tmp/patch.diff", patch, user="root")
            for cmd in GIT_APPLY_CMDS:
                try:
                    res = sbx.commands.run(
                        f"{cmd} /tmp/patch.diff", cwd="/testbed", user="root", timeout=300
                    )
                    if res.exit_code == 0:
                        applied = True
                        break
                except CommandExitException:
                    continue
        if not applied:
            return {
                "instance_id": ts.instance_id,
                "resolved": False,
                "patch_successfully_applied": False,
                "error": "patch_apply_failed",
            }

        # 2. run eval.sh (it applies the gold test_patch + runs the repo's tests)
        sbx.files.write("/eval.sh", ts.eval_script, user="root")
        try:
            sbx.commands.run(_EVAL_CMD, cwd="/testbed", user="root", timeout=cmd_timeout)
        except CommandExitException:
            pass  # non-zero exit is normal when tests fail; we grade from the log
        output = sbx.files.read("/tmp/test_output.txt", user="root")

        # 3. grade
        verdict = _grade(ts, prediction, output)
        verdict["collection_error"] = _detect_collection_error(output)
        verdict["warning_error"] = _detect_warning_error(output)
        if keep_output:
            verdict["_output"] = output
        return verdict
    finally:
        sbx.kill()


async def run_instance_async(
    instance: dict,
    prediction: dict,
    template: str,
    sandbox_timeout: int = SANDBOX_TIMEOUT,
    cmd_timeout: int = CMD_TIMEOUT,
    keep_output: bool = False,
) -> dict:
    """Async mirror of run_instance, for concurrent runs via run_many()."""
    ts = make_test_spec(instance, namespace=NAMESPACE, arch=ARCH)
    patch = prediction.get("model_patch") or ""
    sbx = await _create_sandbox_async(template, sandbox_timeout)
    try:
        applied = not patch.strip()
        if patch.strip():
            await sbx.files.write("/tmp/patch.diff", patch, user="root")
            for cmd in GIT_APPLY_CMDS:
                try:
                    res = await sbx.commands.run(
                        f"{cmd} /tmp/patch.diff", cwd="/testbed", user="root", timeout=300
                    )
                    if res.exit_code == 0:
                        applied = True
                        break
                except CommandExitException:
                    continue
        if not applied:
            return {
                "instance_id": ts.instance_id,
                "resolved": False,
                "patch_successfully_applied": False,
                "error": "patch_apply_failed",
            }

        await sbx.files.write("/eval.sh", ts.eval_script, user="root")
        try:
            await sbx.commands.run(_EVAL_CMD, cwd="/testbed", user="root", timeout=cmd_timeout)
        except CommandExitException:
            pass
        output = await sbx.files.read("/tmp/test_output.txt", user="root")

        verdict = _grade(ts, prediction, output)
        verdict["collection_error"] = _detect_collection_error(output)
        verdict["warning_error"] = _detect_warning_error(output)
        if keep_output:
            verdict["_output"] = output
        return verdict
    finally:
        await sbx.kill()
