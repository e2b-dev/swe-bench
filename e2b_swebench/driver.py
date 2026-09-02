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
from .metrics import summarize_metrics

# Sandbox creation is the only call that can hit the account's concurrent-sandbox
# cap (RateLimitException / 429). Our semaphore keeps us under the cap, but build
# provisioning or stray sandboxes can briefly fill it — so back off and retry
# rather than failing the instance. Backoff: ~5,10,20,40,60,60,60s (+jitter).
_RL_RETRIES = 8


def _rl_backoff(attempt: int) -> float:
    return min(5 * (2**attempt), 60) + random.uniform(0, 2)


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

_RESOURCE_EXHAUSTED = re.compile(
    r"(^|\n).*\b(killed|out of memory|oom-kill|memoryerror)\b", re.IGNORECASE
)


def _detect_collection_error(output: str) -> bool:
    """Detect pytest import/collection failures for explicit ledger reporting."""
    o = output.lower()
    return "errors during collection" in o or "error collecting" in o


# A dependency warning promoted to a test error (pytest ``E <Warning>:`` line).
_WARNING_E_LINE = re.compile(
    r"^\s*E\s+.*?(DeprecationWarning|PendingDeprecationWarning|FutureWarning|"
    r"PytestRemovedIn\d+Warning|PytestDeprecationWarning|PytestUnraisableExceptionWarning)",
    re.MULTILINE,
)


def _detect_warning_error(output: str) -> bool:
    return (
        bool(_WARNING_E_LINE.search(output))
        or "is using nose-specific method" in output
    )


def _is_whole_repository_restore(command: str, base_commit: str) -> bool:
    """Return whether one generated shell command restores the whole repo."""
    tokens = command.split()
    return tokens in (
        ["git", "checkout", base_commit],
        ["git", "reset", "--hard", base_commit],
    )


def _eval_script_preserving_image_setup(test_spec, instance: dict) -> str:
    """Keep the SWE-bench image's setup commit active during evaluation.

    Per-instance images commit repo-specific test-output and dependency fixes on
    top of ``base_commit``. The generated eval script checks out ``base_commit``
    before and after the test patch, which drops those fixes. That can make a
    passing pytest run ungradeable (compact ``.`` output instead of the test ID),
    as in sphinx-doc__sphinx-8595. Every run uses a fresh sandbox, so cleanup is
    unnecessary. Replacing only complete-repository restore commands preserves
    path-specific held-out-test checkout commands, the candidate working-tree
    patch, and the generated test sequence.
    """
    commands = test_spec.eval_script_list
    serialized_commands = "\n".join(commands) + "\n"
    script = test_spec.eval_script
    if not script.endswith(serialized_commands):
        raise ValueError("unsupported SWE-bench TestSpec eval_script serialization")

    rewritten_commands = [
        (
            ": # preserve SWE-bench image setup commit"
            if _is_whole_repository_restore(command, instance["base_commit"])
            else command
        )
        for command in commands
    ]
    return script[: -len(serialized_commands)] + "\n".join(rewritten_commands) + "\n"


def _attach_execution_stats(verdict: dict, sbx, started_at: float) -> None:
    verdict["runtime_seconds"] = round(time.monotonic() - started_at, 2)
    try:
        verdict["metrics"] = summarize_metrics(sbx.get_metrics())
    except Exception as exc:  # noqa: BLE001 - metrics cannot invalidate a result
        verdict["metrics"] = {"samples": 0, "error": repr(exc)}


async def _attach_execution_stats_async(verdict: dict, sbx, started_at: float) -> None:
    verdict["runtime_seconds"] = round(time.monotonic() - started_at, 2)
    try:
        verdict["metrics"] = summarize_metrics(await sbx.get_metrics())
    except Exception as exc:  # noqa: BLE001 - metrics cannot invalidate a result
        verdict["metrics"] = {"samples": 0, "error": repr(exc)}


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
    started_at = time.monotonic()
    sbx = _create_sandbox(template, sandbox_timeout)
    try:
        # 1. apply the prediction patch (empty patch = no-op, still graded)
        applied = not patch.strip()
        if patch.strip():
            sbx.files.write("/tmp/patch.diff", patch, user="root")
            for cmd in GIT_APPLY_CMDS:
                try:
                    res = sbx.commands.run(
                        f"{cmd} /tmp/patch.diff",
                        cwd="/testbed",
                        user="root",
                        timeout=300,
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
        sbx.files.write(
            "/eval.sh", _eval_script_preserving_image_setup(ts, instance), user="root"
        )
        try:
            sbx.commands.run(
                _EVAL_CMD, cwd="/testbed", user="root", timeout=cmd_timeout
            )
        except CommandExitException:
            pass  # non-zero exit is normal when tests fail; we grade from the log
        output = sbx.files.read("/tmp/test_output.txt", user="root")

        # 3. grade
        verdict = _grade(ts, prediction, output)
        verdict["collection_error"] = _detect_collection_error(output)
        verdict["warning_error"] = _detect_warning_error(output)
        verdict["resource_exhausted"] = bool(_RESOURCE_EXHAUSTED.search(output))
        _attach_execution_stats(verdict, sbx, started_at)
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
    started_at = time.monotonic()
    sbx = await _create_sandbox_async(template, sandbox_timeout)
    try:
        applied = not patch.strip()
        if patch.strip():
            await sbx.files.write("/tmp/patch.diff", patch, user="root")
            for cmd in GIT_APPLY_CMDS:
                try:
                    res = await sbx.commands.run(
                        f"{cmd} /tmp/patch.diff",
                        cwd="/testbed",
                        user="root",
                        timeout=300,
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

        await sbx.files.write(
            "/eval.sh", _eval_script_preserving_image_setup(ts, instance), user="root"
        )
        try:
            await sbx.commands.run(
                _EVAL_CMD, cwd="/testbed", user="root", timeout=cmd_timeout
            )
        except CommandExitException:
            pass
        output = await sbx.files.read("/tmp/test_output.txt", user="root")

        verdict = _grade(ts, prediction, output)
        verdict["collection_error"] = _detect_collection_error(output)
        verdict["warning_error"] = _detect_warning_error(output)
        verdict["resource_exhausted"] = bool(_RESOURCE_EXHAUSTED.search(output))
        await _attach_execution_stats_async(verdict, sbx, started_at)
        if keep_output:
            verdict["_output"] = output
        return verdict
    finally:
        await sbx.kill()
