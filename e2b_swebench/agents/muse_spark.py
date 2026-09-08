"""Generate SWE-bench predictions with Muse Spark in E2B sandboxes.

The model loop runs in the caller process. Commands and file writes run in a
fresh sandbox created from the selected instance's template. The model receives
the repository name and problem statement, but not the held-out tests, reference
patch, or grader output.
"""

from __future__ import annotations

import json
import os
import posixpath
import shlex
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from e2b import CommandExitException

from .. import config as cfg
from ..driver import _create_sandbox
from ..templates import template_name

WORKSPACE = "/testbed"

# Defaults published in Meta's Model API cookbook. Both can be overridden for
# accounts that use a different endpoint or model deployment.
DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.1"

# Let the OpenAI client retry transient API and connection failures. Permanent
# request, authentication, and model errors are surfaced immediately.
DEFAULT_MAX_RETRIES = 5

DEFAULT_MAX_STEPS = 30
DEFAULT_COMMAND_TIMEOUT_SECONDS = 300
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 12_000

# SWE-bench images install the project in the `testbed` conda environment. The
# fallback keeps custom templates without conda usable.
CONDA_ACTIVATE = "source /opt/miniconda3/bin/activate testbed >/dev/null 2>&1 || true"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run",
            "description": (
                "Run one shell command inside the sandbox workspace. "
                "Use this to inspect files, run tests, and check your work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "A shell command to run from the repository workspace.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or replace one text file inside the sandbox workspace. "
                "The path must be relative to the repository workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from the repository workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete replacement content for the file.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
]


def _truncate_for_model(text: str, limit: int) -> str:
    """Keep the most recent command output while preserving the truncation signal."""
    if len(text) <= limit:
        return text

    omitted = len(text) - limit
    return f"... ({omitted} characters omitted) ...\n{text[-limit:]}"


def system_prompt(workspace: str = WORKSPACE) -> str:
    """Instructions for working in the sandbox without altering benchmark tests."""
    return f"""You are a software engineer fixing a bug in the repository at {workspace}.

You have exactly two tools: run and write_file. Every command and file change
must stay inside {workspace}. You are working in an isolated sandbox; there is
no access to the caller machine, its filesystem, or its credentials.

Work like an engineer: reproduce the failure first, localize it, make the
smallest correct change to the source, and re-run the affected tests to confirm
the fix.

Rules:
- Do NOT modify, add, or delete any test files. The tests that judge this work
  are held out and will be supplied separately.
- Do NOT run `git add`, `git commit`, `git stash`, `git checkout`, or `git reset`.
  Leave your changes in the working tree exactly as you made them.
- Do not try to discover which tests will be used to grade the change."""


@dataclass(frozen=True)
class CommandExecution:
    exit_code: int
    output: str


@dataclass
class GenerationResult:
    """Generation metadata stored separately from the prediction."""

    instance_id: str
    model: str
    status: str  # "ok" | "empty_patch" | "error"
    steps: int = 0
    patch_bytes: int = 0
    sandbox_id: str | None = None
    head_commit: str | None = None
    error: str | None = None
    final_message: str | None = None
    # Stop the batch when every remaining instance would fail for the same reason.
    fatal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "model": self.model,
            "status": self.status,
            "steps": self.steps,
            "patch_bytes": self.patch_bytes,
            "sandbox_id": self.sandbox_id,
            "head_commit": self.head_commit,
            "error": self.error,
            "fatal": self.fatal,
            "final_message": self.final_message,
        }


class TrajectoryWriteError(RuntimeError):
    """Generation cannot continue without its audit trail."""


class Trajectory:
    """Persist the conversation after each message, before the next action."""

    def __init__(self, path: str | Path | None, result: GenerationResult):
        self.path = Path(path) if path is not None else None
        self.result = result
        self.messages: list[dict[str, Any]] = []
        self.stop_reason = "running"

    def record(self, message: dict[str, Any], step: int) -> None:
        self.messages.append(message)
        self.result.steps = step
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        temporary = None
        try:
            payload = json.dumps(
                {
                    "trajectory_format": "muse-spark-1.0",
                    "instance_id": self.result.instance_id,
                    "messages": self.messages,
                    "info": {
                        **self.result.to_dict(),
                        "stop_reason": self.stop_reason,
                        "allow_internet_access": False,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            for name in ("META_API_KEY", "E2B_API_KEY"):
                secret = os.environ.get(name)
                if secret:
                    payload = payload.replace(
                        json.dumps(secret, ensure_ascii=False)[1:-1], "[REDACTED]"
                    )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(payload + "\n")
            os.replace(temporary, self.path)
        except (OSError, TypeError, ValueError) as error:
            raise TrajectoryWriteError("could not save trajectory") from error
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)


class ModelUnavailable(RuntimeError):
    """Raised when the model preflight fails before sandbox creation."""

    def __init__(self, model: str, cause: BaseException) -> None:
        self.model = model
        self.cause = cause
        detail = getattr(cause, "body", None) or str(cause)
        super().__init__(
            f"the model API rejected a connectivity check for {model!r}: {detail}\n"
            f"Check META_API_KEY, META_BASE_URL, and the model id (--model). "
            f"Meta's published id is {DEFAULT_MODEL!r}."
        )


def is_permanent_model_error(error: BaseException) -> bool:
    """Return whether retrying the same request on another instance is pointless."""
    status = getattr(error, "status_code", None)
    if status in (400, 401, 403, 404):
        return True
    return type(error).__name__ in {
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "ModelUnavailable",
    }


class SandboxWorkspace:
    """Bind the model's command and file tools to one sandbox workspace."""

    def __init__(
        self,
        sandbox: Any,
        *,
        workspace: str = WORKSPACE,
        command_timeout: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    ) -> None:
        if not workspace.startswith("/"):
            raise ValueError("workspace must be an absolute sandbox path")

        self.sandbox = sandbox
        self.workspace = workspace.rstrip("/")
        self.command_timeout = command_timeout
        self.max_tool_output_chars = max_tool_output_chars

    def run(self, command: str) -> str:
        """Run a model-selected command from the workspace."""
        if not command.strip():
            return "tool error: command must not be empty"

        return self._render_execution(self._execute(command))

    def write_file(self, path: str, content: str) -> str:
        """Write a model-selected file, refusing anything outside the workspace."""
        remote_path = self._workspace_file_path(path)
        self.sandbox.files.write(remote_path, content, user="root")
        return f"wrote {path} ({len(content.encode('utf-8'))} bytes)"

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> str:
        """Dispatch one tool call without exposing the sandbox to the model."""
        if name == "run":
            command = arguments.get("command")
            if not isinstance(command, str):
                return "tool error: run requires a string command"
            return self.run(command)

        if name == "write_file":
            path = arguments.get("path")
            content = arguments.get("content")
            if not isinstance(path, str) or not isinstance(content, str):
                return "tool error: write_file requires string path and content"
            return self.write_file(path, content)

        return f"tool error: unknown tool {name}"

    def head_commit(self) -> str:
        """Return the template's prepared HEAD before the agent changes it."""
        execution = self._execute("git rev-parse HEAD")
        if execution.exit_code != 0:
            raise RuntimeError(f"could not read HEAD:\n{execution.output}")
        return execution.output.strip().splitlines()[-1].strip()

    def capture_patch(self, base: str) -> str:
        """Capture tracked and newly created files relative to the recorded HEAD."""
        intent_to_add = self._execute(
            f"git -C {shlex.quote(self.workspace)} add --intent-to-add --all -- ."
        )
        if intent_to_add.exit_code != 0:
            raise RuntimeError(
                f"could not prepare untracked files for the patch:\n{intent_to_add.output}"
            )

        execution = self._execute(
            f"git -C {shlex.quote(self.workspace)} diff --no-color --no-ext-diff "
            f"--binary {shlex.quote(base)} --",
            raw=True,
        )
        if execution.exit_code != 0:
            raise RuntimeError(f"could not capture the patch:\n{execution.output}")
        return execution.output

    def wrap_command(self, command: str) -> str:
        return f"{CONDA_ACTIVATE}; {command}"

    def _execute(self, command: str, *, raw: bool = False) -> CommandExecution:
        """Run a command and retain stdout/stderr from non-zero exits."""
        try:
            result = self.sandbox.commands.run(
                self.wrap_command(command),
                cwd=self.workspace,
                user="root",
                timeout=self.command_timeout,
            )
        except CommandExitException as exited:
            result = exited

        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        exit_code = int(getattr(result, "exit_code", -1))
        if raw and exit_code == 0:
            return CommandExecution(
                exit_code=exit_code,
                output=stdout,
            )

        output = "\n".join(part for part in (stdout, stderr) if part).strip()
        return CommandExecution(
            exit_code=exit_code,
            output=output or "(no output)",
        )

    def _render_execution(self, execution: CommandExecution) -> str:
        output = _truncate_for_model(execution.output, self.max_tool_output_chars)
        return f"exit_code: {execution.exit_code}\n{output}"

    def _workspace_file_path(self, path: str) -> str:
        if not path or path.startswith("/"):
            raise ValueError("path must be a non-empty relative workspace path")

        normalized = posixpath.normpath(path)
        if normalized in {".", ".."} or normalized.startswith("../"):
            raise ValueError("path must stay inside the workspace")

        return posixpath.join(self.workspace, normalized)


def _message_as_dict(message: Any) -> dict[str, Any]:
    """Convert an SDK message to the Chat Completions request shape."""
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)

    payload: dict[str, Any] = {"role": "assistant"}
    content = getattr(message, "content", None)
    if content is not None:
        payload["content"] = content

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]

    return payload


def _run_tool_call(workspace: SandboxWorkspace, tool_call: Any) -> str:
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as error:
        return f"tool error: invalid JSON arguments: {error.msg}"

    if not isinstance(arguments, dict):
        return "tool error: tool arguments must be a JSON object"

    try:
        return workspace.dispatch(tool_call.function.name, arguments)
    except Exception as error:  # noqa: BLE001 - tool failures are returned to the model
        return f"tool error: {error}"


def task_prompt(instance: Mapping[str, Any]) -> str:
    """Build a task from only the repository name and problem statement."""
    return (
        f"Fix this bug in the `{instance['repo']}` repository, checked out at "
        f"{WORKSPACE}.\n\n"
        "--- issue ---\n"
        f"{instance['problem_statement']}\n\n"
        "Reproduce the failure, fix the source, and verify your fix by running "
        "the tests that already exist in the repository."
    )


def run_agent(
    client: Any,
    workspace: SandboxWorkspace,
    *,
    task: str,
    model: str,
    max_steps: int,
    trajectory: Trajectory | None = None,
) -> tuple[str, int]:
    """Run the model loop. Returns (final message, steps used).

    Exhausting `max_steps` is not an error here: an agent that ran out of turns
    has usually still written something worth grading, and an unresolved verdict
    is a truer signal than a discarded run.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(workspace.workspace)},
        {"role": "user", "content": task},
    ]
    if trajectory is not None:
        for message in messages:
            trajectory.record(message, 0)

    for step in range(1, max_steps + 1):
        # Meta's Chat Completions tool calling currently accepts only "auto".
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message
        # Replay the assistant tool-call message before its corresponding results.
        messages.append(_message_as_dict(message))
        if trajectory is not None:
            trajectory.record(messages[-1], step)
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            if trajectory is not None:
                trajectory.stop_reason = "completed"
            return message.content or "(the model returned no final summary)", step

        for tool_call in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": _run_tool_call(workspace, tool_call),
                }
            )
            if trajectory is not None:
                trajectory.record(messages[-1], step)

    if trajectory is not None:
        trajectory.stop_reason = "step_limit"
    return f"(step limit of {max_steps} reached)", max_steps


def create_model_client(max_retries: int = DEFAULT_MAX_RETRIES) -> Any:
    """Create the OpenAI-compatible Meta Model API client."""
    try:
        from openai import OpenAI
    except ImportError as error:  # pragma: no cover - exercised by install, not tests
        raise RuntimeError(
            "the Muse Spark agent needs the OpenAI-compatible client: "
            "pip install -e '.[muse]'"
        ) from error

    api_key = os.environ.get("META_API_KEY")
    if not api_key:
        raise RuntimeError("META_API_KEY is required to generate predictions")

    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("META_BASE_URL", DEFAULT_BASE_URL),
        max_retries=max_retries,
    )


def check_model(client: Any, model: str) -> None:
    """Check the API key and model with a minimal request before creating a sandbox."""
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
    except Exception as error:
        raise ModelUnavailable(model, error) from error


def empty_agent_prediction(instance_id: str, model: str) -> dict[str, Any]:
    """Return an unresolved prediction without changing the batch denominator."""
    return {
        "instance_id": instance_id,
        "model_name_or_path": model,
        "model_patch": "",
    }


def generate_prediction(
    instance: Mapping[str, Any],
    template: str,
    client: Any,
    *,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    sandbox_timeout: int = cfg.SANDBOX_TIMEOUT,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    create_sandbox: Any = _create_sandbox,
    trajectory_path: str | Path | None = None,
) -> tuple[dict[str, Any], GenerationResult]:
    """Run the agent on one instance in a fresh sandbox.

    Returns (prediction, generation result). Never raises for a model, tool, or
    sandbox failure: the prediction comes back with an empty patch and the
    reason lands in the GenerationResult. Sandbox cleanup is attempted on every
    path, including KeyboardInterrupt. Interrupts and trajectory persistence
    failures propagate to the caller.
    """
    instance_id = instance["instance_id"]
    result = GenerationResult(instance_id=instance_id, model=model, status="error")
    trajectory = Trajectory(trajectory_path, result)
    trajectory.save()

    sandbox = None
    try:
        # No `envs=`: API keys and held-out benchmark data stay in the caller.
        sandbox = create_sandbox(template, sandbox_timeout, allow_internet_access=False)
        result.sandbox_id = getattr(sandbox, "sandbox_id", None)
        workspace = SandboxWorkspace(sandbox, command_timeout=command_timeout)
        head = workspace.head_commit()
        result.head_commit = head

        final_message, steps = run_agent(
            client,
            workspace,
            task=task_prompt(instance),
            model=model,
            max_steps=max_steps,
            trajectory=trajectory,
        )
        result.steps = steps
        result.final_message = final_message

        patch = workspace.capture_patch(head)
        result.patch_bytes = len(patch.encode("utf-8"))
        result.status = "ok" if patch.strip() else "empty_patch"
        return (
            {
                "instance_id": instance_id,
                "model_name_or_path": model,
                "model_patch": patch,
            },
            result,
        )
    except KeyboardInterrupt:
        trajectory.stop_reason = "interrupted"
        result.error = "KeyboardInterrupt"
        raise
    except TrajectoryWriteError:
        trajectory.stop_reason = "error"
        result.error = "trajectory persistence failed"
        raise
    except Exception as error:  # noqa: BLE001 - instance failures become empty predictions
        trajectory.stop_reason = "error"
        result.status = "error"
        result.error = repr(error)
        result.fatal = is_permanent_model_error(error)
        return empty_agent_prediction(instance_id, model), result
    finally:
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception as cleanup_error:  # noqa: BLE001
                detail = f"sandbox cleanup failed: {cleanup_error!r}"
                result.error = f"{result.error}; {detail}" if result.error else detail
        trajectory.save()


class MissingTemplates(RuntimeError):
    """Raised before any sandbox is created, so a bad selection costs nothing."""

    def __init__(self, instance_ids: Sequence[str]) -> None:
        self.instance_ids = list(instance_ids)
        joined = ",".join(self.instance_ids)
        super().__init__(
            f"{len(self.instance_ids)} instance(s) have no E2B template yet: {joined}\n"
            f"Build them first:\n"
            f"    python scripts/build_templates.py --instances {joined}"
        )


def check_templates(instance_ids: Sequence[str], exists: Any = None) -> None:
    """Fail the whole run up front if any selected template is missing."""
    if exists is None:
        from e2b import Template

        exists = Template.exists

    missing = [iid for iid in instance_ids if not exists(template_name(iid))]
    if missing:
        raise MissingTemplates(missing)
