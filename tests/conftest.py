"""Offline test doubles for the Muse Spark prediction runner."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from e2b import CommandExitException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --- sandbox double ----------------------------------------------------------


@dataclass
class FakeCommandResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class RecordedCommand:
    command: str
    cwd: str | None
    user: str | None
    timeout: int | None


@dataclass
class RecordedWrite:
    path: str
    content: str
    user: str | None


class FakeFiles:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self._sandbox = sandbox

    def write(self, path: str, content: str, user: str | None = None) -> None:
        self._sandbox.writes.append(RecordedWrite(path, content, user))


class FakeSandbox:
    """Answers commands from a list of (substring, result) rules, in order."""

    def __init__(
        self,
        rules: list[tuple[str, FakeCommandResult]] | None = None,
        default: FakeCommandResult | None = None,
    ) -> None:
        self.sandbox_id = "sbx-test-0001"
        self.rules = rules or []
        self.default = default or FakeCommandResult(stdout="ok")
        self.commands_run: list[RecordedCommand] = []
        self.writes: list[RecordedWrite] = []
        self.killed = 0
        self.files = FakeFiles(self)
        self.commands = self

    # `sandbox.commands.run(...)` — FakeSandbox is its own `.commands`.
    def run(self, command, cwd=None, user=None, timeout=None, **_):
        self.commands_run.append(RecordedCommand(command, cwd, user, timeout))
        result = self.default
        for needle, rule in self.rules:
            if needle in command:
                result = rule
                break
        # E2B raises CommandExitException for non-zero command exits.
        if result.exit_code != 0:
            raise CommandExitException(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                error=None,
            )
        return result

    def kill(self) -> None:
        self.killed += 1


@dataclass
class SandboxFactory:
    """Stands in for driver._create_sandbox, recording exactly how it was called."""

    sandbox: FakeSandbox
    calls: list[dict[str, Any]] = field(default_factory=list)
    raises: BaseException | None = None

    def __call__(self, template, timeout=None, **kwargs):
        self.calls.append({"template": template, "timeout": timeout, **kwargs})
        if self.raises is not None:
            raise self.raises
        return self.sandbox


# --- model double ------------------------------------------------------------


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.role = "assistant"

    def model_dump(self, exclude_none: bool = False) -> dict:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None or not exclude_none:
            payload["content"] = self.content
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return payload


@dataclass
class _Choice:
    message: FakeMessage


@dataclass
class _Response:
    choices: list[_Choice]


class FakeCompletions:
    def __init__(self, client: FakeClient) -> None:
        self._client = client

    def create(self, **kwargs):
        # The loop mutates one `messages` list in place; the real client
        # serializes it per call, so snapshot it or every recorded request
        # aliases the final state.
        recorded = dict(kwargs)
        recorded["messages"] = [dict(m) for m in kwargs.get("messages", [])]
        self._client.requests.append(recorded)
        if self._client.raises is not None:
            raise self._client.raises
        if self._client.scripted:
            message = self._client.scripted.pop(0)
        else:
            message = FakeMessage(content="done")
        return _Response(choices=[_Choice(message=message)])


class FakeClient:
    """Scripted OpenAI-compatible client. `scripted` is consumed one call at a time."""

    def __init__(
        self,
        scripted: list[FakeMessage] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.scripted = list(scripted or [])
        self.raises = raises
        self.requests: list[dict] = []
        self.chat = self
        self.completions = FakeCompletions(self)


def tool_call(call_id: str, name: str, **arguments) -> FakeToolCall:
    import json

    return FakeToolCall(id=call_id, function=FakeFunction(name, json.dumps(arguments)))


# --- instance fixture --------------------------------------------------------

HELD_OUT_MARKER = "SECRET_HELD_OUT_TEST_MARKER"


@pytest.fixture
def instance() -> dict:
    """A dataset row shaped like the real thing, with the held-out fields poisoned.

    HELD_OUT_MARKER appears only in fields the agent must never see, so a leak
    is detectable by searching the whole request payload for one string.
    """
    return {
        "instance_id": "psf__requests-6028",
        "repo": "psf/requests",
        "base_commit": "0192aac24123735b3eaf9b08df46429bb770c283",
        "problem_statement": "Proxy authentication bug: I get a 407, expected 200.",
        "patch": f"diff --git a/gold.py b/gold.py\n{HELD_OUT_MARKER}\n",
        "test_patch": f"diff --git a/tests/t.py b/tests/t.py\n{HELD_OUT_MARKER}\n",
        "FAIL_TO_PASS": f'["tests/test_utils.py::{HELD_OUT_MARKER}"]',
        "PASS_TO_PASS": f'["tests/test_utils.py::{HELD_OUT_MARKER}_p2p"]',
    }
