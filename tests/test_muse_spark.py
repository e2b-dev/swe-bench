"""Deterministic tests for the Muse Spark prediction generator.

These never contact Meta or E2B. Everything runs against the doubles in
conftest.py.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from e2b_swebench.agents.muse_spark import (
    DEFAULT_MODEL,
    MissingTemplates,
    SandboxWorkspace,
    _run_tool_call,
    check_templates,
    empty_agent_prediction,
    generate_prediction,
    is_permanent_model_error,
    run_agent,
    task_prompt,
)
from tests.conftest import (
    HELD_OUT_MARKER,
    FakeClient,
    FakeCommandResult,
    FakeMessage,
    FakeSandbox,
    SandboxFactory,
    tool_call,
)

PATCH_TEXT = "diff --git a/requests/utils.py b/requests/utils.py\n+    fixed\n"
HEAD_SHA = "abc1234def5678"


def _install_openai_double(monkeypatch, constructor):
    module = types.ModuleType("openai")
    module.OpenAI = constructor
    monkeypatch.setitem(sys.modules, "openai", module)


def _sandbox(extra_rules=None, patch=PATCH_TEXT) -> FakeSandbox:
    rules = [
        ("git rev-parse HEAD", FakeCommandResult(stdout=HEAD_SHA + "\n")),
        ("git -C /testbed diff", FakeCommandResult(stdout=patch)),
    ]
    return FakeSandbox(rules=(extra_rules or []) + rules)


def _generate(sandbox, client, **kw):
    factory = SandboxFactory(sandbox=sandbox)
    prediction, result = generate_prediction(
        kw.pop("instance"),
        "swebench-psf-requests-6028",
        client,
        create_sandbox=factory,
        **kw,
    )
    return prediction, result, factory


# --- what the agent is allowed to see ----------------------------------------


def test_model_receives_problem_statement_but_no_held_out_fields(instance):
    # a multi-step run, so tool results are checked for leaks too
    client = FakeClient(
        [
            FakeMessage(tool_calls=[tool_call("c1", "run", command="grep -r bug .")]),
            FakeMessage(
                tool_calls=[tool_call("c2", "write_file", path="a.py", content="x")]
            ),
            FakeMessage(content="done"),
        ]
    )
    _generate(_sandbox(), client, instance=instance)

    assert len(client.requests) == 3
    assert instance["problem_statement"] in json.dumps(client.requests[0])
    # test_patch, gold patch, FAIL_TO_PASS and PASS_TO_PASS all carry the marker
    for request in client.requests:
        assert HELD_OUT_MARKER not in json.dumps(request)


def test_task_prompt_mentions_only_the_repo_and_problem(instance):
    prompt = task_prompt(instance)
    assert instance["problem_statement"] in prompt
    assert instance["repo"] in prompt
    assert HELD_OUT_MARKER not in prompt


def test_system_prompt_forbids_editing_tests_and_committing(instance):
    client = FakeClient([FakeMessage(content="done")])
    _generate(_sandbox(), client, instance=instance)

    system = client.requests[0]["messages"][0]
    assert system["role"] == "system"
    assert "test files" in system["content"]
    assert "git commit" in system["content"]


# --- tools -------------------------------------------------------------------


def test_run_executes_from_testbed_as_root_with_the_conda_env():
    sandbox = FakeSandbox(default=FakeCommandResult(stdout="hello"))
    workspace = SandboxWorkspace(sandbox, command_timeout=42)

    workspace.run("pytest -q")

    recorded = sandbox.commands_run[-1]
    assert recorded.cwd == "/testbed"
    assert recorded.user == "root"
    assert recorded.timeout == 42
    assert "activate testbed" in recorded.command
    assert recorded.command.endswith("pytest -q")


def test_run_reports_exit_status_separately_from_output():
    sandbox = FakeSandbox(
        default=FakeCommandResult(stdout="out", stderr="err", exit_code=3)
    )
    rendered = SandboxWorkspace(sandbox).run("false")

    assert rendered.startswith("exit_code: 3\n")
    assert "out" in rendered and "err" in rendered


def test_a_non_zero_exit_is_an_answer_not_a_tool_error():
    sandbox = FakeSandbox(
        default=FakeCommandResult(
            stdout="FAILED tests/test_x.py::test_y", stderr="1 failed", exit_code=1
        )
    )

    rendered = _run_tool_call(
        SandboxWorkspace(sandbox), tool_call("c1", "run", command="pytest -q")
    )

    assert not rendered.startswith("tool error")
    assert rendered.startswith("exit_code: 1\n")
    assert "FAILED tests/test_x.py::test_y" in rendered  # stdout survives
    assert "1 failed" in rendered  # and so does stderr


def test_a_command_not_found_reports_127_with_its_message():
    sandbox = FakeSandbox(
        default=FakeCommandResult(stderr="bash: nope: command not found", exit_code=127)
    )

    rendered = SandboxWorkspace(sandbox).run("nope")

    assert rendered.startswith("exit_code: 127\n")
    assert "command not found" in rendered


def test_run_rejects_an_empty_command():
    sandbox = FakeSandbox()
    assert "tool error" in SandboxWorkspace(sandbox).run("   ")
    assert sandbox.commands_run == []


def test_tool_output_is_truncated_from_the_front():
    sandbox = FakeSandbox(default=FakeCommandResult(stdout="x" * 500 + "TAIL"))
    workspace = SandboxWorkspace(sandbox, max_tool_output_chars=100)

    rendered = workspace.run("noisy")

    assert "TAIL" in rendered
    assert "characters omitted" in rendered
    assert len(rendered) < 300


def test_write_file_stays_inside_the_workspace():
    sandbox = FakeSandbox()
    workspace = SandboxWorkspace(sandbox)

    workspace.write_file("requests/utils.py", "code")

    assert sandbox.writes[-1].path == "/testbed/requests/utils.py"
    assert sandbox.writes[-1].user == "root"


@pytest.mark.parametrize(
    "bad_path",
    ["/etc/passwd", "../escape.py", "../../etc/passwd", "..", "", "/testbed/x.py"],
)
def test_write_file_rejects_absolute_and_escaping_paths(bad_path):
    sandbox = FakeSandbox()
    workspace = SandboxWorkspace(sandbox)

    with pytest.raises(ValueError):
        workspace.write_file(bad_path, "pwned")
    assert sandbox.writes == []


def test_dispatch_reports_unknown_tools_without_raising():
    workspace = SandboxWorkspace(FakeSandbox())
    assert "unknown tool" in workspace.dispatch("rm_rf", {})


# --- the loop ----------------------------------------------------------------


def test_loop_runs_tool_calls_then_stops_on_a_final_message():
    sandbox = FakeSandbox(default=FakeCommandResult(stdout="1 passed"))
    workspace = SandboxWorkspace(sandbox)
    client = FakeClient(
        [
            FakeMessage(tool_calls=[tool_call("c1", "run", command="pytest -q")]),
            FakeMessage(
                tool_calls=[tool_call("c2", "write_file", path="a.py", content="x")]
            ),
            FakeMessage(content="fixed it"),
        ]
    )

    final, steps = run_agent(client, workspace, task="t", model="m", max_steps=10)

    assert final == "fixed it"
    assert steps == 3
    assert sandbox.commands_run[0].command.endswith("pytest -q")
    assert sandbox.writes[0].path == "/testbed/a.py"


def test_loop_terminates_at_the_step_limit():
    workspace = SandboxWorkspace(FakeSandbox())
    forever = [
        FakeMessage(tool_calls=[tool_call(f"c{i}", "run", command="echo hi")])
        for i in range(20)
    ]
    client = FakeClient(forever)

    final, steps = run_agent(client, workspace, task="t", model="m", max_steps=4)

    assert steps == 4
    assert "step limit" in final
    assert len(client.requests) == 4


def test_loop_rejects_a_zero_step_budget():
    with pytest.raises(ValueError):
        run_agent(
            FakeClient(),
            SandboxWorkspace(FakeSandbox()),
            task="t",
            model="m",
            max_steps=0,
        )


def test_a_failing_tool_is_reported_to_the_model_not_raised():
    workspace = SandboxWorkspace(FakeSandbox())
    client = FakeClient(
        [
            FakeMessage(
                tool_calls=[tool_call("c1", "write_file", path="/etc/x", content="p")]
            ),
            FakeMessage(content="ok"),
        ]
    )

    run_agent(client, workspace, task="t", model="m", max_steps=5)

    tool_message = client.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "tool error" in tool_message["content"]


# --- patch capture -----------------------------------------------------------


def test_patch_is_captured_against_the_recorded_head(instance):
    sandbox = _sandbox()
    prediction, result, _ = _generate(
        sandbox, FakeClient([FakeMessage(content="done")]), instance=instance
    )

    diff_commands = [
        c.command for c in sandbox.commands_run if "git -C /testbed diff" in c.command
    ]
    assert len(diff_commands) == 1
    assert (
        f"git -C /testbed diff --no-color --no-ext-diff --binary {HEAD_SHA} --"
        in diff_commands[0]
    )
    assert result.head_commit == HEAD_SHA
    assert prediction["model_patch"] == PATCH_TEXT


def test_patch_capture_includes_untracked_files(instance):
    sandbox = _sandbox()

    _generate(sandbox, FakeClient([FakeMessage(content="done")]), instance=instance)

    commands = [record.command for record in sandbox.commands_run]
    prepare = next(
        i for i, command in enumerate(commands) if "add --intent-to-add" in command
    )
    capture = next(i for i, command in enumerate(commands) if "--binary" in command)
    assert prepare < capture


def test_patch_capture_keeps_stdout_verbatim_and_drops_stderr(instance):
    sandbox = FakeSandbox(
        rules=[
            ("git rev-parse HEAD", FakeCommandResult(stdout=HEAD_SHA)),
            (
                "git -C /testbed diff",
                FakeCommandResult(stdout=PATCH_TEXT, stderr="warning: CRLF"),
            ),
        ]
    )
    prediction, _, _ = _generate(
        sandbox, FakeClient([FakeMessage(content="done")]), instance=instance
    )

    assert prediction["model_patch"] == PATCH_TEXT
    assert "warning" not in prediction["model_patch"]


# --- prediction shape --------------------------------------------------------


def test_prediction_has_exactly_the_official_three_keys(instance):
    prediction, _, _ = _generate(
        _sandbox(), FakeClient([FakeMessage(content="done")]), instance=instance
    )

    assert set(prediction) == {"instance_id", "model_name_or_path", "model_patch"}
    assert prediction["instance_id"] == instance["instance_id"]
    assert prediction["model_name_or_path"] == DEFAULT_MODEL
    json.loads(json.dumps(prediction))  # round-trips as one JSONL record


def test_generation_status_is_structured_and_kept_out_of_the_prediction(instance):
    prediction, result, _ = _generate(
        _sandbox(), FakeClient([FakeMessage(content="all good")]), instance=instance
    )

    record = result.to_dict()
    assert record["instance_id"] == instance["instance_id"]
    assert record["status"] == "ok"
    assert record["steps"] == 1
    assert record["patch_bytes"] == len(PATCH_TEXT.encode("utf-8"))
    assert record["sandbox_id"] == "sbx-test-0001"
    assert record["final_message"] == "all good"
    json.loads(json.dumps(record))
    # none of it leaks into the graded artifact
    assert set(prediction) == {"instance_id", "model_name_or_path", "model_patch"}


def test_empty_agent_prediction_shape():
    prediction = empty_agent_prediction("a__b-1", "custom-model")
    assert prediction == {
        "instance_id": "a__b-1",
        "model_name_or_path": "custom-model",
        "model_patch": "",
    }


# --- failure paths -----------------------------------------------------------


def test_a_model_api_failure_yields_an_empty_patch_and_kills_the_sandbox(instance):
    sandbox = _sandbox()
    client = FakeClient(raises=RuntimeError("meta api 503"))

    prediction, result, _ = _generate(sandbox, client, instance=instance)

    assert prediction["model_patch"] == ""
    assert set(prediction) == {"instance_id", "model_name_or_path", "model_patch"}
    assert result.status == "error"
    assert "meta api 503" in result.error
    assert sandbox.killed == 1


def test_a_failed_head_read_yields_an_empty_patch_and_kills_the_sandbox(instance):
    sandbox = FakeSandbox(
        rules=[
            (
                "git rev-parse HEAD",
                FakeCommandResult(stderr="not a repo", exit_code=128),
            ),
        ]
    )

    prediction, result, _ = _generate(
        sandbox, FakeClient([FakeMessage(content="done")]), instance=instance
    )

    assert prediction["model_patch"] == ""
    assert result.status == "error"
    assert sandbox.killed == 1


def test_a_failed_patch_capture_yields_an_empty_patch(instance):
    sandbox = FakeSandbox(
        rules=[
            ("git rev-parse HEAD", FakeCommandResult(stdout=HEAD_SHA)),
            ("git -C /testbed diff", FakeCommandResult(stderr="boom", exit_code=1)),
        ]
    )

    prediction, result, _ = _generate(
        sandbox, FakeClient([FakeMessage(content="done")]), instance=instance
    )

    assert prediction["model_patch"] == ""
    assert result.status == "error"
    assert "boom" in result.error
    assert sandbox.killed == 1


def test_a_failed_untracked_file_scan_yields_an_empty_patch(instance):
    sandbox = _sandbox(
        extra_rules=[
            (
                "add --intent-to-add",
                FakeCommandResult(stderr="index unavailable", exit_code=1),
            ),
        ]
    )

    prediction, result, _ = _generate(
        sandbox, FakeClient([FakeMessage(content="done")]), instance=instance
    )

    assert prediction["model_patch"] == ""
    assert result.status == "error"
    assert "index unavailable" in result.error
    assert sandbox.killed == 1


def test_a_sandbox_creation_failure_yields_an_empty_patch(instance):
    factory = SandboxFactory(
        sandbox=_sandbox(),
        raises=RuntimeError("sandbox unavailable"),
    )

    prediction, result = generate_prediction(
        instance,
        "swebench-psf-requests-6028",
        FakeClient(),
        create_sandbox=factory,
    )

    assert prediction["model_patch"] == ""
    assert result.status == "error"
    assert "sandbox unavailable" in result.error
    assert factory.sandbox.killed == 0


def test_an_agent_that_changed_nothing_is_recorded_as_empty_patch(instance):
    prediction, result, _ = _generate(
        _sandbox(patch=""),
        FakeClient([FakeMessage(content="no change needed")]),
        instance=instance,
    )

    assert prediction["model_patch"] == ""
    assert result.status == "empty_patch"
    assert result.error is None


def test_step_exhaustion_still_captures_whatever_was_written(instance):
    sandbox = _sandbox()
    client = FakeClient(
        [
            FakeMessage(tool_calls=[tool_call(f"c{i}", "run", command="echo hi")])
            for i in range(10)
        ]
    )

    prediction, result, _ = _generate(sandbox, client, instance=instance, max_steps=2)

    assert prediction["model_patch"] == PATCH_TEXT
    assert result.status == "ok"
    assert result.steps == 2
    assert sandbox.killed == 1


def test_the_sandbox_is_killed_on_success(instance):
    sandbox = _sandbox()
    _generate(sandbox, FakeClient([FakeMessage(content="done")]), instance=instance)
    assert sandbox.killed == 1


def test_a_keyboard_interrupt_kills_the_sandbox_and_propagates(instance):
    sandbox = _sandbox()
    client = FakeClient(raises=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        _generate(sandbox, client, instance=instance)

    assert sandbox.killed == 1


def test_a_kill_failure_does_not_mask_the_result(instance):
    sandbox = _sandbox()
    sandbox.kill = lambda: (_ for _ in ()).throw(RuntimeError("kill failed"))

    prediction, result, _ = _generate(
        sandbox, FakeClient([FakeMessage(content="done")]), instance=instance
    )

    assert result.status == "ok"
    assert prediction["model_patch"] == PATCH_TEXT
    assert "sandbox cleanup failed" in result.error


# --- secrets -----------------------------------------------------------------


def test_the_sandbox_is_created_with_no_environment_at_all(instance, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "sk-meta-do-not-leak")
    sandbox = _sandbox()

    _, _, factory = _generate(
        sandbox, FakeClient([FakeMessage(content="done")]), instance=instance
    )

    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert "envs" not in call
    assert "sk-meta-do-not-leak" not in json.dumps(call)
    # and nothing wrote the key into the sandbox either
    assert all("sk-meta-do-not-leak" not in w.content for w in sandbox.writes)
    assert all("sk-meta-do-not-leak" not in c.command for c in sandbox.commands_run)


def test_the_sandbox_is_created_from_the_instance_template_with_the_timeout(instance):
    _, _, factory = _generate(
        _sandbox(),
        FakeClient([FakeMessage(content="done")]),
        instance=instance,
        sandbox_timeout=1234,
    )

    assert factory.calls[0]["template"] == "swebench-psf-requests-6028"
    assert factory.calls[0]["timeout"] == 1234


# --- template preflight ------------------------------------------------------


def test_check_templates_names_every_missing_instance_and_the_build_command():
    with pytest.raises(MissingTemplates) as excinfo:
        check_templates(["a__b-1", "c__d-2"], exists=lambda name: False)

    message = str(excinfo.value)
    assert "scripts/build_templates.py --instances" in message
    assert "a__b-1" in message and "c__d-2" in message


def test_check_templates_passes_when_every_template_exists():
    seen: list[str] = []

    def exists(name):
        seen.append(name)
        return True

    check_templates(["psf__requests-6028"], exists=exists)
    assert seen == ["swebench-psf-requests-6028"]


# --- Meta Model API conformance ----------------------------------------------


def test_the_default_model_is_the_id_meta_publishes():
    assert DEFAULT_MODEL == "muse-spark-1.1"


def test_the_default_base_url_is_metas_documented_endpoint():
    from e2b_swebench.agents.muse_spark import DEFAULT_BASE_URL

    assert DEFAULT_BASE_URL == "https://api.meta.ai/v1"


def test_tool_schemas_match_the_documented_function_tool_shape():
    from e2b_swebench.agents.muse_spark import TOOLS

    assert [t["function"]["name"] for t in TOOLS] == ["run", "write_file"]
    for tool in TOOLS:
        # Chat Completions accepts ONLY `function` tools (built-ins such as
        # web_search are Responses-API only).
        assert tool["type"] == "function"
        fn = tool["function"]
        assert set(fn) == {"name", "description", "parameters"}
        assert fn["parameters"]["type"] == "object"
        assert fn["parameters"]["required"]
        assert fn["description"].strip()


def test_every_request_uses_tool_choice_auto():
    """Only "auto" is accepted; "none"/"required"/named return HTTP 400."""
    sandbox = FakeSandbox(default=FakeCommandResult(stdout="ok"))
    client = FakeClient(
        [
            FakeMessage(tool_calls=[tool_call("c1", "run", command="ls")]),
            FakeMessage(content="done"),
        ]
    )

    run_agent(client, SandboxWorkspace(sandbox), task="t", model="m", max_steps=5)

    assert len(client.requests) == 2
    for request in client.requests:
        assert request["tool_choice"] == "auto"
        assert "n" not in request  # n must be 1, so it is never sent
        assert request["tools"]


def test_the_assistant_tool_calls_message_is_replayed_before_tool_results():
    """Skipping it makes the API reject the next request with HTTP 400."""
    sandbox = FakeSandbox(default=FakeCommandResult(stdout="ok"))
    client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    tool_call("c1", "run", command="ls"),
                    tool_call("c2", "run", command="pwd"),
                ]
            ),
            FakeMessage(content="done"),
        ]
    )

    run_agent(client, SandboxWorkspace(sandbox), task="t", model="m", max_steps=5)

    messages = client.requests[1]["messages"]
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assistant = messages[2]
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["c1", "c2"]
    # one tool message per tool_call_id, in order
    assert [m["tool_call_id"] for m in messages[3:]] == ["c1", "c2"]


def test_parallel_tool_calls_in_one_turn_are_all_executed():
    sandbox = FakeSandbox(default=FakeCommandResult(stdout="ok"))
    client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    tool_call("c1", "run", command="echo one"),
                    tool_call("c2", "run", command="echo two"),
                    tool_call("c3", "write_file", path="a.py", content="x"),
                ]
            ),
            FakeMessage(content="done"),
        ]
    )

    run_agent(client, SandboxWorkspace(sandbox), task="t", model="m", max_steps=5)

    assert len(sandbox.commands_run) == 2
    assert len(sandbox.writes) == 1


@pytest.mark.parametrize(
    ("status", "name", "permanent"),
    [
        (401, "AuthenticationError", True),
        (404, "NotFoundError", True),
        (403, "PermissionDeniedError", True),
        (400, "BadRequestError", True),
        (429, "RateLimitError", False),
        (500, "APIStatusError", False),
        (503, "APIStatusError", False),
        (None, "APITimeoutError", False),
        (None, "APIConnectionError", False),
        (None, "RuntimeError", False),
    ],
)
def test_permanent_errors_are_told_apart_from_transient_ones(status, name, permanent):
    """Meta's retry table: 4xx permanent, 429/5xx/timeout transient."""
    from e2b_swebench.agents.muse_spark import is_permanent_model_error

    error = type(name, (Exception,), {})()
    if status is not None:
        error.status_code = status

    assert is_permanent_model_error(error) is permanent


def test_a_permanent_error_marks_the_result_fatal(instance):
    error = type("AuthenticationError", (Exception,), {})()
    error.status_code = 401
    prediction, result, _ = _generate(
        _sandbox(), FakeClient(raises=error), instance=instance
    )

    assert result.status == "error"
    assert result.fatal is True
    assert prediction["model_patch"] == ""


def test_a_transient_error_is_not_fatal(instance):
    error = type("RateLimitError", (Exception,), {})()
    error.status_code = 429
    _, result, _ = _generate(_sandbox(), FakeClient(raises=error), instance=instance)

    assert result.status == "error"
    assert result.fatal is False


def test_the_client_is_built_the_way_meta_documents(monkeypatch):
    from e2b_swebench.agents.muse_spark import DEFAULT_MAX_RETRIES, create_model_client

    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    _install_openai_double(monkeypatch, fake_openai)
    monkeypatch.setenv("META_API_KEY", "sk-test")
    monkeypatch.delenv("META_BASE_URL", raising=False)

    create_model_client()

    assert captured["base_url"] == "https://api.meta.ai/v1"
    assert captured["api_key"] == "sk-test"
    # the SDK's own default is 2; Meta's error-handling recipe uses 5
    assert captured["max_retries"] == DEFAULT_MAX_RETRIES == 5


def test_the_base_url_can_be_overridden_by_env(monkeypatch):
    from e2b_swebench.agents.muse_spark import create_model_client

    captured = {}
    _install_openai_double(monkeypatch, lambda **kw: captured.update(kw) or object())
    monkeypatch.setenv("META_API_KEY", "sk-test")
    monkeypatch.setenv("META_BASE_URL", "https://proxy.internal/v1")

    create_model_client()
    assert captured["base_url"] == "https://proxy.internal/v1"


def test_metas_own_key_name_is_accepted(monkeypatch):
    """Meta's docs and CLIs export MODEL_API_KEY; that shell should work as-is."""
    from e2b_swebench.agents.muse_spark import create_model_client

    captured = {}
    _install_openai_double(monkeypatch, lambda **kw: captured.update(kw) or object())
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_API_KEY", "sk-model-api")

    create_model_client()
    assert captured["api_key"] == "sk-model-api"


def test_this_projects_key_name_wins_when_both_are_set(monkeypatch):
    from e2b_swebench.agents.muse_spark import create_model_client

    captured = {}
    _install_openai_double(monkeypatch, lambda **kw: captured.update(kw) or object())
    monkeypatch.setenv("META_API_KEY", "sk-meta")
    monkeypatch.setenv("MODEL_API_KEY", "sk-model-api")

    create_model_client()
    assert captured["api_key"] == "sk-meta"


def test_a_missing_api_key_is_refused_before_any_request(monkeypatch):
    from e2b_swebench.agents.muse_spark import create_model_client

    _install_openai_double(monkeypatch, lambda **_: object())
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        create_model_client()


def test_check_model_sends_one_short_request():
    from e2b_swebench.agents.muse_spark import check_model

    client = FakeClient([FakeMessage(content="ready")])
    check_model(client, "muse-spark-1.1")

    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["model"] == "muse-spark-1.1"
    assert "max_tokens" not in request
    assert "tools" not in request


def test_check_model_wraps_a_rejection_with_actionable_guidance():
    from e2b_swebench.agents.muse_spark import ModelUnavailable, check_model

    error = type("NotFoundError", (Exception,), {})()
    error.status_code = 404
    error.body = {"code": "model_not_found", "type": "invalid_request_error"}

    with pytest.raises(ModelUnavailable) as excinfo:
        check_model(FakeClient(raises=error), "muse-spark-9.9")

    message = str(excinfo.value)
    assert "muse-spark-9.9" in message
    assert "META_API_KEY" in message
    assert "model_not_found" in message
    assert is_permanent_model_error(excinfo.value) is True
