import json

import pytest

from e2b_swebench import driver
from e2b_swebench.agents.muse_spark import TrajectoryWriteError, generate_prediction
from tests.conftest import (
    FakeClient,
    FakeCommandResult,
    FakeMessage,
    FakeSandbox,
    SandboxFactory,
    tool_call,
)


def test_network_policy_reaches_sdk(monkeypatch):
    calls = []
    monkeypatch.setattr(driver.Sandbox, "create", lambda *a, **kw: calls.append(kw))
    driver._create_sandbox("template", 60)
    driver._create_sandbox("template", 60, allow_internet_access=True)
    assert calls == [
        {"timeout": 60, "allow_internet_access": False},
        {"timeout": 60, "allow_internet_access": True},
    ]


@pytest.mark.parametrize(
    "failure", [None, "model", "interrupt", "patch", "cleanup", "create", "limit"]
)
def test_trajectory_survives_run_outcomes(tmp_path, instance, failure):
    path = tmp_path / "trace.traj.json"
    sandbox = FakeSandbox(
        rules=[
            ("git rev-parse HEAD", FakeCommandResult(stdout="abc123\n")),
            ("git -C /testbed diff", FakeCommandResult(stdout="PATCH\n")),
        ]
    )
    factory = SandboxFactory(sandbox)
    if failure == "create":
        factory.raises = RuntimeError("creation failed")
    if failure == "patch":
        sandbox.rules.insert(
            0,
            (
                "add --intent-to-add",
                FakeCommandResult(exit_code=1, stderr="capture failed"),
            ),
        )
    if failure == "cleanup":

        def kill():
            raise RuntimeError("cleanup failed")

        sandbox.kill = kill
    client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    tool_call("c1", "run", command="echo hello"),
                    tool_call("c2", "write_file", path="../bad", content="x"),
                ]
            ),
            FakeMessage(content="done"),
        ]
    )
    original = client.completions.create
    count = 0

    def create(**kwargs):
        nonlocal count
        count += 1
        saved = json.loads(path.read_text())
        if count == 1:
            assert [m["role"] for m in saved["messages"]] == ["system", "user"]
        if count == 2:
            assert [m["role"] for m in saved["messages"]] == [
                "system",
                "user",
                "assistant",
                "tool",
                "tool",
            ]
            if failure == "model":
                raise RuntimeError("model failed")
            if failure == "interrupt":
                raise KeyboardInterrupt
        return original(**kwargs)

    client.completions.create = create
    kwargs = {
        "create_sandbox": factory,
        "trajectory_path": path,
        "max_steps": 1 if failure == "limit" else 3,
    }
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            generate_prediction(instance, "template", client, **kwargs)
    else:
        generate_prediction(instance, "template", client, **kwargs)
    trace = json.loads(path.read_text())
    assert factory.calls[0]["allow_internet_access"] is False
    assert trace["info"]["allow_internet_access"] is False
    expected = {
        None: (2, "completed"),
        "model": (1, "error"),
        "interrupt": (1, "interrupted"),
        "patch": (2, "error"),
        "cleanup": (2, "completed"),
        "create": (0, "error"),
        "limit": (1, "step_limit"),
    }
    assert (trace["info"]["steps"], trace["info"]["stop_reason"]) == expected[failure]
    if failure != "create":
        assert "tool error" in trace["messages"][4]["content"]
    if failure == "cleanup":
        assert "cleanup failed" in trace["info"]["error"]
    elif failure != "create":
        assert sandbox.killed == 1


def test_write_failure_stops_before_tool_execution_and_preserves_json(
    tmp_path, instance, monkeypatch
):
    import e2b_swebench.agents.muse_spark as muse

    path = tmp_path / "trace.json"
    sandbox = FakeSandbox(default=FakeCommandResult(stdout="abc123"))
    client = FakeClient(
        [FakeMessage(tool_calls=[tool_call("c1", "run", command="touch forbidden")])]
    )
    replace = muse.os.replace

    def fail_on_assistant(src, dst):
        if json.loads(src.read_text())["info"]["steps"]:
            raise OSError("disk full")
        replace(src, dst)

    monkeypatch.setattr(muse.os, "replace", fail_on_assistant)
    with pytest.raises(TrajectoryWriteError):
        generate_prediction(
            instance,
            "t",
            client,
            create_sandbox=SandboxFactory(sandbox),
            trajectory_path=path,
        )
    assert json.loads(path.read_text())["messages"][-1]["role"] == "user"
    assert not any("touch forbidden" in c.command for c in sandbox.commands_run)
    assert sandbox.killed == 1


def test_trace_redacts_keys_from_error(tmp_path, instance, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-secret-meta")
    monkeypatch.setenv("E2B_API_KEY", "test-secret-e2b")
    path = tmp_path / "trace.json"
    generate_prediction(
        instance,
        "t",
        FakeClient(raises=RuntimeError("test-secret-meta test-secret-e2b")),
        create_sandbox=SandboxFactory(FakeSandbox()),
        trajectory_path=path,
    )
    assert "test-secret" not in path.read_text()
    assert "[REDACTED]" in path.read_text()


def test_initial_write_failure_creates_no_sandbox(tmp_path, instance):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("keep")
    factory = SandboxFactory(FakeSandbox())
    with pytest.raises(TrajectoryWriteError):
        generate_prediction(
            instance,
            "t",
            FakeClient(),
            create_sandbox=factory,
            trajectory_path=blocked / "trace.json",
        )
    assert factory.calls == []
    assert blocked.read_text() == "keep"
