"""Tests for scripts/run_agent.py — selection, the agent registry, output files.

The script is loaded from its path (scripts/ is not a package). Instead of
monkeypatching module globals, these register a stub AgentSpec and select it
with `--agent`, so the registry indirection is itself under test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from e2b_swebench.agents import REGISTRY, AgentSpec, agent_names, get_agent
from e2b_swebench.agents.muse_spark import GenerationResult

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_agent.py"
STUB = "test-stub"


@pytest.fixture
def run_agent_script():
    spec = importlib.util.spec_from_file_location("run_agent_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _instances(*ids, repo="psf/requests"):
    return {
        iid: {
            "instance_id": iid,
            "repo": repo,
            "problem_statement": f"bug in {iid}",
            "FAIL_TO_PASS": "[]",
            "PASS_TO_PASS": "[]",
        }
        for iid in ids
    }


def _stub_spec(generate, *, default_model="stub-model-1.0", check_model=None):
    return AgentSpec(
        name=STUB,
        description="stub agent for tests",
        default_model=default_model,
        create_client=lambda: object(),
        check_model=check_model or (lambda client, model: None),
        generate_prediction=generate,
        empty_prediction=lambda iid, model: {
            "instance_id": iid,
            "model_name_or_path": model,
            "model_patch": "",
        },
        requires_env=(),
    )


def _wire(module, monkeypatch, instances, generate, **kw):
    monkeypatch.setattr(module, "load_instances", lambda: instances)
    monkeypatch.setattr(module, "check_templates", lambda ids: None)
    monkeypatch.setitem(REGISTRY, STUB, _stub_spec(generate, **kw))


def _ok_generator(model=None):
    def generate(instance, template, client, **kw):
        iid = instance["instance_id"]
        used = model or kw["model"]
        patch = f"diff --git a/{iid}.py b/{iid}.py\n"
        return (
            {"instance_id": iid, "model_name_or_path": used, "model_patch": patch},
            GenerationResult(
                instance_id=iid,
                model=used,
                status="ok",
                steps=2,
                patch_bytes=len(patch),
            ),
        )

    return generate


def _argv(monkeypatch, *extra):
    monkeypatch.setattr("sys.argv", ["run_agent.py", "--agent", STUB, *extra])


# --- the registry ------------------------------------------------------------


def test_muse_spark_is_registered_and_not_hardcoded():
    assert "muse-spark" in agent_names()
    spec = get_agent("muse-spark")
    assert spec.default_model == "muse-spark-1.1"
    assert spec.requires_env == ("META_API_KEY",)
    # the runner only ever needs these four callables
    for attr in (
        "create_client",
        "check_model",
        "generate_prediction",
        "empty_prediction",
    ):
        assert callable(getattr(spec, attr))


def test_an_unknown_agent_names_the_available_ones():
    with pytest.raises(KeyError) as excinfo:
        get_agent("gpt-nope")
    assert "muse-spark" in str(excinfo.value)


def test_registering_a_duplicate_name_is_refused():
    from e2b_swebench.agents import register

    clash = AgentSpec(
        name="muse-spark",
        description="an impostor",
        default_model="m",
        create_client=lambda: None,
        check_model=lambda c, m: None,
        generate_prediction=lambda *a, **k: None,
        empty_prediction=lambda i, m: {},
    )
    with pytest.raises(ValueError):
        register(clash)


def test_list_agents_prints_the_registry(run_agent_script, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_agent.py", "--list-agents"])
    assert run_agent_script.main() == 0
    out = capsys.readouterr().out
    assert "muse-spark" in out
    assert "muse-spark-1.1" in out
    assert "META_API_KEY" in out


# --- selection ---------------------------------------------------------------


def test_select_ids_supports_instances_limit_and_per_repo(run_agent_script):
    import argparse

    instances = {}
    instances.update(_instances("a__a-1", "a__a-2", repo="a/a"))
    instances.update(_instances("b__b-1", repo="b/b"))

    def args(**kw):
        fields = {"instances": None, "limit": None, "per_repo": None}
        fields.update(kw)
        return argparse.Namespace(**fields)

    select = run_agent_script.select_ids
    assert select(instances, args(instances="a__a-2, b__b-1")) == ["a__a-2", "b__b-1"]
    assert select(instances, args(limit=2)) == ["a__a-1", "a__a-2"]
    assert sorted(select(instances, args(per_repo=1))) == ["a__a-1", "b__b-1"]
    assert select(instances, args()) == []


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_cli_values_reject_zero_and_negative(run_agent_script, value):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        run_agent_script.positive_int(value)


def test_missing_selection_is_rejected_before_loading_the_dataset(
    run_agent_script, monkeypatch
):
    monkeypatch.setattr("sys.argv", ["run_agent.py"])
    monkeypatch.setattr(
        run_agent_script,
        "load_instances",
        lambda: (_ for _ in ()).throw(AssertionError("must not load")),
    )

    with pytest.raises(SystemExit):
        run_agent_script.main()


# --- output files ------------------------------------------------------------


def test_multiple_instances_write_stable_parseable_jsonl(
    run_agent_script, monkeypatch, tmp_path
):
    ids = ["a__a-1", "b__b-2", "c__c-3"]
    _wire(run_agent_script, monkeypatch, _instances(*ids), _ok_generator())

    out = tmp_path / "nested" / "predictions.jsonl"
    _argv(monkeypatch, "--instances", ",".join(ids), "--out", str(out))
    assert run_agent_script.main() == 0

    lines = out.read_text().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert [r["instance_id"] for r in records] == ids  # selection order preserved
    for record in records:
        assert set(record) == {"instance_id", "model_name_or_path", "model_patch"}

    status = (out.parent / "generation.jsonl").read_text().splitlines()
    assert len(status) == 3
    assert [json.loads(s)["status"] for s in status] == ["ok", "ok", "ok"]


def test_a_rerun_truncates_rather_than_appending(
    run_agent_script, monkeypatch, tmp_path
):
    _wire(run_agent_script, monkeypatch, _instances("a__a-1"), _ok_generator())
    out = tmp_path / "predictions.jsonl"
    out.write_text('{"stale": true}\n{"also": "stale"}\n')

    _argv(monkeypatch, "--instances", "a__a-1", "--out", str(out))
    run_agent_script.main()

    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["instance_id"] == "a__a-1"


def test_the_status_path_can_be_set_independently(
    run_agent_script, monkeypatch, tmp_path
):
    _wire(run_agent_script, monkeypatch, _instances("a__a-1"), _ok_generator())
    out = tmp_path / "p.jsonl"
    status = tmp_path / "elsewhere" / "s.jsonl"

    _argv(
        monkeypatch, "--instances", "a__a-1", "--out", str(out), "--status", str(status)
    )
    run_agent_script.main()

    assert json.loads(status.read_text().splitlines()[0])["instance_id"] == "a__a-1"
    assert not (tmp_path / "generation.jsonl").exists()


def test_prediction_and_status_outputs_must_be_different(
    run_agent_script, monkeypatch, tmp_path
):
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_agent.py",
            "--instances",
            "a__a-1",
            "--out",
            str(output),
            "--status",
            str(output),
        ],
    )

    with pytest.raises(SystemExit):
        run_agent_script.main()

    assert not output.exists()


# --- batch resilience --------------------------------------------------------


def test_one_instance_raising_does_not_sink_the_batch(
    run_agent_script, monkeypatch, tmp_path
):
    ids = ["a__a-1", "b__b-2", "c__c-3"]
    ok = _ok_generator()

    def generate(instance, template, client, **kw):
        if instance["instance_id"] == "b__b-2":
            raise RuntimeError("sandbox exploded")
        return ok(instance, template, client, **kw)

    _wire(run_agent_script, monkeypatch, _instances(*ids), generate)
    out = tmp_path / "predictions.jsonl"
    _argv(monkeypatch, "--instances", ",".join(ids), "--out", str(out))
    assert run_agent_script.main() == 0

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["instance_id"] for r in records] == ids  # every instance represented
    failed = next(r for r in records if r["instance_id"] == "b__b-2")
    assert failed["model_patch"] == ""
    assert set(failed) == {"instance_id", "model_name_or_path", "model_patch"}

    status = [
        json.loads(s) for s in (tmp_path / "generation.jsonl").read_text().splitlines()
    ]
    assert [s["status"] for s in status] == ["ok", "error", "ok"]
    assert "sandbox exploded" in status[1]["error"]


def test_a_fatal_failure_stops_the_batch_early(
    run_agent_script, monkeypatch, tmp_path, capsys
):
    """A bad key would otherwise burn a sandbox per instance for nothing."""
    ids = ["a__a-1", "b__b-2", "c__c-3"]
    seen: list[str] = []

    def generate(instance, template, client, **kw):
        iid = instance["instance_id"]
        seen.append(iid)
        return (
            {"instance_id": iid, "model_name_or_path": kw["model"], "model_patch": ""},
            GenerationResult(
                instance_id=iid,
                model=kw["model"],
                status="error",
                error="AuthenticationError(...)",
                fatal=True,
            ),
        )

    _wire(run_agent_script, monkeypatch, _instances(*ids), generate)
    out = tmp_path / "predictions.jsonl"
    _argv(monkeypatch, "--instances", ",".join(ids), "--out", str(out))

    assert run_agent_script.main() == 1  # non-zero: the run did not complete
    assert seen == ["a__a-1"]  # stopped after the first
    assert len(out.read_text().splitlines()) == 1
    output = capsys.readouterr().out
    assert "permanent" in output
    assert "run_eval.py" not in output  # no misleading next-step hint


def test_a_failed_model_preflight_aborts_before_any_sandbox(
    run_agent_script, monkeypatch, tmp_path, capsys
):
    from e2b_swebench.agents import ModelUnavailable

    called: list[str] = []

    def generate(*a, **kw):
        called.append("nope")
        raise AssertionError("must not run")

    def check_model(client, model):
        raise ModelUnavailable(model, RuntimeError("401 invalid_api_key"))

    _wire(
        run_agent_script,
        monkeypatch,
        _instances("a__a-1"),
        generate,
        check_model=check_model,
    )
    out = tmp_path / "predictions.jsonl"
    _argv(monkeypatch, "--instances", "a__a-1", "--out", str(out))

    assert run_agent_script.main() == 1
    assert called == []
    assert not out.exists()
    assert "invalid_api_key" in capsys.readouterr().out


def test_preflight_can_be_skipped(run_agent_script, monkeypatch, tmp_path):
    def check_model(client, model):
        raise AssertionError("preflight should have been skipped")

    _wire(
        run_agent_script,
        monkeypatch,
        _instances("a__a-1"),
        _ok_generator(),
        check_model=check_model,
    )
    out = tmp_path / "predictions.jsonl"
    _argv(monkeypatch, "--instances", "a__a-1", "--out", str(out), "--skip-preflight")

    assert run_agent_script.main() == 0


def test_missing_templates_abort_before_any_generation(
    run_agent_script, monkeypatch, tmp_path, capsys
):
    from e2b_swebench.agents.muse_spark import MissingTemplates

    called = []

    def generate(*a, **kw):
        called.append(a)
        raise AssertionError("must not run")

    _wire(run_agent_script, monkeypatch, _instances("a__a-1"), generate)

    def boom(ids):
        raise MissingTemplates(list(ids))

    monkeypatch.setattr(run_agent_script, "check_templates", boom)
    out = tmp_path / "predictions.jsonl"
    _argv(monkeypatch, "--instances", "a__a-1", "--out", str(out))

    assert run_agent_script.main() == 1
    assert called == []
    assert not out.exists()
    assert "build_templates.py" in capsys.readouterr().out


def test_ids_not_in_the_dataset_are_reported_and_skipped(
    run_agent_script, monkeypatch, tmp_path, capsys
):
    _wire(run_agent_script, monkeypatch, _instances("a__a-1"), _ok_generator())
    out = tmp_path / "predictions.jsonl"
    _argv(monkeypatch, "--instances", "a__a-1,nope__nope-9", "--out", str(out))
    assert run_agent_script.main() == 0

    assert "NOT IN DATASET: nope__nope-9" in capsys.readouterr().out
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["instance_id"] for r in records] == ["a__a-1"]


# --- model resolution --------------------------------------------------------


def test_the_model_defaults_to_the_selected_agents_default(
    run_agent_script, monkeypatch, tmp_path
):
    seen = {}

    def generate(instance, template, client, **kw):
        seen.update(kw)
        return _ok_generator()(instance, template, client, **kw)

    _wire(
        run_agent_script,
        monkeypatch,
        _instances("a__a-1"),
        generate,
        default_model="stub-model-9.9",
    )
    monkeypatch.delenv("META_MODEL", raising=False)
    out = tmp_path / "predictions.jsonl"
    _argv(monkeypatch, "--instances", "a__a-1", "--out", str(out))
    run_agent_script.main()

    assert seen["model"] == "stub-model-9.9"


def test_the_model_flag_overrides_and_lands_in_model_name_or_path(
    run_agent_script, monkeypatch, tmp_path
):
    seen = {}

    def generate(instance, template, client, **kw):
        seen.update(kw)
        return _ok_generator()(instance, template, client, **kw)

    _wire(run_agent_script, monkeypatch, _instances("a__a-1"), generate)
    out = tmp_path / "predictions.jsonl"
    _argv(
        monkeypatch,
        "--instances",
        "a__a-1",
        "--out",
        str(out),
        "--model",
        "muse-spark-1.1",
        "--max-steps",
        "7",
    )
    run_agent_script.main()

    assert seen["model"] == "muse-spark-1.1"
    assert seen["max_steps"] == 7
    record = json.loads(out.read_text().splitlines()[0])
    assert record["model_name_or_path"] == "muse-spark-1.1"


def test_meta_model_env_overrides_the_agent_default(
    run_agent_script, monkeypatch, tmp_path
):
    seen = {}

    def generate(instance, template, client, **kw):
        seen.update(kw)
        return _ok_generator()(instance, template, client, **kw)

    _wire(run_agent_script, monkeypatch, _instances("a__a-1"), generate)
    monkeypatch.setenv("META_MODEL", "muse-spark-from-env")
    out = tmp_path / "predictions.jsonl"
    _argv(monkeypatch, "--instances", "a__a-1", "--out", str(out))
    run_agent_script.main()

    assert seen["model"] == "muse-spark-from-env"
