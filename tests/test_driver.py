import json
import unittest
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from swebench.harness.test_spec.test_spec import make_test_spec

from e2b_swebench.driver import (
    _eval_script_preserving_image_setup,
    run_instance,
    run_instance_async,
)

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/swebench-4.1.0-eval-commands.json").read_text()
)
_TEST_PATCH = """diff --git a/testing/test_example.py b/testing/test_example.py
index 1111111..2222222 100644
--- a/testing/test_example.py
+++ b/testing/test_example.py
@@ -1 +1 @@
-old
+new
"""


def _make_spec(case_name: str):
    case = _FIXTURE[case_name]
    test_patch = "" if case_name == "whole_repository" else _TEST_PATCH
    return make_test_spec(
        {
            "instance_id": f"{case['repo'].replace('/', '__')}-synthetic",
            "repo": case["repo"],
            "version": case["version"],
            "base_commit": case["base_commit"],
            "problem_statement": "synthetic fixture",
            "hints_text": "",
            "test_patch": test_patch,
            "PASS_TO_PASS": [],
            "FAIL_TO_PASS": [],
        },
        namespace="swebench",
        arch="x86_64",
    )


class _SyncFiles:
    def __init__(self):
        self.writes = {}

    def write(self, path, content, **_kwargs):
        self.writes[path] = content

    def read(self, _path, **_kwargs):
        return ""


class _SyncSandbox:
    def __init__(self):
        self.files = _SyncFiles()
        self.commands = SimpleNamespace(run=lambda *_args, **_kwargs: None)
        self.killed = False

    def get_metrics(self):
        return []

    def kill(self):
        self.killed = True


class _AsyncFiles:
    def __init__(self):
        self.writes = {}

    async def write(self, path, content, **_kwargs):
        self.writes[path] = content

    async def read(self, _path, **_kwargs):
        return ""


class _AsyncSandbox:
    def __init__(self):
        self.files = _AsyncFiles()
        self.commands = SimpleNamespace(run=AsyncMock(return_value=None))
        self.killed = False

    async def get_metrics(self):
        return []

    async def kill(self):
        self.killed = True


class EvalScriptRewriteTests(unittest.TestCase):
    def test_fixture_matches_pinned_swebench_generator_commands(self):
        self.assertEqual(_FIXTURE["generator"], f"swebench=={version('swebench')}")

        for case_name in ("whole_repository", "path_specific"):
            with self.subTest(case=case_name):
                case = _FIXTURE[case_name]
                commands = _make_spec(case_name).eval_script_list
                self.assertEqual(commands[10], case["install"])
                self.assertEqual(commands[11], case["checkout"])
                self.assertTrue(commands[12].startswith(case["apply_prefix"]))
                self.assertEqual(commands[14], case["test"])
                self.assertEqual(commands[16], case["checkout"])

    def test_removes_whitespace_variant_of_generated_whole_repo_checkout(self):
        case = _FIXTURE["whole_repository"]
        spec = _make_spec("whole_repository")
        padded_checkout = f" \tgit   checkout\t{case['base_commit']} \t"
        spec.eval_script_list[11] = padded_checkout
        spec.eval_script_list[16] = padded_checkout

        result = _eval_script_preserving_image_setup(
            spec, {"base_commit": case["base_commit"]}
        )

        self.assertNotIn(padded_checkout, result)
        self.assertEqual(result.count("preserve SWE-bench image setup commit"), 2)
        self.assertIn(f"git -c core.fileMode=false diff {case['base_commit']}", result)
        self.assertLess(
            result.index(case["install"]), result.index(case["apply_prefix"])
        )
        self.assertLess(result.index(case["apply_prefix"]), result.index(case["test"]))

    def test_removes_only_whole_repo_reset_to_base_commit(self):
        case = _FIXTURE["whole_repository"]
        spec = _make_spec("whole_repository")
        spec.eval_script_list[11] = f"  {case['reset']}  "
        spec.eval_script_list[16] = f"  {case['reset']}  "

        result = _eval_script_preserving_image_setup(
            spec, {"base_commit": case["base_commit"]}
        )

        self.assertNotIn(case["reset"], result)
        self.assertEqual(result.count("preserve SWE-bench image setup commit"), 2)

    def test_preserves_generated_path_checkout_patch_and_test_order(self):
        case = _FIXTURE["path_specific"]
        spec = _make_spec("path_specific")

        result = _eval_script_preserving_image_setup(
            spec, {"base_commit": case["base_commit"]}
        )

        self.assertEqual(result.count(case["checkout"]), 2)
        self.assertNotIn("preserve SWE-bench image setup commit", result)
        self.assertLess(result.index(case["install"]), result.index(case["checkout"]))
        self.assertLess(
            result.index(case["checkout"]), result.index(case["apply_prefix"])
        )
        self.assertIn(_TEST_PATCH.strip(), result)
        self.assertLess(result.index(case["apply_prefix"]), result.index(case["test"]))
        self.assertLess(result.index(case["test"]), result.rindex(case["checkout"]))

    def test_preserves_path_checkout_with_separator_and_unknown_forms(self):
        case = _FIXTURE["path_specific"]
        spec = _make_spec("path_specific")
        spec.eval_script_list[11] = case["checkout_with_separator"]
        spec.eval_script_list[16] = case["checkout_with_separator"]
        unknown = f"git switch --detach {case['base_commit']}"
        spec.eval_script_list.insert(11, unknown)

        result = _eval_script_preserving_image_setup(
            spec, {"base_commit": case["base_commit"]}
        )

        self.assertEqual(result.count(case["checkout_with_separator"]), 2)
        self.assertIn(unknown, result)

    def test_does_not_remove_a_whole_repo_checkout_for_another_commit(self):
        base_commit = _FIXTURE["whole_repository"]["base_commit"]
        other_commit = "c" * 40
        command = f"git checkout {other_commit}"
        spec = SimpleNamespace(
            eval_script_list=[command],
            eval_script=f"#!/bin/bash\n{command}\n",
        )

        result = _eval_script_preserving_image_setup(spec, {"base_commit": base_commit})

        self.assertEqual(result, f"#!/bin/bash\ngit checkout {other_commit}\n")

    @patch("e2b_swebench.driver._grade", return_value={"resolved": True})
    @patch("e2b_swebench.driver._create_sandbox")
    @patch("e2b_swebench.driver.make_test_spec")
    def test_sync_driver_writes_repaired_script(
        self, make_spec, create_sandbox, _grade
    ):
        case = _FIXTURE["whole_repository"]
        spec = _make_spec("whole_repository")
        sandbox = _SyncSandbox()
        make_spec.return_value = spec
        create_sandbox.return_value = sandbox

        run_instance(
            {"instance_id": spec.instance_id, "base_commit": case["base_commit"]},
            {"instance_id": spec.instance_id, "model_patch": ""},
            "content-addressed-template",
        )

        written_script = sandbox.files.writes["/eval.sh"]
        self.assertNotIn(case["checkout"], written_script)
        self.assertEqual(
            written_script.count("preserve SWE-bench image setup commit"), 2
        )
        self.assertTrue(sandbox.killed)


class AsyncEvalScriptRewriteTests(unittest.IsolatedAsyncioTestCase):
    @patch("e2b_swebench.driver._grade", return_value={"resolved": True})
    @patch("e2b_swebench.driver._create_sandbox_async", new_callable=AsyncMock)
    @patch("e2b_swebench.driver.make_test_spec")
    async def test_async_driver_writes_repaired_script(
        self, make_spec, create_sandbox, _grade
    ):
        case = _FIXTURE["whole_repository"]
        spec = _make_spec("whole_repository")
        sandbox = _AsyncSandbox()
        make_spec.return_value = spec
        create_sandbox.return_value = sandbox

        await run_instance_async(
            {"instance_id": spec.instance_id, "base_commit": case["base_commit"]},
            {"instance_id": spec.instance_id, "model_patch": ""},
            "content-addressed-template",
        )

        written_script = sandbox.files.writes["/eval.sh"]
        self.assertNotIn(case["checkout"], written_script)
        self.assertEqual(
            written_script.count("preserve SWE-bench image setup commit"), 2
        )
        self.assertTrue(sandbox.killed)


if __name__ == "__main__":
    unittest.main()
