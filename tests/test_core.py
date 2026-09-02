import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from e2b import BuildException, SandboxException

from e2b_swebench.driver import _eval_script_preserving_image_setup
from e2b_swebench.ledger import Ledger, categorize_verdict
from e2b_swebench.metrics import summarize_metrics
from e2b_swebench.templates import ensure_template, template_name, template_ready


class TemplateNameTests(unittest.TestCase):
    def test_resources_are_part_of_template_name(self):
        small = template_name("astropy__astropy-12907", 2, 4096)
        large = template_name("astropy__astropy-12907", 8, 16384)
        self.assertEqual(small, "swebench-astropy-astropy-12907-2c-4096m")
        self.assertNotEqual(small, large)

    @patch("e2b_swebench.templates.Template")
    def test_alias_without_default_tag_is_not_ready(self, template):
        template.exists.return_value = True
        template.get_tags.return_value = [SimpleNamespace(tag="staging")]
        self.assertFalse(template_ready("incomplete"))

    @patch("e2b_swebench.templates.Sandbox")
    @patch("e2b_swebench.templates.Template")
    def test_default_tag_that_cannot_spawn_is_not_ready(self, template, sandbox):
        template.exists.return_value = True
        template.get_tags.return_value = [SimpleNamespace(tag="default")]
        sandbox.create.side_effect = SandboxException("tag not found")
        self.assertFalse(template_ready("incomplete"))

    @patch("e2b_swebench.templates._wait_until_ready", return_value=True)
    @patch("e2b_swebench.templates.template_ready", return_value=False)
    @patch("e2b_swebench.templates.instance_image", return_value="example/image:latest")
    @patch("e2b_swebench.templates.Template")
    def test_unspawnable_existing_alias_is_rebuilt(self, template, image, ready, wait):
        name, built = ensure_template({"instance_id": "task"})
        self.assertEqual(name, "swebench-task-4c-4096m")
        self.assertTrue(built)
        template.build.assert_called_once()
        wait.assert_called_once_with(name)

    @patch("e2b_swebench.templates.template_ready", side_effect=[False, True])
    @patch("e2b_swebench.templates.instance_image", return_value="example/image:latest")
    @patch("e2b_swebench.templates.Template")
    def test_internal_build_error_recovers_when_template_is_spawnable(
        self, template, image, ready
    ):
        template.build.side_effect = BuildException("internal error")
        name, built = ensure_template({"instance_id": "task"})
        self.assertEqual(name, "swebench-task-4c-4096m")
        self.assertTrue(built)
        template.build.assert_called_once()


class MetricsTests(unittest.TestCase):
    def test_cache_is_excluded_from_working_set(self):
        samples = [
            SimpleNamespace(
                cpu_count=4,
                cpu_used_pct=50.0,
                mem_total=8 * 1024**3,
                mem_used=3 * 1024**3,
                mem_cache=2 * 1024**3,
                disk_total=100 * 1024**3,
                disk_used=10 * 1024**3,
            ),
            SimpleNamespace(
                cpu_count=4,
                cpu_used_pct=75.0,
                mem_total=8 * 1024**3,
                mem_used=4 * 1024**3,
                mem_cache=2 * 1024**3,
                disk_total=100 * 1024**3,
                disk_used=11 * 1024**3,
            ),
        ]
        result = summarize_metrics(samples)
        self.assertEqual(result["peak_cpu_cores"], 3.0)
        self.assertEqual(result["peak_memory_used_mb"], 4096.0)
        self.assertEqual(result["peak_memory_working_set_mb"], 2048.0)


class DriverTests(unittest.TestCase):
    def test_eval_script_preserves_setup_commit(self):
        base = "a" * 40
        spec = SimpleNamespace(
            eval_script=(
                "git status\n"
                f"git checkout {base} \n"
                "git apply /tmp/tests.patch\n"
                f"git checkout {base}\n"
            )
        )
        result = _eval_script_preserving_image_setup(spec, {"base_commit": base})
        self.assertNotIn(f"git checkout {base}", result)
        self.assertIn("git apply /tmp/tests.patch", result)
        self.assertEqual(result.count("preserve SWE-bench image setup commit"), 2)


class LedgerTests(unittest.TestCase):
    def test_done_is_resource_profile_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(f"{directory}/ledger.json")
            ledger.update("task", verify="pass", cpu_count=4, memory_mb=8192)
            self.assertTrue(ledger.is_done("task", 4, 8192))
            self.assertFalse(ledger.is_done("task", 2, 4096))

    def test_only_audited_p2p_failure_is_an_artifact(self):
        verdict = {
            "instance_id": "new__case-1",
            "patch_successfully_applied": True,
            "tests_status": {
                "FAIL_TO_PASS": {"failure": []},
                "PASS_TO_PASS": {"failure": ["test_regression"]},
            },
        }
        self.assertEqual(categorize_verdict(verdict)[0], "fail")

        verdict["instance_id"] = "astropy__astropy-7606"
        category, detail = categorize_verdict(verdict)
        self.assertEqual(category, "grader_artifact")
        self.assertIn("unit0", detail["artifact_reason"])

    def test_environment_failure_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(f"{directory}/ledger.json")
            ledger.update(
                "task", verify="collection_error", cpu_count=4, memory_mb=2048
            )
            self.assertFalse(ledger.is_done("task", 4, 2048))


if __name__ == "__main__":
    unittest.main()
