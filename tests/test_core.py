import asyncio
import tempfile
import unittest
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from e2b import BuildException, SandboxException

from e2b_swebench.driver import _eval_script_preserving_image_setup
from e2b_swebench.ledger import Ledger, categorize_verdict
from e2b_swebench.metrics import summarize_metrics
from e2b_swebench.runner import _run_one
from e2b_swebench.templates import (
    TemplateSpec,
    content_key,
    ensure_template,
    resolve_image,
    template_name_from_spec,
    template_ready,
)


class _RegistryResponse(BytesIO):
    def __init__(self, body=b"", headers=None):
        super().__init__(body)
        self.headers = headers or {}


class TemplateIdentityTests(unittest.TestCase):
    def setUp(self):
        self.spec = TemplateSpec(
            instance_id="astropy__astropy-12907",
            source_image="example/image@sha256:" + "a" * 64,
        )

    @patch("e2b_swebench.templates._open_registry")
    def test_resolves_docker_hub_tag_to_digest(self, open_registry):
        open_registry.side_effect = [
            _RegistryResponse(b'{"token":"registry-token"}'),
            _RegistryResponse(headers={"Docker-Content-Digest": "sha256:" + "a" * 64}),
        ]

        self.assertEqual(
            resolve_image("example/image:latest"),
            "example/image@sha256:" + "a" * 64,
        )

    def test_content_key_and_alias_use_canonical_complete_spec(self):
        self.assertEqual(
            content_key(self.spec),
            "88f24c7ae103d6e3c318f1dab434ce4c556f6b6f56f83ab4472da90bb9a65239",
        )
        self.assertEqual(content_key(self.spec), content_key(self.spec))

        mutations = {
            "digest": replace(
                self.spec, source_image="example/image@sha256:" + "b" * 64
            ),
            "workdir": replace(self.spec, workdir="/other"),
            "cpu": replace(self.spec, cpu_count=8),
            "memory": replace(self.spec, memory_mb=8192),
            "schema": replace(self.spec, construction_schema=2),
            "architecture": replace(self.spec, architecture="arm64"),
            "namespace": replace(self.spec, namespace="other"),
        }
        for field, changed_spec in mutations.items():
            with self.subTest(field=field):
                self.assertNotEqual(content_key(self.spec), content_key(changed_spec))
                self.assertNotEqual(
                    template_name_from_spec(self.spec),
                    template_name_from_spec(changed_spec),
                )

    def test_alias_is_bounded_and_changes_with_content(self):
        name = template_name_from_spec(
            replace(self.spec, instance_id="owner__" + "very-long-repository-" * 8)
        )
        self.assertLessEqual(len(name), 63)
        self.assertRegex(name, r"^[a-z0-9-]+$")

    @patch("e2b_swebench.templates._wait_until_ready", return_value=True)
    @patch("e2b_swebench.templates.template_ready")
    @patch("e2b_swebench.templates.instance_image", return_value="example/image:latest")
    @patch("e2b_swebench.templates.resolve_image", create=True)
    @patch("e2b_swebench.templates.Template")
    def test_resolves_image_before_content_alias_reuse_check(
        self, template, resolve_image, image, ready, wait
    ):
        events = []
        pinned_image = "example/image@sha256:" + "a" * 64
        resolve_image.side_effect = lambda source: (
            events.append("resolve") or pinned_image
        )
        ready.side_effect = lambda name: events.append(("ready", name)) or False

        name, built = ensure_template({"instance_id": "task"})

        self.assertEqual(name, "swebench-task-88f24c7ae103d6e3c318f1da")
        self.assertTrue(built)
        self.assertEqual(events[0], "resolve")
        self.assertEqual(
            {event[1] for event in events if isinstance(event, tuple)}, {name}
        )
        self.assertNotIn("swebench-task-4c-4096m", events)
        template.return_value.from_image.assert_called_once_with(pinned_image)

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
    @patch(
        "e2b_swebench.templates.resolve_image",
        return_value="example/image@sha256:" + "a" * 64,
    )
    @patch("e2b_swebench.templates.Template")
    def test_unspawnable_existing_alias_is_rebuilt(
        self, template, resolve, image, ready, wait
    ):
        name, built = ensure_template({"instance_id": "task"})
        self.assertEqual(name, "swebench-task-88f24c7ae103d6e3c318f1da")
        self.assertTrue(built)
        template.build.assert_called_once()
        wait.assert_called_once_with(name)

    @patch("e2b_swebench.templates.template_ready", side_effect=[False, True])
    @patch("e2b_swebench.templates.instance_image", return_value="example/image:latest")
    @patch(
        "e2b_swebench.templates.resolve_image",
        return_value="example/image@sha256:" + "a" * 64,
    )
    @patch("e2b_swebench.templates.Template")
    def test_internal_build_error_recovers_when_template_is_spawnable(
        self, template, resolve, image, ready
    ):
        template.build.side_effect = BuildException("internal error")
        name, built = ensure_template({"instance_id": "task"})
        self.assertEqual(name, "swebench-task-88f24c7ae103d6e3c318f1da")
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


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    @patch("e2b_swebench.runner.run_instance_async", new_callable=AsyncMock)
    @patch("e2b_swebench.templates.instance_image", return_value="example/image:latest")
    @patch(
        "e2b_swebench.templates.resolve_image",
        return_value="example/image@sha256:" + "a" * 64,
    )
    async def test_runtime_selects_content_alias_without_building(
        self, resolve, image, run_instance
    ):
        instance = {"instance_id": "task"}
        prediction = {"instance_id": "task", "model_patch": ""}
        run_instance.return_value = {"resolved": True}

        verdict = await _run_one(asyncio.Semaphore(1), instance, prediction)

        self.assertTrue(verdict["resolved"])
        run_instance.assert_awaited_once_with(
            instance,
            prediction,
            "swebench-task-88f24c7ae103d6e3c318f1da",
        )


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

    def test_arbitrary_p2p_failure_is_not_an_artifact(self):
        verdict = {
            "instance_id": "astropy__astropy-7606",
            "patch_successfully_applied": True,
            "tests_status": {
                "FAIL_TO_PASS": {"failure": []},
                "PASS_TO_PASS": {"failure": ["test_regression"]},
            },
        }
        self.assertEqual(categorize_verdict(verdict)[0], "fail")

    def test_artifact_requires_literal_true_patch_flag(self):
        verdict = {
            "instance_id": "astropy__astropy-7606",
            "tests_status": {
                "FAIL_TO_PASS": {"failure": []},
                "PASS_TO_PASS": {
                    "failure": [
                        "astropy/units/tests/test_units.py::test_compose_roundtrip[]"
                    ]
                },
            },
        }

        for patch_applied in (1, "false"):
            with self.subTest(patch_applied=patch_applied):
                verdict["patch_successfully_applied"] = patch_applied
                self.assertEqual(categorize_verdict(verdict)[0], "fail")

    def test_malformed_failure_containers_are_not_artifacts(self):
        expected_failure = "astropy/units/tests/test_units.py::test_compose_roundtrip[]"
        malformed_failures = {
            "FAIL_TO_PASS": (
                ("mapping", {}),
                ("set", set()),
                ("none", None),
                ("integer", 1),
                ("non_string_list", [1]),
            ),
            "PASS_TO_PASS": (
                ("mapping", {expected_failure: True}),
                ("set", {expected_failure}),
                ("none", None),
                ("integer", 1),
                ("unhashable_member", [[]]),
                ("non_string_list", [1]),
            ),
        }

        for status_name, malformed_values in malformed_failures.items():
            for malformed_name, malformed_value in malformed_values:
                with self.subTest(
                    status_name=status_name, malformed_name=malformed_name
                ):
                    verdict = {
                        "instance_id": "astropy__astropy-7606",
                        "patch_successfully_applied": True,
                        "tests_status": {
                            "FAIL_TO_PASS": {"failure": []},
                            "PASS_TO_PASS": {"failure": [expected_failure]},
                        },
                    }
                    verdict["tests_status"][status_name]["failure"] = malformed_value
                    self.assertEqual(categorize_verdict(verdict)[0], "fail")

    def test_only_exact_audited_p2p_failure_signatures_are_artifacts(self):
        signatures = {
            "astropy__astropy-7606": (
                "grader_artifact",
                {
                    "astropy/units/tests/test_units.py::test_compose_roundtrip[]",
                },
            ),
            "django__django-10097": (
                "ordering_artifact",
                {
                    "test_add (generic_inline_admin.tests.GenericInlineAdminWithUniqueTogetherTest)",
                    "test_delete (generic_inline_admin.tests.GenericInlineAdminWithUniqueTogetherTest)",
                    "test_no_param (generic_inline_admin.tests.GenericInlineAdminParametersTest)",
                    "test_basic_add_GET (generic_inline_admin.tests.GenericAdminViewTest)",
                    "test_basic_edit_GET (generic_inline_admin.tests.GenericAdminViewTest)",
                },
            ),
        }

        for instance_id, (expected_category, failed_tests) in signatures.items():
            with self.subTest(instance_id=instance_id, mutation="exact"):
                verdict = {
                    "instance_id": instance_id,
                    "patch_successfully_applied": True,
                    "tests_status": {
                        "FAIL_TO_PASS": {"failure": []},
                        "PASS_TO_PASS": {"failure": list(failed_tests)},
                    },
                }
                self.assertEqual(categorize_verdict(verdict)[0], expected_category)

            mutated_signatures = {
                "wrong": (failed_tests - {next(iter(failed_tests))}) | {"wrong_test"},
                "missing": failed_tests - {next(iter(failed_tests))},
                "extra": failed_tests | {"extra_test"},
                "duplicate": [*failed_tests, next(iter(failed_tests))],
            }
            for mutation, mutated_failed_tests in mutated_signatures.items():
                with self.subTest(instance_id=instance_id, mutation=mutation):
                    verdict["tests_status"]["PASS_TO_PASS"]["failure"] = list(
                        mutated_failed_tests
                    )
                    self.assertEqual(categorize_verdict(verdict)[0], "fail")

    def test_environment_failure_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(f"{directory}/ledger.json")
            ledger.update(
                "task", verify="collection_error", cpu_count=4, memory_mb=2048
            )
            self.assertFalse(ledger.is_done("task", 4, 2048))


if __name__ == "__main__":
    unittest.main()
