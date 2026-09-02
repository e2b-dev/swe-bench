"""Strategy A: one E2B template per instance, built FROM the prebuilt
swebench/sweb.eval.x86_64.<instance_id> Docker Hub image.

The image already has /testbed checked out at base_commit and the conda env
`testbed` installed, so the sandbox spawns ready-to-eval with everything on
the local microVM disk (fast). Builds run server-side on E2B — no local Docker.
"""

import re
import time

from e2b import (
    BuildException,
    RateLimitException,
    Sandbox,
    SandboxException,
    Template,
    default_build_logger,
)
from swebench.harness.test_spec.test_spec import make_test_spec

from .config import (
    ARCH,
    BUILD_RECOVERY_TIMEOUT,
    DEFAULT_CPU,
    DEFAULT_MEMORY_MB,
    NAMESPACE,
    TEMPLATE_PREFIX,
)

_BUILD_RECOVERY_INTERVAL = 10
_BUILD_RETRIES = 2


def template_name(
    instance_id: str,
    cpu_count: int = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> str:
    """E2B template names are lowercase [a-z0-9-]. e.g.
    'astropy__astropy-12907' at 4 CPU / 8 GiB becomes
    'swebench-astropy-astropy-12907-4c-8192m'.

    Resources are part of the name because E2B bakes them into the template.
    Without the suffix, a custom-size run can silently reuse a template built
    with different resources.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    return f"{TEMPLATE_PREFIX}{slug}-{cpu_count}c-{memory_mb}m"


def instance_image(instance: dict) -> str:
    """The Docker Hub image key for this instance, e.g.
    'swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest'."""
    return make_test_spec(instance, namespace=NAMESPACE, arch=ARCH).instance_image_key


def template_ready(name: str) -> bool:
    """Return whether the template can create a sandbox from ``default``.

    ``Template.exists`` only checks that an alias row exists. An E2B build that
    reports a transient internal error can leave that alias visible before its
    default tag is usable. Even ``Template.get_tags`` can expose ``default``
    while sandbox creation still returns a tag-not-found error, so a short-lived
    probe sandbox is the authoritative readiness check.
    """
    if not Template.exists(name):
        return False
    if not any(tag.tag == "default" for tag in Template.get_tags(name)):
        return False
    for attempt in range(5):
        try:
            sandbox = Sandbox.create(name, timeout=60)
            try:
                return True
            finally:
                sandbox.kill()
        except RateLimitException:
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))
        except SandboxException:
            return False
    return False


def _wait_until_ready(name: str, timeout: int = BUILD_RECOVERY_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if template_ready(name):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_BUILD_RECOVERY_INTERVAL, remaining))


def ensure_template(
    instance: dict,
    cpu_count: int = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
    force: bool = False,
    quiet: bool = False,
) -> tuple[str, bool]:
    """Build the template if it doesn't already exist (lazy). Returns (name, built).

    Lazy build means your template count tracks what you actually evaluate:
    1 for the smoke test, N for an N-instance run.
    """
    # Runs in ProcessPoolExecutor children too, which don't inherit the parent's
    # logging config — quiet the per-request SDK/HTTP spam here so build logs stay readable.
    from .logs import quiet_logs

    quiet_logs()

    name = template_name(instance["instance_id"], cpu_count, memory_mb)
    if not force and template_ready(name):
        return name, False

    image = instance_image(instance)
    builder = Template().from_image(image).set_workdir("/testbed")
    last_error = None
    for attempt in range(_BUILD_RETRIES + 1):
        try:
            Template.build(
                builder,
                name,
                cpu_count=cpu_count,
                memory_mb=memory_mb,
                skip_cache=force,
                on_build_logs=None if quiet else default_build_logger(),
            )
            if template_ready(name) or _wait_until_ready(name):
                return name, True
            last_error = BuildException(
                f"template build returned without publishing {name}:default"
            )
        except BuildException as exc:
            last_error = exc
            # A generic internal build error can leave an alias and even a
            # misleading default tag behind. Only a successful probe is enough.
            if template_ready(name):
                return name, True
        if attempt < _BUILD_RETRIES:
            time.sleep(_BUILD_RECOVERY_INTERVAL)
    raise last_error


def build_many(
    instances_list: list,
    workers: int = 4,
    cpu_count: int = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
    force: bool = False,
    progress: bool = True,
) -> dict:
    """Build templates for many instances in parallel. Returns
    {instance_id: (name, built) | Exception}.

    Uses a PROCESS pool, NOT threads. The E2B SDK shares a single HTTP/2
    connection; many concurrent build-status streams on one connection collide
    (RemoteProtocolError: invalid_new_stream_id / SEND_HEADERS in state 5). One
    process per worker = one connection per worker, which is safe. Each child
    inherits E2B_API_KEY from the environment.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    total = len(instances_list)
    results: dict = {}

    def _record(i, iid, outcome):
        results[iid] = outcome
        if not progress:
            return
        if isinstance(outcome, Exception):
            print(f"[{i}/{total}] {iid}: FAILED {outcome!r}", flush=True)
        else:
            name, built = outcome
            print(f"[{i}/{total}] {name}: {'built' if built else 'exists'}", flush=True)

    if workers <= 1:
        for i, inst in enumerate(instances_list, 1):
            try:
                _record(
                    i,
                    inst["instance_id"],
                    ensure_template(
                        inst,
                        cpu_count=cpu_count,
                        memory_mb=memory_mb,
                        force=force,
                        quiet=True,
                    ),
                )
            except Exception as e:  # noqa: BLE001
                _record(i, inst["instance_id"], e)
        return results

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(ensure_template, inst, cpu_count, memory_mb, force, True): inst[
                "instance_id"
            ]
            for inst in instances_list
        }
        for i, fut in enumerate(as_completed(futs), 1):
            iid = futs[fut]
            try:
                _record(i, iid, fut.result())
            except Exception as e:  # noqa: BLE001
                _record(i, iid, e)
    return results
