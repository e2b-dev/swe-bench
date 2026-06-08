"""Strategy A: one E2B template per instance, built FROM the prebuilt
swebench/sweb.eval.x86_64.<instance_id> Docker Hub image.

The image already has /testbed checked out at base_commit and the conda env
`testbed` installed, so the sandbox spawns ready-to-eval with everything on
the local microVM disk (fast). Builds run server-side on E2B — no local Docker.
"""

import re

from e2b import Template, default_build_logger
from swebench.harness.test_spec.test_spec import make_test_spec

from .config import ARCH, DEFAULT_CPU, DEFAULT_MEMORY_MB, NAMESPACE, TEMPLATE_PREFIX


def template_name(instance_id: str) -> str:
    """E2B template names are lowercase [a-z0-9-]. e.g.
    'astropy__astropy-12907' -> 'swebench-astropy-astropy-12907'."""
    slug = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    return f"{TEMPLATE_PREFIX}{slug}"


def instance_image(instance: dict) -> str:
    """The Docker Hub image key for this instance, e.g.
    'swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest'."""
    return make_test_spec(instance, namespace=NAMESPACE, arch=ARCH).instance_image_key


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

    name = template_name(instance["instance_id"])
    if not force and Template.exists(name):
        return name, False

    image = instance_image(instance)
    builder = Template().from_image(image).set_workdir("/testbed")
    Template.build(
        builder,
        name,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        skip_cache=force,
        on_build_logs=None if quiet else default_build_logger(),
    )
    return name, True


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
                _record(i, inst["instance_id"], ensure_template(
                    inst, cpu_count=cpu_count, memory_mb=memory_mb, force=force, quiet=True))
            except Exception as e:  # noqa: BLE001
                _record(i, inst["instance_id"], e)
        return results

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(ensure_template, inst, cpu_count, memory_mb, force, True): inst["instance_id"]
            for inst in instances_list
        }
        for i, fut in enumerate(as_completed(futs), 1):
            iid = futs[fut]
            try:
                _record(i, iid, fut.result())
            except Exception as e:  # noqa: BLE001
                _record(i, iid, e)
    return results
