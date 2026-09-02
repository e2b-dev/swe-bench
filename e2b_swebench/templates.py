"""Strategy A: one E2B template per instance, built FROM the prebuilt
swebench/sweb.eval.x86_64.<instance_id> Docker Hub image.

The image already has /testbed checked out at base_commit and the conda env
`testbed` installed, so the sandbox spawns ready-to-eval with everything on
the local microVM disk (fast). Builds run server-side on E2B — no local Docker.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

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
    TEMPLATE_CONSTRUCTION_SCHEMA,
    TEMPLATE_PREFIX,
    TEMPLATE_WORKDIR,
)

_BUILD_RECOVERY_INTERVAL = 10
_BUILD_RETRIES = 2
_CONTENT_KEY_SUFFIX_LENGTH = 24
_DOCKER_HUB_REGISTRIES = {
    "docker.io",
    "index.docker.io",
    "registry-1.docker.io",
}
_MANIFEST_ACCEPT = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.docker.distribution.manifest.v2+json"
)
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class TemplateSpec:
    """Every input that determines an E2B template's immutable contents."""

    instance_id: str
    source_image: str
    workdir: str = TEMPLATE_WORKDIR
    architecture: str = ARCH
    namespace: str = NAMESPACE
    cpu_count: int = DEFAULT_CPU
    memory_mb: int = DEFAULT_MEMORY_MB
    construction_schema: int = TEMPLATE_CONSTRUCTION_SCHEMA

    def __post_init__(self) -> None:
        _, separator, digest = self.source_image.rpartition("@")
        if not separator or not _SHA256_DIGEST.fullmatch(digest):
            raise ValueError(
                "source_image must be pinned to a lowercase SHA-256 digest"
            )
        if not isinstance(self.construction_schema, int):
            raise TypeError("construction_schema must be an integer")


def _open_registry(request: Request):
    return urlopen(request, timeout=30)


def _docker_hub_reference(image: str) -> tuple[str, str, str]:
    """Return (display repository, registry repository, tag)."""
    first_component, separator, remainder = image.partition("/")
    if separator and (
        "." in first_component
        or ":" in first_component
        or first_component == "localhost"
    ):
        if first_component not in _DOCKER_HUB_REGISTRIES:
            raise ValueError(f"only Docker Hub images are supported: {image!r}")
        display_repository = remainder
    else:
        display_repository = image

    last_slash = display_repository.rfind("/")
    last_colon = display_repository.rfind(":")
    if last_colon > last_slash:
        display_repository, tag = (
            display_repository[:last_colon],
            display_repository[last_colon + 1 :],
        )
    else:
        tag = "latest"
    if not display_repository or not tag:
        raise ValueError(f"invalid Docker Hub image reference: {image!r}")

    registry_repository = (
        display_repository
        if "/" in display_repository
        else f"library/{display_repository}"
    )
    return display_repository, registry_repository, tag


def resolve_image(image: str) -> str:
    """Resolve a Docker Hub tag to a repository reference pinned by digest."""
    repository, separator, digest = image.rpartition("@")
    if separator:
        if not repository or not _SHA256_DIGEST.fullmatch(digest):
            raise ValueError(f"invalid SHA-256 image reference: {image!r}")
        return image

    display_repository, registry_repository, tag = _docker_hub_reference(image)
    token_query = urlencode(
        {
            "service": "registry.docker.io",
            "scope": f"repository:{registry_repository}:pull",
        }
    )
    token_request = Request(f"https://auth.docker.io/token?{token_query}")
    with _open_registry(token_request) as response:
        token_payload = json.load(response)
    token = token_payload.get("token") or token_payload.get("access_token")
    if not token:
        raise ValueError("Docker Hub token response did not include an access token")

    manifest_url = (
        "https://registry-1.docker.io/v2/"
        f"{quote(registry_repository, safe='/')}/manifests/{quote(tag, safe='')}"
    )
    manifest_request = Request(
        manifest_url,
        headers={"Accept": _MANIFEST_ACCEPT, "Authorization": f"Bearer {token}"},
        method="HEAD",
    )
    with _open_registry(manifest_request) as response:
        resolved_digest = response.headers.get("Docker-Content-Digest", "").lower()
    if not _SHA256_DIGEST.fullmatch(resolved_digest):
        raise ValueError(
            "Docker Hub manifest response did not include a SHA-256 digest"
        )
    return f"{display_repository}@{resolved_digest}"


def content_key(spec: TemplateSpec) -> str:
    """Hash the canonical, complete template construction contract."""
    canonical = {
        "architecture": spec.architecture,
        "construction_schema": spec.construction_schema,
        "cpu_count": spec.cpu_count,
        "memory_mb": spec.memory_mb,
        "namespace": spec.namespace,
        "source_image": spec.source_image,
        "workdir": spec.workdir,
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def template_name_from_spec(spec: TemplateSpec) -> str:
    """Return the only alias eligible for reuse for this complete spec."""
    suffix = content_key(spec)[:_CONTENT_KEY_SUFFIX_LENGTH]
    slug = re.sub(r"[^a-z0-9]+", "-", spec.instance_id.lower()).strip("-")
    slug_room = 63 - len(TEMPLATE_PREFIX) - len(suffix) - 1
    slug = slug[:slug_room].rstrip("-") or "template"[:slug_room]
    return f"{TEMPLATE_PREFIX}{slug}-{suffix}"


def template_spec(
    instance: dict,
    cpu_count: int = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> TemplateSpec:
    """Resolve an instance's mutable source and return its immutable spec."""
    return TemplateSpec(
        instance_id=instance["instance_id"],
        source_image=resolve_image(instance_image(instance)),
        cpu_count=cpu_count,
        memory_mb=memory_mb,
    )


def template_name(
    instance: dict,
    cpu_count: int = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> str:
    """Resolve an instance and derive its current immutable template alias."""
    return template_name_from_spec(template_spec(instance, cpu_count, memory_mb))


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

    spec = template_spec(instance, cpu_count, memory_mb)
    name = template_name_from_spec(spec)
    if not force and template_ready(name):
        return name, False

    builder = Template().from_image(spec.source_image).set_workdir(spec.workdir)
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
