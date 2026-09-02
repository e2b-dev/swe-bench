"""Central configuration. Override via env vars where useful."""

import json
import os
from pathlib import Path


def configure_e2b_auth() -> None:
    """Reuse the E2B CLI team key when the SDK env var is not already set.

    ``e2b auth login`` stores the active team's key in ~/.e2b/config.json, while
    the Python SDK normally only checks E2B_API_KEY. Reading the CLI config keeps
    one credential source of truth and avoids copying a secret into this repo.
    """
    if os.environ.get("E2B_API_KEY"):
        return
    config_path = Path.home() / ".e2b" / "config.json"
    try:
        key = json.loads(config_path.read_text()).get("teamApiKey")
    except (OSError, json.JSONDecodeError):
        return
    if key:
        os.environ["E2B_API_KEY"] = key


configure_e2b_auth()

# --- dataset ---
DATASET = os.environ.get("SWEBENCH_DATASET", "princeton-nlp/SWE-bench_Verified")
SPLIT = os.environ.get("SWEBENCH_SPLIT", "test")

# --- image selection ---
# namespace="swebench" => use the prebuilt per-instance images on Docker Hub
# (swebench/sweb.eval.<arch>.<instance_id>). arch must be x86_64 for E2B (amd64).
NAMESPACE = os.environ.get("SWEBENCH_NAMESPACE", "swebench")
ARCH = "x86_64"

# --- template build resources ---
DEFAULT_CPU = int(os.environ.get("SWEBENCH_CPU", "4"))
DEFAULT_MEMORY_MB = int(os.environ.get("SWEBENCH_MEMORY_MB", "4096"))  # must be even
TEMPLATE_PREFIX = "swebench-"
TEMPLATE_WORKDIR = "/testbed"
TEMPLATE_CONSTRUCTION_SCHEMA = 1

# --- runtime timeouts (seconds) ---
SANDBOX_TIMEOUT = int(
    os.environ.get("SWEBENCH_SANDBOX_TIMEOUT", "2400")
)  # whole sandbox life
CMD_TIMEOUT = int(os.environ.get("SWEBENCH_CMD_TIMEOUT", "1800"))  # the eval.sh run
BUILD_RECOVERY_TIMEOUT = int(os.environ.get("SWEBENCH_BUILD_RECOVERY_TIMEOUT", "120"))

# --- concurrency: this account permits up to 100 concurrent sandboxes. The
# driver retries Sandbox.create on 429, so short-lived account-wide contention
# does not invalidate a task. Override with SWEBENCH_CONCURRENCY when a provider
# or another workload needs a lower cap. ---
DEFAULT_CONCURRENCY = int(os.environ.get("SWEBENCH_CONCURRENCY", "100"))
