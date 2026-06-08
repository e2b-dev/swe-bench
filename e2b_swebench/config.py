"""Central configuration. Override via env vars where useful."""

import os

# --- dataset ---
DATASET = os.environ.get("SWEBENCH_DATASET", "princeton-nlp/SWE-bench_Verified")
SPLIT = os.environ.get("SWEBENCH_SPLIT", "test")

# --- image selection ---
# namespace="swebench" => use the prebuilt per-instance images on Docker Hub
# (swebench/sweb.eval.<arch>.<instance_id>). arch must be x86_64 for E2B (amd64).
NAMESPACE = os.environ.get("SWEBENCH_NAMESPACE", "swebench")
ARCH = "x86_64"

# --- template build resources ---
DEFAULT_CPU = int(os.environ.get("SWEBENCH_CPU", "8"))  # SWE-bench recommends 8 vCPU/instance
DEFAULT_MEMORY_MB = int(os.environ.get("SWEBENCH_MEMORY_MB", "16384"))  # 16 GB; must be even
TEMPLATE_PREFIX = "swebench-"

# --- runtime timeouts (seconds) ---
SANDBOX_TIMEOUT = int(os.environ.get("SWEBENCH_SANDBOX_TIMEOUT", "2400"))  # whole sandbox life
CMD_TIMEOUT = int(os.environ.get("SWEBENCH_CMD_TIMEOUT", "1800"))         # the eval.sh run

# --- concurrency: default 20 = the E2B free-tier cap on concurrent sandboxes.
# Override with SWEBENCH_CONCURRENCY (or --concurrency / --verify-concurrency) to
# raise it on paid tiers. The driver retries Sandbox.create on 429, so running
# right at the cap is safe. ---
DEFAULT_CONCURRENCY = int(os.environ.get("SWEBENCH_CONCURRENCY", "20"))
