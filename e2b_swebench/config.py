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
# SWE-bench recommends 16 GiB, but E2B accounts cap template memory at 8 GiB by
# default, so 16384 makes the very first build fail with a 400. 8 GiB is what
# every account can actually build; raise it (support raises the cap) with
# SWEBENCH_MEMORY_MB=16384 to match the upstream recommendation. Must be even.
DEFAULT_MEMORY_MB = int(os.environ.get("SWEBENCH_MEMORY_MB", "8192"))
TEMPLATE_PREFIX = "swebench-"

# --- runtime timeouts (seconds) ---
SANDBOX_TIMEOUT = int(os.environ.get("SWEBENCH_SANDBOX_TIMEOUT", "2400"))  # whole sandbox life
CMD_TIMEOUT = int(os.environ.get("SWEBENCH_CMD_TIMEOUT", "1800"))         # the eval.sh run

# --- concurrency: default 20 = the E2B free-tier cap on concurrent sandboxes.
# Override with SWEBENCH_CONCURRENCY (or --concurrency / --verify-concurrency) to
# raise it on paid tiers. The driver retries Sandbox.create on 429, so running
# right at the cap is safe. ---
DEFAULT_CONCURRENCY = int(os.environ.get("SWEBENCH_CONCURRENCY", "20"))
