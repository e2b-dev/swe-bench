#!/usr/bin/env python
"""POC sanity check on a SINGLE instance — run this before anything else.

Proves the whole Strategy-A path works end to end:
  1. build an E2B template FROM the prebuilt image (lazy)
  2. spawn it; confirm /testbed is at base_commit and the conda env activates
  3. GOLD patch  -> must resolve True   (harness round-trips)
  4. EMPTY patch -> must resolve False  (no false positives)

Requires: E2B_API_KEY in the environment.

    python scripts/smoke_test.py                       # default instance
    python scripts/smoke_test.py sympy__sympy-20438     # a specific one
"""

import argparse

from e2b import Sandbox

from e2b_swebench import (
    empty_prediction,
    ensure_template,
    gold_prediction,
    load_instances,
    quiet_logs,
    run_instance,
)
from e2b_swebench.config import DEFAULT_CPU, DEFAULT_MEMORY_MB

DEFAULT_INSTANCE = "astropy__astropy-12907"


def main() -> int:
    quiet_logs()
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_id", nargs="?", default=DEFAULT_INSTANCE)
    ap.add_argument("--cpu", type=int, default=DEFAULT_CPU)
    ap.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB)
    args = ap.parse_args()
    instance_id = args.instance_id

    print(f"Loading dataset and selecting {instance_id} ...")
    instances = load_instances()
    if instance_id not in instances:
        print(f"  ! {instance_id} not in dataset ({len(instances)} instances)")
        return 2
    inst = instances[instance_id]

    print("Ensuring template (building FROM the prebuilt image if needed) ...")
    name, built = ensure_template(inst, cpu_count=args.cpu, memory_mb=args.memory_mb)
    print(f"  template: {name}  ({'built' if built else 'already existed'})")

    print("Verifying the environment inside a fresh sandbox ...")
    base = inst["base_commit"]
    sbx = Sandbox.create(name, timeout=600)
    try:
        head = sbx.commands.run(
            "git rev-parse HEAD", cwd="/testbed", user="root"
        ).stdout.strip()
        # SWE-bench images add a "SWE-bench" setup commit ON TOP of base_commit
        # (it tweaks pyproject.toml so the env installs), so HEAD != base_commit
        # by design. The correct invariant: base_commit is an ANCESTOR of HEAD.
        anc = sbx.commands.run(
            f"git merge-base --is-ancestor {base} HEAD && echo yes || echo no",
            cwd="/testbed",
            user="root",
        ).stdout.strip()
        pyver = sbx.commands.run(
            "source /opt/miniconda3/bin/activate testbed && python --version",
            user="root",
        ).stdout.strip()
    finally:
        sbx.kill()
    ok_commit = anc == "yes"
    print(
        f"  /testbed HEAD: {head[:12]}  (base_commit {base[:12]} ancestor? {anc})  {'OK' if ok_commit else 'FAIL'}"
    )
    print(f"  conda env 'testbed' python: {pyver}")

    print("GOLD patch (expect resolved=True) ...")
    gold = run_instance(inst, gold_prediction(inst), name)
    print(
        f"  resolved={gold.get('resolved')}  applied={gold.get('patch_successfully_applied')}  err={gold.get('error')}"
    )
    metrics = gold.get("metrics", {})
    if metrics.get("samples"):
        print(
            f"  runtime={gold['runtime_seconds']:.1f}s  "
            f"peak_cpu={metrics['peak_cpu_cores']:.2f} cores  "
            f"peak_working_set={metrics['peak_memory_working_set_mb']:.0f} MiB  "
            f"peak_used={metrics['peak_memory_used_mb']:.0f} MiB"
        )

    print("EMPTY patch (expect resolved=False) ...")
    empty = run_instance(inst, empty_prediction(inst), name)
    print(f"  resolved={empty.get('resolved')}")

    ok = ok_commit and gold.get("resolved") is True and empty.get("resolved") is False
    print(
        "\n"
        + (
            "PASS ✅  harness round-trips correctly"
            if ok
            else "FAIL ❌  see output above"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
