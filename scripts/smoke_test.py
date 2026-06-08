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

import sys

from e2b import Sandbox

from e2b_swebench import (
    ensure_template,
    gold_prediction,
    empty_prediction,
    load_instances,
    quiet_logs,
    run_instance,
)

DEFAULT_INSTANCE = "astropy__astropy-12907"


def main() -> int:
    quiet_logs()
    instance_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INSTANCE

    print(f"Loading dataset and selecting {instance_id} ...")
    instances = load_instances()
    if instance_id not in instances:
        print(f"  ! {instance_id} not in dataset ({len(instances)} instances)")
        return 2
    inst = instances[instance_id]

    print(f"Ensuring template (building FROM the prebuilt image if needed) ...")
    name, built = ensure_template(inst)
    print(f"  template: {name}  ({'built' if built else 'already existed'})")

    print("Verifying the environment inside a fresh sandbox ...")
    base = inst["base_commit"]
    sbx = Sandbox.create(name, timeout=600)
    try:
        head = sbx.commands.run("git rev-parse HEAD", cwd="/testbed", user="root").stdout.strip()
        # SWE-bench images add a "SWE-bench" setup commit ON TOP of base_commit
        # (it tweaks pyproject.toml so the env installs), so HEAD != base_commit
        # by design. The correct invariant: base_commit is an ANCESTOR of HEAD.
        anc = sbx.commands.run(
            f"git merge-base --is-ancestor {base} HEAD && echo yes || echo no",
            cwd="/testbed", user="root",
        ).stdout.strip()
        pyver = sbx.commands.run(
            "source /opt/miniconda3/bin/activate testbed && python --version",
            user="root",
        ).stdout.strip()
    finally:
        sbx.kill()
    ok_commit = anc == "yes"
    print(f"  /testbed HEAD: {head[:12]}  (base_commit {base[:12]} ancestor? {anc})  {'OK' if ok_commit else 'FAIL'}")
    print(f"  conda env 'testbed' python: {pyver}")

    print("GOLD patch (expect resolved=True) ...")
    gold = run_instance(inst, gold_prediction(inst), name)
    print(f"  resolved={gold.get('resolved')}  applied={gold.get('patch_successfully_applied')}  err={gold.get('error')}")

    print("EMPTY patch (expect resolved=False) ...")
    empty = run_instance(inst, empty_prediction(inst), name)
    print(f"  resolved={empty.get('resolved')}")

    ok = ok_commit and gold.get("resolved") is True and empty.get("resolved") is False
    print("\n" + ("PASS ✅  harness round-trips correctly" if ok else "FAIL ❌  see output above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
