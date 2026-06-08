#!/usr/bin/env python
"""Pre-build E2B templates (Strategy A) for a set of instances.

Build is server-side on E2B (no local Docker) and idempotent: existing
templates are skipped unless --force. Builds run in parallel PROCESSES (one
E2B connection each — threads share an HTTP/2 connection and collide).

    python scripts/build_templates.py --per-repo 1            # one per repo (~12)
    python scripts/build_templates.py --limit 20
    python scripts/build_templates.py --instances astropy__astropy-12907,sympy__sympy-20438
    python scripts/build_templates.py --all --workers 8       # all 500
"""

import argparse

from e2b_swebench import build_many, load_instances, quiet_logs, select_per_repo
from e2b_swebench.config import DEFAULT_CPU, DEFAULT_MEMORY_MB


def main() -> int:
    quiet_logs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", help="comma-separated instance_ids")
    ap.add_argument("--per-repo", type=int, help="build N instances per distinct repo")
    ap.add_argument("--limit", type=int, help="build the first N instances")
    ap.add_argument("--all", action="store_true", help="build every instance in the dataset")
    ap.add_argument("--cpu", type=int, default=DEFAULT_CPU)
    ap.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB)
    ap.add_argument("--workers", type=int, default=4, help="parallel build processes")
    ap.add_argument("--force", action="store_true", help="rebuild even if it exists")
    args = ap.parse_args()

    instances = load_instances()
    if args.instances:
        ids = [s.strip() for s in args.instances.split(",") if s.strip()]
    elif args.per_repo:
        ids = select_per_repo(instances, args.per_repo)
    elif args.all:
        ids = list(instances)
    elif args.limit:
        ids = list(instances)[: args.limit]
    else:
        ap.error("pass one of --instances / --per-repo / --limit / --all")

    missing = [i for i in ids if i not in instances]
    for i in missing:
        print(f"NOT IN DATASET: {i}")
    selected = [instances[i] for i in ids if i in instances]

    print(f"Building {len(selected)} template(s) with {args.workers} worker process(es) ...")
    results = build_many(
        selected, workers=args.workers, cpu_count=args.cpu,
        memory_mb=args.memory_mb, force=args.force,
    )

    built = sum(1 for r in results.values() if not isinstance(r, Exception) and r[1])
    exists = sum(1 for r in results.values() if not isinstance(r, Exception) and not r[1])
    failed = sum(1 for r in results.values() if isinstance(r, Exception)) + len(missing)
    print(f"\nbuilt={built} exists={exists} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
