#!/usr/bin/env python
"""Build + gold-verify SWE-bench templates in BATCHES, with a persistent ledger.

For each batch: build the templates (parallel processes), then gold-verify each
built template (the gold patch must resolve). Every instance's outcome is
recorded in a ledger that is saved after each batch, so the full 500 can be done
across sessions — stop tonight, re-run the same command tomorrow to continue.

    python scripts/build_and_verify.py --status                       # progress so far
    python scripts/build_and_verify.py --all --batch-size 25          # churn all (resumable)
    python scripts/build_and_verify.py --all --batch-size 25 --max-batches 4   # a few tonight
    python scripts/build_and_verify.py --all --batch-size 25 --status # then check

Resume is automatic: instances already pass/ordering_artifact are skipped.
"""

import argparse
import asyncio
from collections import Counter

from e2b_swebench import (
    build_many,
    gold_prediction,
    load_instances,
    quiet_logs,
    select_per_repo,
)
from e2b_swebench.config import DEFAULT_CONCURRENCY, DEFAULT_CPU, DEFAULT_MEMORY_MB
from e2b_swebench.ledger import Ledger, categorize_verdict
from e2b_swebench.runner import run_many


def select_ids(instances: dict, args) -> list:
    if args.instances:
        return [s.strip() for s in args.instances.split(",") if s.strip()]
    if args.per_repo:
        return select_per_repo(instances, args.per_repo)
    if args.limit:
        return list(instances)[: args.limit]
    if args.all:
        return list(instances)
    return []


def print_summary(ledger: Ledger) -> None:
    s = ledger.summary()
    print(f"ledger: {ledger.path}  ({s['total']} instances tracked)")
    print("  build :", s["build"])
    print("  verify:", s["verify"])
    for label in ("fail", "collection_error", "warning_error", "error"):
        ids = ledger.instances_with_verify(label)
        if ids:
            shown = ", ".join(ids[:12]) + (" ..." if len(ids) > 12 else "")
            print(f"  {label.upper()} ({len(ids)}): {shown}")


def main() -> int:
    quiet_logs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances")
    ap.add_argument("--per-repo", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8, help="parallel template builds")
    ap.add_argument("--verify-concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="concurrent verify sandboxes (keep below your E2B account cap)")
    ap.add_argument("--cpu", type=int, default=DEFAULT_CPU)
    ap.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB)
    ap.add_argument("--max-batches", type=int, help="stop after this many batches (this session)")
    ap.add_argument("--ledger", default="results/ledger.json")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="halt on a genuine 'fail' or build failure; known env classes "
                         "(ordering_artifact / collection_error) are tallied, not halted")
    ap.add_argument("--status", action="store_true", help="print ledger summary and exit")
    args = ap.parse_args()

    ledger = Ledger(args.ledger)
    if args.status:
        print_summary(ledger)
        return 0

    instances = load_instances()
    selected = [i for i in select_ids(instances, args) if i in instances]
    if not selected:
        ap.error("pass one of --instances / --per-repo / --limit / --all")

    todo = [i for i in selected if not ledger.is_done(i)]
    print(f"selected={len(selected)}  done={len(selected) - len(todo)}  todo={len(todo)}")
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    if args.max_batches:
        batches = batches[: args.max_batches]
    print(f"this session: {len(batches)} batch(es) of up to {args.batch_size}\n")

    for bi, batch in enumerate(batches, 1):
        print(f"===== Batch {bi}/{len(batches)}  ({len(batch)} instances) =====")
        # BUILD — idempotent: build_many skips templates that already exist
        results = build_many(
            [instances[i] for i in batch], workers=args.workers,
            cpu_count=args.cpu, memory_mb=args.memory_mb, progress=True,
        )
        for iid, out in results.items():
            if isinstance(out, Exception):
                ledger.update(iid, build="failed", build_error=repr(out), verify=None)
            else:
                ledger.update(iid, build="ok", template=out[0])
        ledger.save()

        # VERIFY — gold patch must resolve
        built_ok = [i for i in batch if ledger.get(i).get("build") == "ok"]
        print(f"  verifying {len(built_ok)} built template(s) with gold ...")
        verdicts = asyncio.run(run_many(
            instances, [gold_prediction(instances[i]) for i in built_ok],
            concurrency=args.verify_concurrency,
        ))
        for v in verdicts:
            cat, detail = categorize_verdict(v)
            ledger.update(v["instance_id"], verify=cat, **detail)
        ledger.save()

        c = Counter(ledger.get(i).get("verify") for i in batch)
        bf_ids = [i for i in batch if ledger.get(i).get("build") == "failed"]
        print(f"  -> pass={c.get('pass', 0)} ordering={c.get('ordering_artifact', 0)} "
              f"collection_error={c.get('collection_error', 0)} warning_error={c.get('warning_error', 0)} "
              f"fail={c.get('fail', 0)} error={c.get('error', 0)} build_failed={len(bf_ids)}\n")

        # Halt only on a GENUINE failure (or a build failure). ordering_artifact and
        # collection_error are known *environment* classes — tallied, not halted.
        # error is transient (retried on resume).
        genuine = [i for i in batch if ledger.get(i).get("verify") == "fail"] + bf_ids
        if args.stop_on_fail and genuine:
            print(f"!! Batch {bi}: {len(genuine)} genuine failure(s) — HALTING for investigation "
                  f"(--stop-on-fail). Fix/understand, then re-run to continue:")
            for i in genuine:
                r = ledger.get(i)
                print(f"   [{r.get('verify') or 'build_failed'}] {i}: "
                      f"resolved={r.get('resolved')} applied={r.get('patch_applied')} "
                      f"f2p_fail={r.get('f2p_fail')} p2p_fail={r.get('p2p_fail')} "
                      f"error={r.get('error') or r.get('build_error')}")
            print()
            print_summary(ledger)
            return 2  # nonzero = halted for investigation (not a crash)

    print("===== session complete =====")
    print_summary(ledger)
    remaining = sum(1 for i in selected if not ledger.is_done(i))
    print(f"\nremaining (not pass/ordering): {remaining}. Re-run the same command to continue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
