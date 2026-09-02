#!/usr/bin/env python
"""Compare E2B CPU/RAM profiles on the same cross-repo SWE-bench sample.

Each profile gets distinct templates, gold-patch evaluations, five-second E2B
metrics, and a crash-safe JSON report. The default sample spans every repository
in SWE-bench Verified once; the default profiles test progressively larger
single-instance allocations.
"""

import argparse
import asyncio
import datetime
import json
import math
import os
import statistics
import subprocess
import time
from collections import Counter
from importlib.metadata import version

from e2b_swebench import (
    build_many,
    gold_prediction,
    load_instances,
    quiet_logs,
    select_per_repo,
)
from e2b_swebench.config import DATASET, SPLIT
from e2b_swebench.ledger import categorize_verdict
from e2b_swebench.runner import run_many


def parse_profiles(value: str) -> list[tuple[int, int]]:
    profiles = []
    for raw in value.split(","):
        try:
            cpu, memory_mb = (int(part) for part in raw.split(":", 1))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid profile {raw!r}; use CPU:MEMORY_MB"
            ) from exc
        if cpu < 1 or (cpu != 1 and cpu % 2):
            raise argparse.ArgumentTypeError("CPU must be 1 or an even number")
        if memory_mb < 512 or memory_mb % 2:
            raise argparse.ArgumentTypeError("memory must be >= 512 MiB and even")
        profiles.append((cpu, memory_mb))
    return profiles


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def aggregate(verdicts: list[dict]) -> dict:
    categories = []
    for verdict in verdicts:
        category, _ = categorize_verdict(verdict)
        categories.append(category)

    runtimes = [v["runtime_seconds"] for v in verdicts if "runtime_seconds" in v]
    metrics = [
        v.get("metrics", {}) for v in verdicts if v.get("metrics", {}).get("samples")
    ]
    peak_working_sets = [m["peak_memory_working_set_mb"] for m in metrics]
    peak_used = [m["peak_memory_used_mb"] for m in metrics]
    peak_cpu_cores = [m["peak_cpu_cores"] for m in metrics]
    headroom = [m["min_memory_working_set_headroom_mb"] for m in metrics]
    return {
        "categories": dict(Counter(categories)),
        "resource_exhausted": sum(bool(v.get("resource_exhausted")) for v in verdicts),
        "runtime_seconds": {
            "median": round(statistics.median(runtimes), 2) if runtimes else None,
            "p95": round(percentile(runtimes, 0.95), 2) if runtimes else None,
            "max": round(max(runtimes), 2) if runtimes else None,
        },
        "metrics": {
            "runs_sampled": len(metrics),
            "max_peak_memory_working_set_mb": max(peak_working_sets, default=None),
            "p95_peak_memory_working_set_mb": percentile(peak_working_sets, 0.95),
            "max_peak_memory_used_mb": max(peak_used, default=None),
            "max_peak_cpu_cores": max(peak_cpu_cores, default=None),
            "p95_peak_cpu_cores": percentile(peak_cpu_cores, 0.95),
            "min_working_set_headroom_mb": min(headroom, default=None),
        },
    }


def write_report(path: str, report: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main() -> int:
    quiet_logs()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profiles",
        type=parse_profiles,
        default=parse_profiles("2:4096,4:4096,4:8192,8:16384"),
    )
    ap.add_argument("--instances", help="comma-separated instance IDs")
    ap.add_argument(
        "--per-repo", type=int, help="instances per repository (default: 1)"
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--build-workers", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default="results/resource-benchmark.json")
    args = ap.parse_args()

    instances = load_instances()
    if args.instances:
        ids = [value.strip() for value in args.instances.split(",") if value.strip()]
    elif args.limit:
        ids = list(instances)[: args.limit]
    else:
        ids = select_per_repo(instances, args.per_repo or 1)
    missing = [iid for iid in ids if iid not in instances]
    if missing:
        ap.error(f"not in {DATASET}/{SPLIT}: {', '.join(missing)}")

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    report = {
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": DATASET,
        "split": SPLIT,
        "instances": ids,
        "environment": {
            "git_sha": git_sha,
            "git_dirty": dirty,
            "e2b": version("e2b"),
            "swebench": version("swebench"),
            "datasets": version("datasets"),
        },
        "profiles": [],
    }

    for cpu_count, memory_mb in args.profiles:
        label = f"{cpu_count}c-{memory_mb}m"
        print(f"\n===== {label}: build {len(ids)} templates =====", flush=True)
        build_started = time.monotonic()
        builds = build_many(
            [instances[iid] for iid in ids],
            workers=args.build_workers,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
        )
        build_seconds = round(time.monotonic() - build_started, 2)
        build_failures = {
            iid: repr(out) for iid, out in builds.items() if isinstance(out, Exception)
        }
        runnable = [iid for iid in ids if iid not in build_failures]

        print(
            f"===== {label}: gold-evaluate {len(runnable)} instances =====", flush=True
        )
        verdicts = asyncio.run(
            run_many(
                instances,
                [gold_prediction(instances[iid]) for iid in runnable],
                concurrency=args.concurrency,
                cpu_count=cpu_count,
                memory_mb=memory_mb,
            )
        )
        profile = {
            "label": label,
            "cpu_count": cpu_count,
            "memory_mb": memory_mb,
            "build_seconds": build_seconds,
            "build_failures": build_failures,
            "aggregate": aggregate(verdicts),
            "verdicts": verdicts,
        }
        report["profiles"].append(profile)
        write_report(args.out, report)
        print(json.dumps(profile["aggregate"], indent=2), flush=True)

    print(f"\nwrote {args.out}")
    return 1 if any(p["build_failures"] for p in report["profiles"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
