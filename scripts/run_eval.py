#!/usr/bin/env python
"""Run a full (or subset) SWE-bench evaluation on E2B.

Predictions come either from a predictions.jsonl file (real agent output) or
--gold (reference patches, for validating the harness). Templates are built
lazily for any selected instance that doesn't have one yet.

    # validate the harness on 20 instances with gold patches
    python scripts/run_eval.py --gold --limit 20 --build

    # score a real model's predictions
    python scripts/run_eval.py --predictions preds.jsonl --concurrency 50
"""

import argparse
import asyncio
import json
import os

from e2b_swebench import (
    build_many,
    gold_prediction,
    load_instances,
    quiet_logs,
    run_many,
    select_per_repo,
)
from e2b_swebench.config import DEFAULT_CONCURRENCY
from e2b_swebench.runner import summarize


def load_predictions(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    quiet_logs()
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gold", action="store_true", help="evaluate gold patches (sanity)")
    src.add_argument("--predictions", help="path to predictions.jsonl")
    ap.add_argument("--instances", help="comma-separated instance_ids to restrict to")
    ap.add_argument("--limit", type=int, help="first N instances")
    ap.add_argument("--per-repo", type=int, help="gold mode: select N instances per distinct repo")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--build", action="store_true", help="lazily build missing templates first")
    ap.add_argument("--build-workers", type=int, default=8, help="parallel template builds")
    ap.add_argument("--out", default="results", help="output dir for predictions.jsonl + report.json")
    ap.add_argument("--resume", action="store_true", help="skip instances already in <out>/verdicts.jsonl")
    args = ap.parse_args()

    instances = load_instances()

    if args.gold:
        if args.instances:
            ids = [s.strip() for s in args.instances.split(",")]
        elif args.per_repo:
            ids = select_per_repo(instances, args.per_repo)
        else:
            ids = list(instances)
        if args.limit:
            ids = ids[: args.limit]
        predictions = [gold_prediction(instances[i]) for i in ids]
    else:
        predictions = load_predictions(args.predictions)
        if args.instances:
            keep = {s.strip() for s in args.instances.split(",")}
            predictions = [p for p in predictions if p["instance_id"] in keep]
        if args.limit:
            predictions = predictions[: args.limit]

    # only evaluate predictions whose instance is in the dataset
    predictions = [p for p in predictions if p["instance_id"] in instances]
    print(f"Selected {len(predictions)} prediction(s)")

    os.makedirs(args.out, exist_ok=True)
    verdicts_path = os.path.join(args.out, "verdicts.jsonl")

    # full prediction set in official-harness format (written before running)
    with open(os.path.join(args.out, "predictions.jsonl"), "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    if args.resume and os.path.exists(verdicts_path):
        done = set()
        with open(verdicts_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["instance_id"])
        before = len(predictions)
        predictions = [p for p in predictions if p["instance_id"] not in done]
        print(f"resume: {len(done)} already done; {len(predictions)}/{before} remaining")
    elif os.path.exists(verdicts_path):
        os.remove(verdicts_path)  # fresh run — don't mix with a stale verdicts log

    if args.build:
        print(f"Building missing templates ({args.build_workers} parallel processes) ...")
        build_many([instances[p["instance_id"]] for p in predictions], workers=args.build_workers)

    print(f"Evaluating {len(predictions)} prediction(s) at concurrency={args.concurrency}")
    asyncio.run(run_many(
        instances, predictions, concurrency=args.concurrency, result_path=verdicts_path,
    ))

    # summarize from verdicts.jsonl (source of truth: resumed + newly completed)
    all_verdicts = []
    with open(verdicts_path) as f:
        for line in f:
            if line.strip():
                all_verdicts.append(json.loads(line))
    report = summarize(all_verdicts)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"report": report, "verdicts": all_verdicts}, f, indent=2)

    print(f"\nresolved {report['resolved']}/{report['total']}  "
          f"({report['resolved_rate'] * 100:.1f}%)  errored={report['errored']}")
    print(f"wrote {args.out}/{{predictions.jsonl, verdicts.jsonl, report.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
