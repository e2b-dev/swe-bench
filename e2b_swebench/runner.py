"""Concurrent orchestration over many predictions, gated by a semaphore so we
stay within the configured E2B concurrency cap."""

import asyncio
import json
from collections import Counter

from .config import DEFAULT_CONCURRENCY, DEFAULT_CPU, DEFAULT_MEMORY_MB
from .driver import run_instance_async
from .templates import template_name


async def _run_one(
    sem: asyncio.Semaphore,
    instance: dict,
    prediction: dict,
    result_path: str | None = None,
    cpu_count: int = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
    **kw,
) -> dict:
    async with sem:
        try:
            verdict = await run_instance_async(
                instance,
                prediction,
                template_name(prediction["instance_id"], cpu_count, memory_mb),
                **kw,
            )
        except Exception as e:  # noqa: BLE001 - one failure cannot sink the batch
            verdict = {"resolved": False, "error": repr(e)}
        verdict["instance_id"] = prediction["instance_id"]
        # Append each verdict as it completes so a multi-hour run is crash-safe
        # (asyncio has no preemption mid-write, so concurrent appends are atomic).
        if result_path:
            with open(result_path, "a") as f:  # noqa: ASYNC230 - bounded atomic append
                f.write(json.dumps(verdict) + "\n")
        return verdict


async def run_many(
    instances: dict,
    predictions: list[dict],
    concurrency: int = DEFAULT_CONCURRENCY,
    result_path: str | None = None,
    cpu_count: int = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
    **kw,
) -> list[dict]:
    """Run all predictions concurrently. Assumes templates already exist
    (build them with scripts/build_templates.py first). If result_path is given,
    each verdict is appended to it as it completes (for resume/crash safety)."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _run_one(
            sem,
            instances[p["instance_id"]],
            p,
            result_path=result_path,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            **kw,
        )
        for p in predictions
    ]
    return await asyncio.gather(*tasks)


def summarize(verdicts: list[dict]) -> dict:
    from .ledger import categorize_verdict

    resolved = [v["instance_id"] for v in verdicts if v.get("resolved")]
    errored = [v["instance_id"] for v in verdicts if v.get("error")]
    categories = Counter(categorize_verdict(v)[0] for v in verdicts)
    return {
        "total": len(verdicts),
        "resolved": len(resolved),
        "resolved_rate": round(len(resolved) / len(verdicts), 4) if verdicts else 0.0,
        "errored": len(errored),
        "resource_exhausted": sum(bool(v.get("resource_exhausted")) for v in verdicts),
        "categories": dict(categories),
        "resolved_ids": sorted(resolved),
        "errored_ids": sorted(errored),
    }
