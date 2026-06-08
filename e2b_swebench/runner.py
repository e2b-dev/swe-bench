"""Concurrent orchestration over many predictions, gated by a semaphore so we
stay under the E2B concurrency cap (20 Hobby / 100 Pro)."""

import asyncio
import json

from .config import DEFAULT_CONCURRENCY
from .driver import run_instance_async
from .templates import template_name


async def _run_one(
    sem: asyncio.Semaphore,
    instance: dict,
    prediction: dict,
    result_path: str | None = None,
    **kw,
) -> dict:
    async with sem:
        try:
            verdict = await run_instance_async(
                instance, prediction, template_name(prediction["instance_id"]), **kw
            )
        except Exception as e:  # never let one failure sink the batch
            verdict = {"resolved": False, "error": repr(e)}
        verdict["instance_id"] = prediction["instance_id"]
        # Append each verdict as it completes so a multi-hour run is crash-safe
        # (asyncio has no preemption mid-write, so concurrent appends are atomic).
        if result_path:
            with open(result_path, "a") as f:
                f.write(json.dumps(verdict) + "\n")
        return verdict


async def run_many(
    instances: dict,
    predictions: list[dict],
    concurrency: int = DEFAULT_CONCURRENCY,
    result_path: str | None = None,
    **kw,
) -> list[dict]:
    """Run all predictions concurrently. Assumes templates already exist
    (build them with scripts/build_templates.py first). If result_path is given,
    each verdict is appended to it as it completes (for resume/crash safety)."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _run_one(sem, instances[p["instance_id"]], p, result_path=result_path, **kw)
        for p in predictions
    ]
    return await asyncio.gather(*tasks)


def summarize(verdicts: list[dict]) -> dict:
    resolved = [v["instance_id"] for v in verdicts if v.get("resolved")]
    errored = [v["instance_id"] for v in verdicts if v.get("error")]
    return {
        "total": len(verdicts),
        "resolved": len(resolved),
        "resolved_rate": round(len(resolved) / len(verdicts), 4) if verdicts else 0.0,
        "errored": len(errored),
        "resolved_ids": sorted(resolved),
        "errored_ids": sorted(errored),
    }
