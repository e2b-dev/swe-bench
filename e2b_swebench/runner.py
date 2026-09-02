"""Concurrent orchestration over many predictions, gated by a semaphore so we
stay within the configured E2B concurrency cap."""

import asyncio
import hashlib
import json
from collections import Counter

from .config import DEFAULT_CONCURRENCY, DEFAULT_CPU, DEFAULT_MEMORY_MB
from .driver import run_instance_async
from .templates import (
    TemplateSpec,
    content_key,
    instance_image,
    resolve_image,
    template_name_from_spec,
)

_RESOLUTION_CONCURRENCY = 8
_EVALUATION_SCHEMA = 1


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def evaluation_identity(
    spec: TemplateSpec, instance: dict, prediction: dict
) -> dict[str, str | int]:
    """Return the complete deterministic identity for one evaluation route."""
    identity = {
        "evaluation_schema": _EVALUATION_SCHEMA,
        "template": template_name_from_spec(spec),
        "content_key": content_key(spec),
        "source_image": spec.source_image,
        "instance_key": _canonical_hash(instance),
        "prediction_key": _canonical_hash(prediction),
    }
    identity["run_key"] = _canonical_hash(identity)
    return identity


async def resolve_template_specs(
    instances: dict[str, dict],
    predictions: list[dict],
    cpu_count: int = DEFAULT_CPU,
    memory_mb: int = DEFAULT_MEMORY_MB,
    concurrency: int = _RESOLUTION_CONCURRENCY,
) -> dict[str, TemplateSpec]:
    """Resolve each selected instance once on bounded worker threads."""
    selected_ids = dict.fromkeys(
        prediction["instance_id"] for prediction in predictions
    )
    sources = {iid: instance_image(instances[iid]) for iid in selected_ids}
    unique_sources = dict.fromkeys(sources.values())
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def resolve_one(source: str) -> tuple[str, str]:
        async with semaphore:
            return source, await asyncio.to_thread(resolve_image, source)

    resolved_sources = dict(
        await asyncio.gather(*(resolve_one(source) for source in unique_sources))
    )
    return {
        iid: TemplateSpec(
            instance_id=iid,
            source_image=resolved_sources[source],
            cpu_count=cpu_count,
            memory_mb=memory_mb,
        )
        for iid, source in sources.items()
    }


async def _run_one(
    sem: asyncio.Semaphore,
    instance: dict,
    prediction: dict,
    spec: TemplateSpec,
    result_path: str | None = None,
    **kw,
) -> dict:
    identity = evaluation_identity(spec, instance, prediction)
    async with sem:
        try:
            verdict = await run_instance_async(
                instance,
                prediction,
                str(identity["template"]),
                **kw,
            )
        except Exception as e:  # noqa: BLE001 - one failure cannot sink the batch
            verdict = {"resolved": False, "error": repr(e)}
        verdict["instance_id"] = prediction["instance_id"]
        verdict.update(identity)
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
    template_specs: dict[str, TemplateSpec] | None = None,
    **kw,
) -> list[dict]:
    """Run all predictions concurrently. Assumes templates already exist
    (build them with scripts/build_templates.py first). If result_path is given,
    each verdict is appended to it as it completes (for resume/crash safety)."""
    if template_specs is None:
        template_specs = await resolve_template_specs(
            instances,
            predictions,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
        )

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _run_one(
            sem,
            instances[p["instance_id"]],
            p,
            template_specs[p["instance_id"]],
            result_path=result_path,
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
