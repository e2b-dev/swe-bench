"""Compact E2B sandbox metrics for benchmark verdicts."""

import math
from collections.abc import Sequence

_MIB = 1024 * 1024


def _round_mib(value: float) -> float:
    return round(value / _MIB, 1)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def summarize_metrics(metrics) -> dict:
    """Summarize the SDK's five-second samples without storing a large trace.

    E2B reports memory cache separately. ``working_set`` subtracts that
    reclaimable page cache from used memory, which is the useful signal for RAM
    sizing; raw used memory is retained as the conservative bound.
    """
    if not metrics:
        return {"samples": 0}

    cpu_count = metrics[-1].cpu_count
    cpu_pct = [m.cpu_used_pct for m in metrics]
    working_set = [max(0, m.mem_used - m.mem_cache) for m in metrics]
    mem_used = [m.mem_used for m in metrics]
    disk_used = [m.disk_used for m in metrics]
    mem_total = metrics[-1].mem_total

    peak_cpu_pct = max(cpu_pct)
    mean_cpu_pct = sum(cpu_pct) / len(cpu_pct)
    return {
        "samples": len(metrics),
        "sample_interval_seconds": 5,
        "cpu_count": cpu_count,
        "peak_cpu_pct": round(peak_cpu_pct, 1),
        "mean_cpu_pct": round(mean_cpu_pct, 1),
        "peak_cpu_cores": round(peak_cpu_pct * cpu_count / 100, 2),
        "mean_cpu_cores": round(mean_cpu_pct * cpu_count / 100, 2),
        "memory_total_mb": _round_mib(mem_total),
        "peak_memory_used_mb": _round_mib(max(mem_used)),
        "peak_memory_working_set_mb": _round_mib(max(working_set)),
        "p95_memory_working_set_mb": _round_mib(_percentile(working_set, 0.95)),
        "min_memory_working_set_headroom_mb": _round_mib(mem_total - max(working_set)),
        "disk_total_mb": _round_mib(metrics[-1].disk_total),
        "peak_disk_used_mb": _round_mib(max(disk_used)),
    }
