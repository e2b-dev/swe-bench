"""Load SWE-bench instances and build prediction dicts.

A prediction is what the harness grades: {instance_id, model_name_or_path, model_patch}.
"""

import json

from datasets import load_dataset

from .config import DATASET, SPLIT


def load_instances(dataset: str = DATASET, split: str = SPLIT) -> dict:
    """Return {instance_id: instance_dict}. Each instance is a plain dict that
    swebench.make_test_spec accepts directly (SWEbenchInstance is a TypedDict)."""
    ds = load_dataset(dataset, split=split)
    return {row["instance_id"]: dict(row) for row in ds}


def parse_tests(instance: dict):
    """FAIL_TO_PASS / PASS_TO_PASS are JSON-encoded *strings* in the dataset."""
    return json.loads(instance["FAIL_TO_PASS"]), json.loads(instance["PASS_TO_PASS"])


def gold_prediction(instance: dict) -> dict:
    """The reference fix. Must resolve — this is the harness sanity check."""
    return {
        "instance_id": instance["instance_id"],
        "model_name_or_path": "gold",
        "model_patch": instance["patch"],
    }


def empty_prediction(instance: dict) -> dict:
    """No-op patch. Must NOT resolve — negative control against false positives."""
    return {
        "instance_id": instance["instance_id"],
        "model_name_or_path": "empty",
        "model_patch": "",
    }


def select_per_repo(instances: dict, n: int = 1) -> list[str]:
    """Pick the first n instance_ids from each distinct repo (sorted for
    determinism). Use this to span all ~12 repos — exercising every per-repo
    log parser — instead of alphabetically over-sampling with --limit."""
    by_repo: dict[str, list[str]] = {}
    for iid in sorted(instances):
        repo = instances[iid]["repo"]
        bucket = by_repo.setdefault(repo, [])
        if len(bucket) < n:
            bucket.append(iid)
    return [iid for bucket in by_repo.values() for iid in bucket]
