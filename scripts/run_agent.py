"""Generate SWE-bench predictions with a coding agent running on E2B."""

import argparse
import json
import os

from e2b_swebench import load_instances, quiet_logs, select_per_repo
from e2b_swebench.agents import (
    REGISTRY,
    GenerationResult,
    MissingTemplates,
    ModelUnavailable,
    agent_names,
    check_templates,
    get_agent,
)
from e2b_swebench.agents.muse_spark import DEFAULT_MAX_STEPS
from e2b_swebench.config import SANDBOX_TIMEOUT
from e2b_swebench.templates import template_name

DEFAULT_AGENT = "muse-spark"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def select_ids(instances: dict, args: argparse.Namespace) -> list[str]:
    """Same selection vocabulary as build_templates.py / run_eval.py."""
    if args.instances:
        ids = [s.strip() for s in args.instances.split(",") if s.strip()]
    elif args.per_repo:
        ids = select_per_repo(instances, args.per_repo)
    elif args.limit:
        ids = list(instances)
    else:
        return []
    if args.limit:
        ids = ids[: args.limit]
    return ids


def _open_fresh(path: str):
    """Create a line-buffered JSONL output, replacing any previous run."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return open(path, "w", buffering=1, encoding="utf-8")


def _print_agents() -> int:
    print("Available agents:\n")
    for name in agent_names():
        spec = REGISTRY[name]
        env = ", ".join(spec.requires_env) or "none"
        marker = "  (default)" if name == DEFAULT_AGENT else ""
        print(f"  {name}{marker}")
        print(f"      {spec.description}")
        print(f"      default model: {spec.default_model}")
        print(f"      requires env : {env}\n")
    return 0


def main() -> int:
    quiet_logs()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--agent",
        default=DEFAULT_AGENT,
        choices=agent_names(),
        help=f"which agent generates the patches (default {DEFAULT_AGENT})",
    )
    ap.add_argument(
        "--list-agents",
        action="store_true",
        help="print the registered agents and exit",
    )
    ap.add_argument("--instances", help="comma-separated instance_ids")
    ap.add_argument("--limit", type=positive_int, help="first N instances")
    ap.add_argument(
        "--per-repo", type=positive_int, help="select N instances per distinct repo"
    )
    ap.add_argument("--model", help="model id (default: the agent's own default)")
    ap.add_argument(
        "--max-steps",
        type=positive_int,
        default=DEFAULT_MAX_STEPS,
        help="maximum model tool-loop iterations per instance",
    )
    ap.add_argument(
        "--sandbox-timeout",
        type=positive_int,
        default=SANDBOX_TIMEOUT,
        help="sandbox lifetime in seconds",
    )
    ap.add_argument(
        "--out",
        default="results/predictions.jsonl",
        help="predictions JSONL, in the official harness format",
    )
    ap.add_argument(
        "--status",
        help="generation status JSONL (default: generation.jsonl beside --out)",
    )
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the model connectivity check (it costs one short request)",
    )
    args = ap.parse_args()

    if args.list_agents:
        return _print_agents()

    status_path = args.status or os.path.join(
        os.path.dirname(args.out) or ".", "generation.jsonl"
    )
    if os.path.realpath(status_path) == os.path.realpath(args.out):
        ap.error("--status and --out must be different files")
    if not (args.instances or args.per_repo or args.limit):
        ap.error("pass one of --instances / --per-repo / --limit")

    agent = get_agent(args.agent)
    model = args.model or os.environ.get("META_MODEL") or agent.default_model

    instances = load_instances()
    ids = select_ids(instances, args)
    if not ids:
        ap.error("the instance selection is empty")

    for iid in [i for i in ids if i not in instances]:
        print(f"NOT IN DATASET: {iid}")
    ids = [i for i in ids if i in instances]
    if not ids:
        print("nothing to do")
        return 1

    # Both preflights run before a single sandbox exists, so a bad selection,
    # a bad key or a bad model id costs nothing.
    try:
        check_templates(ids)
    except MissingTemplates as error:
        print(error)
        return 1

    try:
        client = agent.create_client()
        if not args.skip_preflight:
            agent.check_model(client, model)
    except (ModelUnavailable, RuntimeError) as error:
        print(error)
        return 1

    print(f"Agent    : {args.agent} ({agent.description})")
    print(f"Model    : {model}")
    print(f"Instances: {len(ids)}  (max_steps={args.max_steps})\n")

    counts: dict[str, int] = {"ok": 0, "empty_patch": 0, "error": 0}
    stopped_early = False
    with _open_fresh(args.out) as preds, _open_fresh(status_path) as status:
        for i, iid in enumerate(ids, 1):
            try:
                prediction, result = agent.generate_prediction(
                    instances[iid],
                    template_name(iid),
                    client,
                    model=model,
                    max_steps=args.max_steps,
                    sandbox_timeout=args.sandbox_timeout,
                )
            except KeyboardInterrupt:
                print("\ninterrupted; the sandbox was killed and partial output kept")
                raise
            except Exception as error:  # noqa: BLE001 - keep other instances running
                prediction = agent.empty_prediction(iid, model)
                result = GenerationResult(
                    instance_id=iid,
                    model=model,
                    status="error",
                    error=repr(error),
                )

            preds.write(json.dumps(prediction) + "\n")
            status.write(json.dumps(result.to_dict()) + "\n")
            counts[result.status] = counts.get(result.status, 0) + 1
            print(
                f"[{i}/{len(ids)}] {iid}: {result.status} "
                f"steps={result.steps} patch_bytes={result.patch_bytes}"
                + (f" error={result.error}" if result.error else "")
            )

            if result.fatal:
                print(
                    f"\nStopping: that failure is permanent, so the remaining "
                    f"{len(ids) - i} instance(s) would fail identically.\n"
                    f"Fix the key/model and re-run; nothing further was started."
                )
                stopped_early = True
                break

    print(
        f"\nok={counts['ok']} empty_patch={counts['empty_patch']} "
        f"error={counts['error']}"
    )
    print(f"wrote {args.out} and {status_path}")
    if not stopped_early:
        print(
            f"\nGrade them (fresh sandbox per instance, canonical grader):\n"
            f"    python scripts/run_eval.py --predictions {args.out} --out results/eval"
        )
    return 1 if stopped_early else 0


if __name__ == "__main__":
    raise SystemExit(main())
