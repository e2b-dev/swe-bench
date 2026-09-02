# SWE-bench on E2B

This repository is the standalone E2B execution integration for
[SWE-bench](https://github.com/SWE-bench/SWE-bench). It builds an immutable E2B
Template for each selected instance, runs candidate patches in fresh sandboxes,
and passes the resulting test logs to the pinned `swebench` grader.

The repository owns template construction, E2B execution, result identity, and
resume behavior. It does not own agent scheduling, model inference, or benchmark
result publication. A Harbor adapter may call these commands as an optional
consumer; Harbor is not required to build or run the integration.

## Execution contract

SWE-bench publishes a Docker image for each instance. The image contains the
repository under `/testbed`, the benchmark setup commit, and the task's Python
environment. This integration:

1. Resolves the selected Docker Hub tag to an OCI digest.
2. Derives a content identity from that digest and every template-construction
   input.
3. Reuses or builds the corresponding E2B Template.
4. Starts a fresh sandbox from that Template for each prediction.
5. Applies the prediction, runs the generated evaluation script, and grades the
   combined output with `swebench==4.1.0`.

Template aliases are derived from a canonical SHA-256 identity covering the
digest-pinned source image, work directory, architecture, namespace, CPU,
memory, and construction schema. A mutable image tag is never a reusable
artifact identity. If the tag resolves to a different digest, the integration
selects a different Template alias.

Build and evaluation are separate operations. Evaluation assumes its immutable
Template already exists unless the caller explicitly passes `--build`.

### Resume identity

Evaluation resume records are accepted only when their complete current identity
matches: evaluation schema, Template alias, template content key, source-image
digest, instance hash, prediction hash, and aggregate run key. Build-and-verify
ledger records also bind successful work to the Template identity and selected
resources. Legacy or stale records are reprocessed.

### SWE-bench setup preservation

Published SWE-bench images may add a setup commit on top of an instance's
`base_commit`. Resetting the whole repository to `base_commit` would discard
that setup. The driver therefore removes only the generated whole-repository
checkout or reset that targets the exact `base_commit`. Path-specific checkouts
used to place held-out tests are preserved, as are install, patch, test, and
cleanup command order.

## Requirements

- Python 3.10, 3.11, or 3.12
- `uv` for the locked development and operator environment
- An E2B team and API key for live Template or sandbox operations
- Network access to Docker Hub, the configured dataset source, and E2B for live
  operations

Install the locked environment:

```bash
uv sync --locked
```

For live operations, authenticate with the E2B CLI or set `E2B_API_KEY` in the
process environment. `HF_TOKEN` is optional when the configured dataset source
accepts anonymous access. Never commit either value; `.env` files are ignored.

## Live commands

The commands in this section contact external services. Template builds and
sandbox runs can consume E2B resources and incur charges.

Run one end-to-end control containing a gold patch and an empty patch:

```bash
uv run --locked python scripts/smoke_test.py
uv run --locked python scripts/smoke_test.py sympy__sympy-20438
```

Build immutable Templates without evaluating them:

```bash
uv run --locked python scripts/build_templates.py --instances astropy__astropy-12907
uv run --locked python scripts/build_templates.py --per-repo 1
uv run --locked python scripts/build_templates.py --all --workers 8
```

Build and gold-verify in resumable batches:

```bash
uv run --locked python scripts/build_and_verify.py --all --batch-size 25 --stop-on-fail
uv run --locked python scripts/build_and_verify.py --all --batch-size 25 --max-batches 4
uv run --locked python scripts/build_and_verify.py --status
```

Evaluate reference patches or model predictions:

```bash
uv run --locked python scripts/run_eval.py --gold --per-repo 1 --build
uv run --locked python scripts/run_eval.py --predictions predictions.jsonl --build --resume
```

Prediction files use one JSON object per line:

```json
{"instance_id":"owner__repo-123","model_name_or_path":"model","model_patch":"diff --git ..."}
```

Capture an agent's patch relative to the setup-aware image state with
`git -C /testbed diff`; do not reset the repository to the dataset
`base_commit` first.

## Configuration

Command-line flags take precedence over environment defaults where both are
available.

| Variable | Code default | Purpose |
| --- | --- | --- |
| `SWEBENCH_DATASET` | `princeton-nlp/SWE-bench_Verified` | Dataset identifier passed to `datasets` |
| `SWEBENCH_SPLIT` | `test` | Dataset split |
| `SWEBENCH_NAMESPACE` | `swebench` | Docker Hub image namespace |
| `SWEBENCH_CPU` | `4` | vCPUs in each Template |
| `SWEBENCH_MEMORY_MB` | `4096` | Template memory in MiB |
| `SWEBENCH_SANDBOX_TIMEOUT` | `2400` | Sandbox lifetime in seconds |
| `SWEBENCH_CMD_TIMEOUT` | `1800` | Evaluation-command timeout in seconds |
| `SWEBENCH_BUILD_RECOVERY_TIMEOUT` | `120` | Wait for a recoverable Template build in seconds |
| `SWEBENCH_CONCURRENCY` | `100` | Maximum sandbox tasks requested concurrently |

Architecture is fixed to `x86_64`. Choose CPU, memory, worker count, and
concurrency within the limits of the E2B team running the evaluation.

## Generated files

All generated run artifacts are ignored by Git.

| Path | Producer | Contents |
| --- | --- | --- |
| `results/ledger.json` | `build_and_verify.py` | Atomic, identity-bound build and gold-verification state |
| `results/<resources>/run_manifest.json` | `run_eval.py` | Identity of every selected evaluation |
| `results/<resources>/predictions.jsonl` | `run_eval.py` | Predictions supplied to the run |
| `results/<resources>/verdicts.jsonl` | `run_eval.py` | Crash-safe completed verdicts |
| `results/<resources>/report.json` | `run_eval.py` | Aggregate report and verdicts |

The ledger accepts only `pass` and explicitly audited, exact failure signatures
as complete. Other gold failures remain incomplete for investigation. Transient
execution errors are retried on a later run.

## Python API

Resolve an immutable Template specification before selecting its alias:

```python
from e2b_swebench import resolve_template_spec, template_name_from_spec

spec = resolve_template_spec(instance, cpu_count=4, memory_mb=4096)
template_alias = template_name_from_spec(spec)
```

`immutable_template_name(instance, ...)` is the one-call equivalent. The legacy
`template_name(...)` entry point requires a complete instance and rejects a bare
instance ID because an ID cannot determine the current image digest.

## Local verification

The unit suite installs process-wide guards against sockets, Docker registry
requests, E2B SDK calls, and dataset downloads. It requires no API key and makes
no live service calls.

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check .
uv run --locked ruff format --check .
uv build --no-build-isolation --clear
uv run --locked twine check dist/*
```

## Limitations

- Source-image resolution supports Docker Hub references with SHA-256 digests.
- Each selected task must have a compatible published `x86_64` SWE-bench image.
- Dataset rows must match the API and schema consumed by `swebench==4.1.0`.
- Live builds and evaluations are not offline or cost-free operations.
- This repository does not validate model quality or publish benchmark scores.

Report vulnerabilities according to [SECURITY.md](SECURITY.md).
