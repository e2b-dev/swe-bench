# SWE-bench on E2B

Run the [SWE-bench](https://github.com/swe-bench/SWE-bench) benchmark on [E2B](https://e2b.dev/?utm_source=github&utm_medium=referral&utm_campaign=readme&utm_content=swe-bench) sandboxes.

---


A harness that evaluates SWE-bench instances inside **E2B sandboxes** instead of
local Docker. SWE-bench's grading logic is pure Python (no Docker dependency), so
this project reuses all of it and only swaps the *execution environment*: where
the official harness runs each task in a Docker container, this runs it in an
E2B sandbox.

Per instance it: places the repo at the right commit, applies a candidate patch
(a model's fix — or the **gold** fix, for validation), runs the held-out tests,
and computes `resolved` with the canonical `swebench` grader.

It ships:
- a one-shot **smoke test**,
- a **batched, resumable** build-and-verify pipeline for the whole dataset, backed by a persistent ledger,
- plain **build** and **eval** scripts,
- an importable Python package (`e2b_swebench`).

---

## 2. Architecture

**One E2B template per instance.**

SWE-bench publishes a prebuilt Docker image *per instance* on Docker Hub:
`swebench/sweb.eval.x86_64.<instance_id>` (with `__` → `_1776_`). Each image
already has the repo checked out at `base_commit` in `/testbed` and a conda env
`testbed` with the project installed.

We build **one E2B template per instance, `FROM` that image**. A sandbox then
spawns ready-to-eval, with everything on the sandbox's fast **local disk** — no
per-run clone or install. Builds run server-side on E2B (**no local Docker**).

```
SWE-bench image (Docker Hub)        E2B template (per instance)      E2B sandbox (per run)
sweb.eval.x86_64.<id>        ──►     swebench-<id>            ──►      /testbed @ base_commit
  /testbed @ base_commit             (FROM the image)                 conda env "testbed" ready
  conda env "testbed"
```

The per-run flow maps the official Docker harness 1:1 onto E2B:

| official harness (Docker)     | this project (E2B)                              |
| ----------------------------- | ----------------------------------------------- |
| build_container               | `Sandbox.create(template)`                      |
| copy patch into container     | `sandbox.files.write(...)`                      |
| exec `git apply`              | `sandbox.commands.run("git apply ...")`         |
| exec `/eval.sh`               | `sandbox.commands.run("/bin/bash /eval.sh ...")`|
| parse logs → resolved         | `swebench.harness.grading.get_eval_report` (unchanged) |

**Grading is canonical.** Combined stdout+stderr is captured and fed to
`get_eval_report`, which selects the correct per-repo log parser and applies the
rule: **`resolved` iff every `FAIL_TO_PASS` test passes AND every `PASS_TO_PASS`
test still passes.**

**Three things worth knowing:**

- The SWE-bench image adds a `"SWE-bench"` commit on top of `base_commit` (it
  tweaks a build file), so `/testbed` HEAD ≠ `base_commit` — `base_commit` is its
  *ancestor*. A real agent should capture its work as `git -C /testbed diff`
  (against HEAD), not reset to `base_commit`.
- Some instances run the *entire* repo test-suite, whose order depends on
  filesystem `scandir` order. On a non-Docker filesystem this can differ from
  Docker and cause a few benign `PASS_TO_PASS` failures (test pollution) — **not**
  a build defect. The pipeline flags these as **`ordering_artifact`** (gold fix
  applies, all `FAIL_TO_PASS` pass, only `PASS_TO_PASS` regress) so they're
  distinguishable from real failures.
- The published **`:latest` images drift** — they now ship newer base packages (e.g.
  **setuptools 68**, newer **pytest**) than these strict old test suites tolerate, and
  astropy's *programmatic* warnings-as-errors promotes the resulting deprecation
  warnings to failures. Two manifestations: setuptools' distutils `DeprecationWarning`
  breaks **collection** (the module won't import → `collection_error`, e.g.
  astropy-8872), and pytest's nose-`setup()` deprecation breaks tests at **run time**
  (→ `warning_error`, e.g. astropy-8707). This hits the **official Docker harness too**
  (it's the image's env, not the runtime) and **can't be cleanly suppressed** by env/CLI
  warning filters. So we **don't silently hack around it** — the pipeline classifies
  these distinctly and reports them, kept separate from genuine failures. (Recovering
  them means pinning older image deps, which diverges from the published image; out of scope.)

---

## 3. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # makes the `e2b_swebench` package importable from scripts
# pip install -e '.[muse]'       # optional: only to GENERATE predictions (§4e)
# pip install -e '.[dev]'        # optional: to run the offline tests — `pytest tests/`
export E2B_API_KEY=...           # required (or put it in a .env file)
# export HF_TOKEN=...            # optional: faster dataset downloads
```

Validated with `e2b 2.26`, `swebench 4.1`, `datasets 5.0`. Dataset defaults to
**SWE-bench_Verified** (500 instances, full x86_64 image coverage).

### Choosing a dataset

Pick the dataset with `SWEBENCH_DATASET` (and `SWEBENCH_SPLIT`, default `test`):

```bash
SWEBENCH_DATASET=princeton-nlp/SWE-bench_Lite \
  python scripts/build_and_verify.py --all --batch-size 25 --stop-on-fail
```

| variant | HuggingFace id | test size | when to use | works with this project? |
| --- | --- | --- | --- | --- |
| **Verified** *(default)* | `princeton-nlp/SWE-bench_Verified` | 500 | human-validated, non-flaky — the standard leaderboard set | ✅ 100% prebuilt x86_64 images |
| **Lite** | `princeton-nlp/SWE-bench_Lite` | 300 | cheap/fast iteration; a curated easier subset | ✅ covered (subset of Full) |
| **Full** | `princeton-nlp/SWE-bench` | 2,294 | the complete set — comprehensive but slower & noisier | ✅ ~99.8% image coverage (a few unbuilt) |
| **Multimodal** | `SWE-bench/SWE-bench_Multimodal` | 500 | JS/visual issues with screenshots | ❌ cloud-only (`sb-cli`); private test split, no public images — different harness |
| **Multilingual** | `SWE-bench/SWE-bench_Multilingual` | 300 | 9 programming languages / 42 repos | ⚠️ newer & non-Python; `swebench` can grade it, but **verify prebuilt-image coverage** before a full run |

**Recommendation:** start with **Verified** (the default). Drop to **Lite** for a
cheaper smoke; use **Full** when you need everything. **Multimodal** requires the
cloud `sb-cli` path (out of scope for this per-instance-template project), and
**Multilingual** is newer/experimental here.

Notes:
- Both HuggingFace orgs work for the core sets — `princeton-nlp/…` (legacy, the
  default here) and `SWE-bench/…` (the newer canonical org, which also hosts the
  newest variants like Multilingual).
- All variants evaluate on the `test` split; some also ship a smaller `dev` split.
- *"SWE-bench Pro"* is a separate third-party benchmark — not part of this family
  and not loadable through this project.

---

## 4. How to run

Run from the repo root with the venv active and `E2B_API_KEY` set. Use
`python -u` for live progress on long background runs.

### a) Smoke test — start here (1 instance, ~3 min)

Builds one template, checks the environment, runs the **gold** patch (must
resolve) and an **empty** patch (must not).

```bash
python scripts/smoke_test.py                      # default: astropy__astropy-12907
python scripts/smoke_test.py sympy__sympy-20438   # a specific instance
```

### b) Build + verify the whole dataset, in resumable batches — the main path

Builds templates in batches and gold-verifies each batch, recording every
outcome in a ledger you can **stop and resume across sessions**.

```bash
# everything, batches of 25 — halt on any non-clean batch so you can investigate
python scripts/build_and_verify.py --all --batch-size 25 --stop-on-fail

# only a few batches now (e.g. tonight), continue later
python scripts/build_and_verify.py --all --batch-size 25 --max-batches 4

# check progress any time (reads the ledger, doesn't touch a running job)
python scripts/build_and_verify.py --status

# a quick cross-repo sample instead of the full set
python scripts/build_and_verify.py --per-repo 2 --batch-size 12
```

| option | default | meaning |
| --- | --- | --- |
| `--all` / `--per-repo N` / `--limit N` / `--instances a,b` | — | which instances (choose one) |
| `--batch-size` | 25 | instances per batch (build + verify, then ledger save) |
| `--workers` | 8 | parallel template builds (processes) |
| `--verify-concurrency` | 20 | concurrent verify sandboxes (E2B free-tier cap; raise on paid tiers) |
| `--max-batches` | — | stop after N batches this session |
| `--cpu` / `--memory-mb` | 8 / 16384 | template resources |
| `--stop-on-fail` | — | halt after any batch that isn't all `pass`/`ordering_artifact`, to investigate before continuing |
| `--ledger` | `results/ledger.json` | progress file |
| `--status` | — | print ledger summary and exit |

Ledger verify categories: **`pass`** (resolved), **`ordering_artifact`** (template
good; benign test-ordering), **`collection_error`** / **`warning_error`** (template
good; upstream image drift at import- / test-time — see §2), **`fail`** (genuine
failure to investigate), **`error`** (transient → retried on resume). With
`--stop-on-fail` the run halts **only on a genuine `fail`**; pass / ordering_artifact /
collection_error / warning_error are tallied and skipped on resume.

### c) Build templates only

```bash
python scripts/build_templates.py --per-repo 1          # one per repo (~12)
python scripts/build_templates.py --limit 20
python scripts/build_templates.py --instances astropy__astropy-12907,sympy__sympy-20438
python scripts/build_templates.py --all --workers 8     # all 500
```

Options: `--instances` / `--per-repo N` / `--limit N` / `--all`, `--workers`
(default 4), `--cpu`, `--memory-mb`, `--force` (rebuild even if it exists).

### d) Evaluate — gold sanity, or a real model's predictions

```bash
# gold across a cross-repo sample, building any missing templates first
python scripts/run_eval.py --gold --per-repo 1 --build

# gold across all 500 (--concurrency defaults to 20 = E2B free-tier cap; raise on paid tiers)
python scripts/run_eval.py --gold

# score a model's predictions, resuming if interrupted
python scripts/run_eval.py --predictions preds.jsonl --concurrency 20 --build --resume
```

| option | default | meaning |
| --- | --- | --- |
| `--gold` \| `--predictions PATH` | — | **required**: gold patches, or a predictions file |
| `--instances` / `--limit N` / `--per-repo N` | — | restrict the set |
| `--build` | off | lazily build missing templates first |
| `--build-workers` | 8 | parallel builds when `--build` is set |
| `--concurrency` | 20 | concurrent eval sandboxes (E2B free-tier cap; raise on paid tiers) |
| `--out` | `results` | output directory |
| `--resume` | off | skip instances already in `<out>/verdicts.jsonl` |

A prediction is one JSON object per line: `{"instance_id", "model_name_or_path", "model_patch"}` — the format the official harness consumes. For a real agent: hand it a sandbox at HEAD + the `problem_statement`, let it edit `/testbed`, and capture all changes relative to the starting HEAD, including newly created files.

### e) Generate predictions with an agent — Meta's Muse Spark

`scripts/run_agent.py` generates prediction patches; it does not grade them. The Muse Spark model loop stays in the caller process, while its commands and file writes run in a fresh E2B sandbox created from each instance's template.

The model receives only the repository name and `problem_statement`. It does not receive `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, the reference patch, or grader output. The generated patch is evaluated later in another fresh sandbox with the existing canonical grader.

```bash
pip install -e '.[muse]'
export META_API_KEY=...          # stays in this process
export E2B_API_KEY=...

# Templates must already exist; see §4c.
python scripts/run_agent.py --instances astropy__astropy-12907

# Grade the generated patch independently.
python scripts/run_eval.py --predictions results/predictions.jsonl --out results/eval
```

Use `python scripts/run_agent.py --list-agents` to list the available agents and their required environment variables.

| option | default | meaning |
| --- | --- | --- |
| `--instances` / `--limit N` / `--per-repo N` | — | **required**: which instances to generate for |
| `--agent` | `muse-spark` | agent used to generate patches |
| `--model` | `muse-spark-1.1` | model id (also `META_MODEL`) |
| `--max-steps` | `30` | maximum model turns per instance |
| `--sandbox-timeout` | `2400` | sandbox lifetime, seconds |
| `--out` | `results/predictions.jsonl` | predictions, in the official format |
| `--status` | `generation.jsonl` beside `--out` | per-instance generation outcome |
| `--skip-preflight` | off | skip the model connectivity check |

Before generation, the script checks that every selected template exists and sends a short model request to catch an invalid key or model id. Generation is sequential, and sandbox cleanup runs even if an instance fails. Cleanup failures are reported in the status output. A failed instance still produces an empty-patch row, so it is graded unresolved instead of being omitted. Permanent API errors stop the batch early.

The defaults follow Meta's public [Model API cookbook](https://github.com/meta-models/meta-model-cookbook): the OpenAI-compatible endpoint is `https://api.meta.ai/v1` and the default model is `muse-spark-1.1`. Override the endpoint with `META_BASE_URL` and the model with `--model` or `META_MODEL`.

**Generation uses both Meta Model API and E2B resources.** Start with one explicit `--instances` id. The implementation is covered by offline tests, but a live end-to-end run requires valid credentials and incurs provider usage.

---

## 5. Configuration

All settings live in `e2b_swebench/config.py` and are overridable via env vars.
The per-script CLI flags (§4) take precedence over these defaults.

| env var | default | controls |
| --- | --- | --- |
| `SWEBENCH_DATASET` | `princeton-nlp/SWE-bench_Verified` | which dataset to load |
| `SWEBENCH_SPLIT` | `test` | dataset split |
| `SWEBENCH_NAMESPACE` | `swebench` | image source — `swebench` pulls the prebuilt images from Docker Hub |
| `SWEBENCH_CPU` | `8` | vCPUs per template (also `--cpu`) |
| `SWEBENCH_MEMORY_MB` | `16384` | RAM per template, MiB, must be even (also `--memory-mb`) |
| `SWEBENCH_SANDBOX_TIMEOUT` | `2400` | sandbox lifetime, seconds |
| `SWEBENCH_CMD_TIMEOUT` | `1800` | per-command (`eval.sh`) timeout, seconds |
| `SWEBENCH_CONCURRENCY` | `20` | **max concurrent sandboxes = E2B free-tier cap**; raise on paid tiers (also `--concurrency` / `--verify-concurrency`) |

Notes:
- Arch is fixed to **x86_64** (E2B is amd64; SWE-bench's arm64 images are incomplete).
- **Template-build parallelism** is a separate knob — `--workers` (default 4 in
  `build_templates.py`, 8 in `build_and_verify.py`); it stays below the concurrency cap.

---

## 6. Outputs

| file | written by | purpose |
| --- | --- | --- |
| `results/ledger.json` | build_and_verify | per-instance build+verify state; **resume source of truth** |
| `results/verdicts.jsonl` | run_eval | one verdict per instance as it completes; resume source |
| `results/report.json` | run_eval | summary + all verdicts |
| `results/predictions.jsonl` | run_eval, run_agent | the predictions that were scored / generated |
| `results/generation.jsonl` | run_agent | per-instance generation status (steps, patch size, sandbox id, error) — **not** graded |

---

## 7. Layout

```
e2b_swebench/
  config.py      dataset, image namespace, resources, timeouts, concurrency
  dataset.py     load instances, parse tests, gold/empty predictions, select_per_repo
  templates.py   template_name, instance_image, ensure_template, build_many (process pool)
  driver.py      run_instance / run_instance_async: write patch → eval.sh → grade
  runner.py      run_many (concurrent, resumable) + summarize
  ledger.py      Ledger + categorize_verdict (pass / ordering_artifact / fail / error)
  logs.py        quiet_logs
  agents/
    __init__.py    AgentSpec + REGISTRY: the `--agent` lookup table
    muse_spark.py  generate predictions with Meta's Muse Spark (optional `[muse]` extra)
scripts/
  smoke_test.py        one-instance sanity check
  build_and_verify.py  batched, resumable build + gold-verify with a ledger
  build_templates.py   build templates only
  run_agent.py         generate predictions with Muse Spark (does not grade)
  run_eval.py          evaluate gold patches or a predictions file
tests/                 offline unit tests (doubles only — no E2B, no model API)
```

### Notes

- **Timeouts in seconds**; sandbox max continuous life is 1 h (Base) / 24 h (Pro).
- Combined **stdout+stderr** is captured to a file, so Django-style stderr results
  and non-zero exits are handled.
- **Fresh sandbox per run**; grading always goes through `get_eval_report` (the
  per-repo parser), never a hand-rolled regex.
- Parallel builds use **processes** — the E2B SDK shares one HTTP/2 connection, so
  thread-based parallelism collides (`invalid_new_stream_id`).
