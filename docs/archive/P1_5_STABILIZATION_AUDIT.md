# P1.5 Stabilization Audit (2026-09-22)

No new optimization features. Verified state: **136 collected, 136 passed,
0 failed, 0 skipped, 0 warnings** (`pytest --collect-only -q`, `pytest -q`).

## 1. Lint baseline vs current — exact reconciliation

Ruff configuration: unchanged (`pyproject.toml:41-48`; 22 rule families,
line-length 100, target py311). No config change, so no config-induced
difference. Cache/generated files do not affect counts (verified: every
untouched file reports identical counts in both runs).

Method: the previously reported 531-error output was recovered from the saved
tool transcript and parsed per file/rule; current state captured with
`ruff check lm_optimizer --output-format json` and diffed with normalized paths.

| | Count |
|---|---|
| True pre-existing baseline (32 files) | 531 |
| New files with violations (7 files) | 251 |
| Net delta in edited existing files | +73 |
| Total now | **855** |

531 + 251 + 73 = 855. Exact. (Intermediate totals 853/858/866/882 seen during
the task were the tree changing between runs as fixes and features landed —
per-state counts are deterministic. The "88 new files" figure was superseded
by rewrites, `ruff format`, and new throughput tests; 3 further new files —
`selection.py`, `preheat.py`, `workload.py` — are ruff-clean at 0.)

Per-file delta (edited/new only; all other files byte-identical in count):

| File | Base | Now | Delta |
|---|---|---|---|
| `services/optimizer.py` | 27 | 68 | +41 |
| `cli/main.py` | 73 | 92 | +19 (incl. F821 fix) |
| `api/routes.py` | 41 | 55 | +14 (incl. F401 fix) |
| `services/benchmark.py` | 13 | 18 | +5 |
| `api/main.py` | 7 | 8 | +1 |
| `services/search_space.py` | 1 | 2 | +1 |
| `api/schemas.py` | 3 | 2 | -1 (pydantic fix) |
| `api/websocket.py` | 12 | 5 | -7 (rewrite) |
| `domain/models.py` | 10 | 10 | 0 |
| `services/context_sweep.py` (new) | 0 | 1 | +1 |
| `services/run_summary.py` (new) | 0 | 8 | +8 |
| `storage/run_checkpoint.py` (new) | 0 | 17 | +17 |
| `tests/test_p1_ux.py` (new) | 0 | 123 | +123 |
| `tests/test_selection.py` (new) | 0 | 28 | +28 |
| `tests/test_preheat.py` (new) | 0 | 22 | +22 |
| `tests/test_workload.py` (new) | 0 | 52 | +52 |
| `services/selection.py`, `services/preheat.py`, `services/workload.py` (new) | 0 | 0 | 0 |

Dominant new-violation rules and disposition:

- PLC0415 (+102 across new code): lazy function-level imports — the repo's
  deliberate circular-import avoidance pattern (baseline already 38). Fixing
  would reintroduce import cycles. Left as-is.
- ANN201/ANN202 (+96): missing annotations, mostly in tests — same style as
  legacy tests (`test_optimization_correctness.py` alone has 98). Left as-is.
- PLR2004 (+42): magic values in tests — same as legacy tests. Left as-is.
- TRY301 (+8), PLR0913/17, C901, E501, TRY003, SIM105, PERF401: same
  categories as baseline. Left as-is (fixing = huge unrelated diff).
- Fixed (safe, no logic change): F821 (real `min_ctx` NameError bug in `ctx`),
  F401 x5 (dead imports incl. `RunProgressResponse`), UP035/UP042/UP037,
  PTH105/108/110 (-> `Path` methods), RET501/PLR1711/ARG005, mypy type args.
  No global suppressions, no broad `noqa`, CI not weakened.

## 2. Mypy

- New service files (`selection.py`, `preheat.py`, `workload.py`): clean.
- My additions inside `benchmark.py`/`optimizer.py`: no new mypy errors
  (remaining hits there are legacy lines).
- Repo-wide `mypy lm_optimizer`: NOT clean at baseline and still not clean —
  missing third-party stubs (`psutil`, `GPUtil`, `wmi`, numpy `.pyi` vs
  `python_version 3.11`) block checking, plus strict-mode debt in legacy
  files (`domain/models.py`, `api/client.py`, ...). Pre-existing, reported.

## 3. Workload measurement (throughput audit)

Finding: `workload.aggregate_throughput()` derived aggregate as
`per_request_tps * parallel` — an unproven inference. Replaced on the
decision path:

- `BenchmarkService.measure_throughput()` (new, `services/benchmark.py`):
  one shared load, N concurrent `chat_completion` calls via
  `asyncio.gather`, each timed; always unloads. Records parallelism, total
  tokens, wall-clock elapsed, `aggregate = total_tokens / wall_s`,
  per-request tok/s + elapsed, failures + errors, queue/wait approximation
  (wall minus fastest request, documented as approximation).
- `workload.summarize_throughput()` (new, pure/deterministic) + recorded
  block `result.generation["throughput_measured"]` (persisted via config_json).
- Decision path (`workload_tiebreak_key`, throughput branch) now uses
  `effective_throughput` = measured aggregate when recorded, else the
  single-request rate (parallel=1 semantics — deliberately NOT multiplied).
- Optimizer probes THROUGHPUT configs with parallel > 1 in stage-4/micro
  (`_maybe_measure_throughput`, cancel-aware, bounded).
- `aggregate_throughput()` kept as documented legacy estimate for
  compatibility (existing test unchanged); new tests cover measured semantics.
- Tests: 5 pure `summarize` tests (exact formula incl. failures/zero-wall),
  3 mocked concurrent-probe tests (totals, failure counting, load-failure),
  3 tie-break tests (measured wins; fallback unmultiplied).

## 4. Preheat integrity

Verified + tested (`TestNoContamination`): warmup uses a distinct prompt and
its outputs are discarded — measured metrics contain no warmup text and
`len(metrics) == repetitions x suite`; `load_time_ms` covers only the load
call; `generation["preheat"]` carries `warmup_time_ms` separately; TTFT comes
only from measured-run stats (preheat records no `estimated_ttft`).

## 5. Checkpoint integrity

`TestCheckpoint` (5) + `TestResume` (2) green: required-keys payload,
cancel-before-ack, Ctrl+C -> INTERRUPTED, corrupted -> `ValueError`,
missing -> None, resume skips completed candidates (`test_resume_skips_completed`),
atomic tmp+rename writes. Live CLI `checkpoints` returns empty cleanly.

## 6. UI integrity

Backend `/ws/optimize/{run_id}` exists; frontend `results.js::connectLive`
subscribes on live runs with reconnect (`websocket.js`), renders stage, model,
current config, tested/passed/failed/OOM, best, speed, VRAM/RAM, context
progress, elapsed, `adaptive / unknown` remaining, plus compact live log with
expandable details (`TestWebsocket` 3 green; JS reconnect verified by
inspection — not pytest-coverable).

## 7. Zero/partial + ctx CLI

`TestFinalStates` (8) green: FAILED with reason classes and no Apply offer,
PARTIAL counts, failed configs never win. `ctx --dry-run` prints the ladder;
`--step` rejects non-500/1000; `TestContextSweep` proves 4096 PASS / 8192 FAIL
probes inside the interval. (This audit additionally fixed a latent F821
`min_ctx` NameError on the live `ctx` path — dry-run masked it.)

## 8. Remaining limitations

- Ruff: 855 at P1.5 close (531 legacy + 251 new files + 73 edited net);
  the real-runtime correction pass stands at **1008**
  (531 + 386 new + 91 edited net; `failure_states`/`control_adapter` service
  files are clean at 0; final `ruff format` removed one layout violation).
  Legacy cleanup is a separate task.
- Mypy repo-wide blocked by missing stubs + legacy strict debt.
- `ruff format --check lm_optimizer`: 14 files would be reformatted — all legacy
  files never touched by this work (`api/client.py`, `database/repositories.py`,
  `discovery/inspector.py`, `optimizer/engine.py`, `scoring/normalization.py`,
  `services/{fit,generation_defaults,hostguard,lm_studio,lms_cli,
  model_recommendations,parameter_registry,reporting,speculative}.py`).
  Every file created or edited by P1/P1.5 is formatted.
- Throughput probe adds wall-clock time per parallel>1 THROUGHPUT config
  (bounded to stage-4/micro candidates only).
- `git status`/`git diff`: not a git repository — nothing to commit or push.
