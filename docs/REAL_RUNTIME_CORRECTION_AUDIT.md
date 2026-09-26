# Real-Runtime Correction Audit (2026-09-23)

Scope: correctness pass from real LM Studio execution evidence. No unrelated
features added. Verified: **197 collected, 197 passed** (`pytest
--collect-only -q`, `pytest -q`).

## §1 Failed load is never passed (FIXED)

Evidence: `services/optimizer.py::_test_config` scored failed benchmarks
(quality evaluated on empty metrics, `score=0.83`-class artifacts, "Config
passed" logged). Fix: phase gate — any non-passed benchmark is classified
(`failure_states.classify_error`), stored with `score=None`,
`quality=None`, `score_breakdown=None`, logged `LOAD_FAILED`/class, and
returned only for history. "Config passed"/"Candidate accepted" replaced by
`CONFIG_ELIGIBLE`, emitted solely for scored winners.

## §2 Failure state machine (IMPLEMENTED)

`CREATED -> LOAD_REQUESTED -> LOADED -> VERIFIED -> PREHEATED ->
BENCHMARKED -> QUALITY_EVALUATED -> SCORED -> CANDIDATE_ELIGIBLE`, in
`_test_config`. Terminal classes: `LOAD_FAILED`, `VERIFY_FAILED`,
`PREHEAT_FAILED`, `BENCHMARK_FAILED`, `QUALITY_FAILED`, `INCOMPATIBLE`,
`OOM`, `TIMEOUT` (+ legacy `FAILED`). Only PASSED + score-set results reach
`_update_best` (explicit guard), `best_passed`, `alternatives`, Pareto.

## §3 Quality failure (FIXED)

Below-threshold now yields `QUALITY_FAILED` with metrics + quality evidence
preserved, `score=None`, excluded from winners. Distinct from load failure
(`quality=None`, nothing to preserve).

## §4 Hard vs soft (IMPLEMENTED)

`INCOMPATIBLE` (hard lock: unrecognized/unsupported/invalid/model_not_found),
`OOM`/`TIMEOUT`/`LOAD_FAILED`/`BENCHMARK_FAILED` soft/learned,
`QUALITY_FAILED` runtime-excluded but not a runtime failure.
`failure_states.is_hard/is_soft`; `run_summary.failure_breakdown` maps each
new status to its bucket (never collapsed).

## §5 Windows checkpoint (FIXED)

Root cause of WinError 5: unguarded concurrent writers to one file plus no
retry — `os.replace` fails when the destination is briefly held (scanner,
sync tool, racing save). Fix in `storage/run_checkpoint.py`: per-run
mutexes, pid-unique temp names, closed handle before replace, 5x backoff
retry on `PermissionError`, read-back verification with restore of the
previous valid checkpoint on failure, stale-temp cleanup (>24h),
`CHECKPOINT_SAVED`/`CHECKPOINT_SAVE_FAILED` events. Previous checkpoint is
retained whenever replacement fails. Tests: `test_checkpoint_windows.py` (5).

## §6-§7 Model/run identity (VERIFIED + EXTENDED)

Reports already derive filenames from the run's own `model_id`
(`save_best_report`); order-independence proven by test (qwen3.8 / mistral /
qwen3.5 scrambled). Extended: identity block (display/quant/variant/arch/MoE),
run ID on every configuration asserted, report content asserts model text.
No global mutable current-model in report paths.

## §8-§11 Control adapter + GPU offload (IMPLEMENTED)

New `services/control_adapter.py`: `ControlChannel`
(REST/CLI/SDK/DETECTED_ONLY/MANUAL_ONLY/UNSUPPORTED) derived from the
registry (never from llama.cpp existence). `BenchmarkService(gpu_via_cli=...)`
loads gpu_ratio via `lms load --gpu` when enabled + available, else REST with
the ratio recorded-but-not-applied; CLI failure falls back to REST. Channel
recorded per config (`generation["load_channel"]`). CLI flags
`--gpu-via-cli/--no-gpu-via-cli` on `optimize`/`auto` (auto-on when `lms`
present, with notice). `refine_gpu_boundary` implements the adaptive
0.7/1.0 -> 0.85/0.925/... bisection (tested, wired for boundary work).

## §12 Estimate before load (IMPLEMENTED, HONEST)

`lms_cli.estimate_for(model, context, gpu)` (version-dependent flags; UNKNOWN
on refusal). `ctx` runs an estimate precheck: refusal stops escalation, the
interval is still refined with real empirical loads (marked `FAIL*`
estimate/unloaded in the report). `prune_by_estimate` keeps the boundary
candidate for proof. Estimates are recorded, never proof.

## §13 Requested vs applied (IMPLEMENTED)

`control_adapter.verify_applied` -> MATCH/PARTIAL_MATCH/MISMATCH/UNKNOWN,
recorded per benchmark in `generation["load_verification"]` (REST echo) and
surfaced in the report ("Requested vs applied" + gpu_ratio-not-applied note).

## §14-§20 Registry truth (IMPLEMENTED)

Added `llama_cpp_arguments_override` (GUI-documented, programmatic UNKNOWN —
lifecycle reflects control only); version-dependent notes on
`physical_batch_size`/`parallel`/`context_checkpoints`; CPU threads split
(`n_threads`/`cpu_threads`/`cpu_thread_pool_size`, threads MANUAL_ONLY);
KV quants MANUAL_ONLY; speculative fields separate and never auto-enabled
(`optimizer_enabled=False`); MoE `num_experts` vs CPU-expert ratio separated;
batch params distinct. Matrix regenerated: **102 parameters**.

## §21-§23 Context (VERIFIED)

Geometric -> bracket -> bisect refinement (500/1000) proven by test;
`_context_trio` reports stable/optimal/balanced from measured data;
context-profile 5% test (67@4K vs 63.8@12K -> 12K) green.

## §24 Quality variance (INVESTIGATED)

Evaluator has no randomness (imports: json/re/dataclasses only) and is
deterministic per text (test). Score variance across identical configs is
model sampling output variability (per-case temps frozen, suite temps
asserted frozen); not interpreted as runtime degradation. Root fix: vacuous
`aggregate_quality({}) == 1.0` can no longer pass failed loads (phase gate
skips quality eval entirely on benchmark failure).

## §25-§26 Score eligibility + Pareto (ENFORCED)

Score exclusively after load+verify+benchmark+quality; else None.
`_compute_pareto_frontier` excludes non-PASSED, unscored, quality-less
results; non-empty for valid sets proven by deterministic test.

## §27 UI verdict (FIXED)

"default won" replaced with "default retained because no tested alternative
exceeded the preference threshold"; N/A only when nothing decided.

## §28 Launcher (IMPLEMENTED)

`start_optimizer.bat`: venv create/activate, dependency verify, menu (Web UI +
browser / check / models / benchmark / optimize / exit). No auto huge runs.
README documents `run-*.bat` as utilities.

## §29-§30 Dashboard + live progress (AUDITED)

All documented routes exist (`/`, `/history`, `/results/{id}`, `/settings`,
`/api/status|models|runs|...`, `/ws/optimize/{run_id}` — note: FastAPI 0.141
mounts routers as deferred `_IncludedRouter`; tests expand them). Dashboard
shows status badge + hardware + models + activity; added explicit red banner
when LM Studio is disconnected (was console-only). No `/api/results` route
exists and the frontend never calls it (uses `/runs/...`) — documented, not
invented. Live progress carries stage/model/config/tested/passed/failed/OOM/
speed/VRAM/RAM/context/elapsed/adaptive-remaining; failed counters now cover
all terminal classes.

## §31 Logging (IMPLEMENTED)

Structured events `LOAD_REQUESTED/SUCCEEDED/FAILED`, `VERIFY_SUCCEEDED`,
`PREHEAT_STARTED`, `BENCHMARK_STARTED/COMPLETED`, `QUALITY_FAILED`,
`CONFIG_ELIGIBLE/REJECTED`, `OOM`, `CHECKPOINT_SAVED/FAILED`. "Config passed"
no longer emitted for ineligible configurations.

## §32 Storage/swap (RETAINED)

`hostguard.storage_info` (Windows drives NVMe/SSD/HDD + pagefile; detection
only) now flows into `_host_info_dict` and the report ("Storage / swap"
section). Swap never modified; used for interpretation only.

## §33-§34 Reports (EXTENDED)

Filename from canonical run `model_id`; content now includes run ID, full
model identity, backend/LM Studio version, hardware, storage, baseline,
control channels, requested vs applied + verification, failures, context
trio, crowns, correctness, stability. `score:.3f` crash guards for None.

## §35 Matrix (KEPT GENERATED)

`docs/LM_STUDIO_PARAMETER_MATRIX.md` regenerated from the single registry
(102 rows). GUI availability lives in behavior text; lifecycle reflects
programmatic control only (per §38 ladder).

## §36 Tests (all green, none weakened)

New: `test_failure_states.py` (17), `test_checkpoint_windows.py` (5),
`test_identity.py` (4), `test_control_adapter.py` (28),
`test_dashboard_api.py` (5); extended `test_selection.py` (+2 context
tolerance), `test_workload.py` (measured semantics), `test_preheat.py`
(contamination). One intentional test update: throughput tie-break now
expects measured-first semantics (the old assertion encoded the audited
anti-pattern; documented in-test). Total **197**.

## §37 Validation

- `pytest --collect-only -q`: 197 collected
- `pytest -q`: 197 passed (2 pre-existing third-party deprecation warnings)
- `ruff check lm_optimizer`: **1014** = 531 legacy + 392 new files + 91 edited
  net (method: saved 531-error transcript diffed per file/rule against
  `--output-format json`; untouched files byte-identical in count, so no
  config/cache influence; +6 vs the 1008 draft from 4 added E2E tests).
  New service files `failure_states.py` and `control_adapter.py` are
  ruff-clean (0); new violations elsewhere are the same rule families as
  baseline (ANN/PLC0415/PLR2004). Fixed where safe: F821 (`min_ctx`
  NameError), F401 x6, UP/TC/PTH/RET/ARG mechanicals.
- `ruff format --check lm_optimizer`: 14 files, all legacy/untouched
- `mypy lm_optimizer`: blocked repo-wide by missing third-party stubs
  (pre-existing); new service files clean
- `python -m compileall lm_optimizer`: OK

## §38 Capability states

DETECTED -> SUPPORTED -> CONTROLLABLE -> APPLIED -> VERIFIED ladder enforced
in `channel_of` + `lifecycle_of`; VERIFIED requires request->apply->echo
evidence. No parameter claims control from mere existence.

## §39 Git

Not a git repository (`git status` fatal) — nothing committed or pushed, per
instruction. Full file inventory of this pass: new
`services/{failure_states,control_adapter}.py`,
`tests/test_{failure_states,checkpoint_windows,identity,control_adapter,dashboard_api}.py`,
`start_optimizer.bat`, `docs/REAL_RUNTIME_CORRECTION_AUDIT.md`; extended
domain/optimizer/benchmark/lms_cli/reporting/run_summary/routes/websocket/
context_sweep/CLI/UI/README/matrix.
