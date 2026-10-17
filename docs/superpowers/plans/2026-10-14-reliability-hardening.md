# Reliability Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make experimental flags explicit, pre-flight unified before any model load, winner/context/preset consistent, durations persisted, and fast exploration truthfully labeled — without running real models and without committing.

**Architecture:** Minimal edits to existing modules (`cli/main.py` flag declarations + pre-flight order, `services/optimizer.py` flag parsing + duration stamping + validation-copy override, `services/benchmark.py` duration stamping, reporting/preset round-trip test, `start_optimizer.bat` de-duplication). No new architecture; reuse existing services, repos, checkpoint code.

**Tech Stack:** Python, Typer 0.27.2 / Click 8.5.0 (paired `--x/--no-x` with `None` default verified working), pytest mock-only.

**Spec:** Task text in conversation (P0 flags / pre-flight / winner-context-preset / DB-checkpoint-resume; P1 pause-resume / fast-exploration / Mixtral diagnostics / docs). Evidence: subagent trace reports (winner flow, pause/skip/bat audit).

## Global Constraints

- Do NOT commit or push. Do NOT change branches, reset tree, discard user changes.
- Do NOT delete historical runs, DB rows, JSON exports, presets, checkpoints, logs.
- Do NOT launch real model benchmarks; do NOT download models.
- Do NOT change quality prompts/evaluator semantics/thresholds/weights/scoring/winner policy/Phase A strategy unless a minimal correctness fix requires it (document + test).
- Do NOT weaken quality gates; do NOT mark unvalidated configs as quality-safe winners.
- Preserve historical checkpoint compatibility and CLI invocation patterns.
- Reuse existing modules/services/options/UI/persistence before new architecture.
- Live optimizer process is running (PID 5276): do NOT terminate it, do NOT mutate `data/optimizer.db` or active checkpoints; DB reads only in `mode=ro`.
- Never claim a test/benchmark/action passed unless actually executed and observed.

## Review Focus

- `--enable-rope False` (bool flag + positional) must exit 2, not silently enable + add model `False`. Test pins it.
- Partial opt-in `--enable-rope` alone must still prompt for the other two when interactive, stay OFF when not. Test pins it.
- Speed 4096 / quality 32768 winner must export nested==outer==preset==applied 32768 everywhere. Test pins it.
- Completed `quality_failed` must be skipped (not re-run) on resume. Test pins it.
- Skip-quality run must never produce a validated winner or overwrite `-best.md`. Test pins it.

---

### Task 1: Boolean flag semantics (auto + optimize)

**Files:**
- Modify: `lm_optimizer/cli/main.py` (flag declarations ~1858-1872, ~1113-1127; `_resolve_experimental_flags` ~442-460; `_prompt_experimental_optins` ~423-439)
- Modify: `lm_optimizer/services/optimizer.py` (`parse_experimental_flags` ~56-70)
- Test: `lm_optimizer/tests/test_experimental_optins.py` (extend)

**Interfaces:**
- Consumes: Typer paired bool `bool | None = typer.Option(None, "--enable-rope/--no-enable-rope")` (verified: omitted→None, +→True, −→False, both→last wins).
- Produces: `resolve_experimental(requested: dict[str, bool|None], interactive: bool) -> dict[str, bool]` — explicit wins; unspecified→prompt-if-interactive else OFF; `parse_experimental_flags` keeps `(bool,bool,bool)->dict` signature (None coerced to False) for compat.

- [ ] **Step 1:** Write failing tests: omitted-noninteractive→all OFF; each + individually; each − individually; partial + (`rope=True, others None`, interactive=False→OFF, interactive=True→prompt called for the two); `rope=True+cpu_moe=False` mixed; both + and − same flag → last wins deterministic; `--yes` never enables; non-interactive never calls prompt (patch `typer.confirm`, assert not called); EOF/TTY-unavailable → OFF.
- [ ] **Step 2:** Run: `pytest lm_optimizer/tests/test_experimental_optins.py -v`. Expected: FAIL (new names missing).
- [ ] **Step 3:** Implement: change 6 declarations to paired `None`-default; rework `_resolve_experimental_flags(enable_rope: bool|None, ...)` + `non_interactive: bool` param; per-flag prompting (only `None` ones); keep `parse_experimental_flags` coercing None→False.
- [ ] **Step 4:** Run: `pytest lm_optimizer/tests/test_experimental_optins.py -v` + full file green. Expected: PASS.
- [ ] **Step 5:** No commit (global constraint).

### Task 2: Boolean-positional guard + --yes/--non-interactive

**Files:**
- Modify: `lm_optimizer/cli/main.py` (`auto`, `optimize` signatures + validation block)
- Test: extend `lm_optimizer/tests/test_experimental_optins.py` or `test_cli_skip.py` (whichever owns CLI validation; check first)

**Interfaces:**
- Produces: `_reject_bool_like_model_args(models: list[str]) -> None` — exits 2 on `true/false` (any case); `--yes/--non-interactive` flags on both commands.

- [ ] **Step 1:** Failing tests: `auto --enable-rope False` (CliRunner) exits 2 and never touches backend (patch client, assert no connect); `optimize False` exits 2 pre-connect; `--yes` resolves unspecified experiments OFF without prompting; `--non-interactive` never prompts even on TTY (patch isatty True + confirm, assert confirm uncalled).
- [ ] **Step 2:** Run new tests. Expected: FAIL.
- [ ] **Step 3:** Implement: add `--yes/--non-interactive` options; call guard before any `get_client()/connect`; thread `non_interactive or yes` into resolver as non-interactive.
- [ ] **Step 4:** Run tests green.
- [ ] **Step 5:** No commit.

### Task 3: Unified pre-flight before connect

**Files:**
- Modify: `lm_optimizer/cli/main.py` (`auto` ~1886-1980, `optimize` ~1154-1240)
- Test: new `lm_optimizer/tests/test_preflight.py` (CliRunner + mocks)

**Interfaces:**
- Produces: `_preflight_auto(...) -> dict | NoReturn`, `_preflight_optimize(...) -> dict | NoReturn` — pure validation+resolution returning effective settings; `print_preflight_summary(settings)` — single table; confirmation via `typer.confirm("Start optimization?", default=True)` unless `--yes/--non-interactive/dry-run`.

- [ ] **Step 1:** Failing tests: invalid `--skip-phases`, bad `--physical-batches`, bad `--warmup-runs`, bool-like model arg → exit 2 with mocked client asserting `connect` never called; summary output contains model/profile/phases/experiments; pre-flight rejection performs zero backend calls.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement: move all local validation + experimental resolution + summary + confirm above `_connect_guarded*`; `auto` shows explicit models or `all (small-first)` without enumerating (no server call); `optimize` same resolver semantics.
- [ ] **Step 4:** Run green (new + existing CLI tests).
- [ ] **Step 5:** No commit.

### Task 4: BAT launcher de-duplication

**Files:**
- Modify: `start_optimizer.bat` (exact lines from subagent-B report)
- Test: manual inspection only (batch file; assert Python CLI never re-asks values passed as args — covered by Task 3 tests)

- [ ] **Step 1:** Keep task-selection menu; replace profile/style/etc. questions with pass-through args to one Python pre-flight.
- [ ] **Step 2:** Verify: `cmd /c` parse check or visual diff of questions asked (no execution of models).
- [ ] **Step 3:** No commit.

### Task 5: Experimental metadata precision

**Files:**
- Modify: `lm_optimizer/services/optimizer.py` (`prepare_run`, tail stage), `lm_optimizer/domain/models.py` (optional detail field), `lm_optimizer/api/routes.py` (export), `lm_optimizer/services/reporting.py` (requested-vs-applied section)
- Test: extend `test_experimental_optins.py` + `test_reporting_transparency.py`

**Interfaces:**
- Produces: `experimental_detail: {requested, resolved, applied, exercised, skipped: {flag: reason}}` stored in `run.benchmark_params["experimental_detail"]`; `is_experimental` keeps meaning "any experiment resolved ON at request time" (document in docstring); speculative-without-draft → `skipped: {enable_speculative: "no usable draft model"}` and never `exercised`.

- [ ] **Step 1:** Failing tests: detail dict present after prepare_run; speculative enabled but no draft → skipped reason, `exercised` False; reports show requested vs applied distinctly; legacy run without detail still loads.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement minimal: populate requested/resolved at prepare_run; applied/exercised at tail stage from capability checks; JSON+MD report sections.
- [ ] **Step 4:** Run green.
- [ ] **Step 5:** No commit.

### Task 6: Winner/context/preset invariant

**Files:**
- Modify: `lm_optimizer/services/optimizer.py` (`_phase_final_validation` ~2531-2535: add `vcfg.context_length = best.context_length` override)
- Test: new `lm_optimizer/tests/test_winner_context.py` (fake configs + backend mocks: speed 4096, quality 32768, pass, export winner, save preset, read back, apply reconstructs 32768; mismatch case; failed-quality never winner; skip-quality never validated)

**Interfaces:**
- Consumes: existing `_test_config`, preset repo, apply reconstruction.

- [ ] **Step 1:** Failing tests as above (before fix, validation copy drops override — simulate by constructing divergent nested/outer pair through the copy path).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement one-line override + doc comment; no other flow changes (trace proved no live leak elsewhere).
- [ ] **Step 4:** Run green.
- [ ] **Step 5:** No commit.

### Task 7: duration_ms persistence

**Files:**
- Modify: `lm_optimizer/services/benchmark.py` (`run_cases` ~361-430, `_aggregate_results` ~709-792) and/or `optimizer.py` `_test_config` (~1338-1460)
- Test: extend `test_optimization_correctness.py` or new `test_duration.py` (mock benchmark with controlled clock)

**Interfaces:**
- Produces: every persisted `ConfigurationResult` from a started attempt carries `duration_ms > 0` (success + load_failed + too-slow + quality_failed); never-started attempts keep existing nullable/0 convention documented.

- [ ] **Step 1:** Failing tests: success stamps duration; load failure stamps duration; exception path stamps duration.
- [ ] **Step 2:** Run. Expected: FAIL (all zeros today).
- [ ] **Step 3:** Implement: `time.perf_counter()` around attempt in one choke point (prefer `_test_config` wrapper so all paths covered); ensure exception paths stamp before save.
- [ ] **Step 4:** Run green. Historical zeros untouched.
- [ ] **Step 5:** No commit.

### Task 8: Checkpoint/DB reconciliation + stale-running diagnostics

**Files:**
- Modify: `lm_optimizer/services/optimizer.py` (only if trace finds a gap; baseline says longer-log-wins + DB-authoritative already correct → tests only), `lm_optimizer/cli/main.py` (`resume-menu` listing: last-progress, checkpoint presence, phase/cursor, safe-to-resume hint; never mass-reset)
- Test: new `lm_optimizer/tests/test_resume_reconcile.py` (DB newer than checkpoint; checkpoint newer cursor-only; stale checkpoint cannot erase; quality_failed skipped; interrupted resumed; duplicate resume guarded; legacy checkpoint readable)

- [ ] **Step 1:** Failing tests for any gap found; if all pass already, keep as characterization tests.
- [ ] **Step 2-4:** RED→GREEN per TDD.
- [ ] **Step 5:** No commit.

### Task 9: Pause/resume hardening + tests

**Files:** per subagent-B findings (CLI/API/UI/background — modify only demonstrated gaps)
**Tests:** pause during SPEED/QUALITY/RECOVERY/validation; durable checkpoint; same-ID resume; no re-run of completed; no duplicate workers.

- [ ] **Steps:** TDD per behavior; no live models.
- [ ] **Step:** No commit.

### Task 10: Fast exploration semantics

**Files:** reuse `--skip-phases` + `ctx`; modify reporting (`save_best_report` guard: skip-quality → no `-best.md` overwrite, exploratory report instead, `quality_validated=false` explicit)
**Tests:** no quality suite on explore; skips preserved on resume; exploratory cannot become validated winner; exploratory cannot overwrite validated report; rung statuses match execution.

- [ ] **Steps:** TDD per behavior.
- [ ] **Step:** No commit.

### Task 11: Mixtral repeated-failure diagnostics

**Files:** grouping by normalized error signature + circuit-breaker (only if no existing policy; reuse if present) + timing instrumentation display (load/gen/quality/recovery/total from `duration_ms` after Task 7 — display only, no new benchmark work)
**Tests:** signature grouping; breaker trips only on repeated identical errors across materially different configs with no baseline success; flash-attention-specific failure does NOT trip global breaker.

- [ ] **Steps:** TDD.
- [ ] **Step:** No commit.

### Task 12: Docs + full validation + report

**Files:** `README.md` (version + test count), docs for flags/pre-flight/exploration/pause-resume/context claims (minimal edits, no full rewrite)
**Validation:** targeted tests → full `pytest lm_optimizer/tests -q` → `ruff` on changed files → `compileall` → `--help` smoke of affected commands. Honest counts.

- [ ] **Step 1:** Update version/test-count only after suite runs.
- [ ] **Step 2:** Final report (11 items per spec §Stage 6). No commit/push.
