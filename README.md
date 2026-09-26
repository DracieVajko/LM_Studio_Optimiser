# LM Studio Auto Optimizer

Hardware-aware automatic benchmarking and runtime configuration optimization for LM Studio.
Finds empirically validated inference configurations for your hardware, model and goal —
not theoretical optima.

**Status: working beta, hardware-validated core.** Connect/probe, benchmark suite,
5-stage optimization (+ micro-refinement), 5% non-inferiority selection, preheat
lifecycle, workload model (interactive/throughput), context sweep, interrupt-safe
checkpoint/resume, VRAM matrix, per-model reports and the live web UI are verified
against LM Studio on a 6 GB NVIDIA host (233 automated tests green, E2E-validated on live server). See [Results](#results).

## Measured results (RTX 3060 6 GB, KV Q4, 2026-09-20/21)

Tuned (full pipeline, quality gate passed):

|Model|Best|Speed|Quality|Context|
|-|-|-|-|-|
|qwen3.5-4b (Q4\_K\_M)|ctx 4096, flash on, KV-GPU, batch 1024|66.8 tok/s|0.983 (29/30)|max passing 16384|
|ministral-3-3b (Q4\_K\_M)|ctx 8192, flash on, KV-GPU, batch 512|85.6 tok/s|1.000 (30/30)|max passing 8192|

Fit ceilings, KV-GPU / KV-CPU (load-only ladder, tok/s at 2048 ctx):

|Model|GPU ceiling|CPU ceiling|Speed|
|-|-|-|-|
|qwen3.5-4b / ministral-3b / glint-4b|262144 / 262144 / 262144|same|60–90 tok/s|
|qwen3.8-9b-distill|262144|262144|17.5 / 11 tok/s|
|gemma-4-12b / ministral-14 / deepseek-14|262144 / 262144 / 131072|same|4–6 tok/s|
|gpt-oss-20b / alibaba-30b|131072 / 131072|same|10–13 tok/s|
|qwen3.8-27b|131072 (262144 refused)|131072|\~1.5 tok/s|
|ternary-27b|unloadable (backend refusal at any ctx)|—|—|

Honest blocks (model limitations, not config bugs): glint-4b never emits strict JSON;
marco-8b is too verbose for the conciseness checks; ministral-14-reasoning thinks
aloud into answers. Details in `results/\*-best.md` and per-run summaries.

## Quickstart (Windows)

```bat
start\_optimizer.bat          REM primary Windows launcher (menu: Web UI, checks, models, benchmark, optimize)
run-all.bat                  REM utility: everything overnight (tests + tuning + ceilings)
run-model.bat <key> \[ctx]    REM utility: one model, e.g. run-model.bat qwen3.5-4b 16384
run-web.bat                  REM utility: web UI on http://127.0.0.1:8080
```

Prerequisites: LM Studio running with Developer server on `127.0.0.1:1234`,
KV cache quantization **Q4 starter** + CPU threads at logical maximum (both GUI-only,
REST cannot set them — verified). See `python -m lm\_optimizer recommend --vram 6`.

Manual install:

```bash
pip install -e ".\[dev]"
copy .env.example .env
python -m pytest -q
```

## CLI reference

|Command|What it does|Live load?|
|-|-|-|
|`status`|connection + probed capabilities + hardware|probe only|
|`models`|list local models (size, quant, ctx, MoE)|probe only|
|`inspect <model>`|model detail + supported params|probe only|
|`recommend --vram 6 \[--estimate]`|fit table + GUI/host checklist (+ `lms` estimates)|no|
|`benchmark <model> --context N --style S`|5-test suite, one config|yes|
|`optimize <model> \[--dry-run] \[--workload interactive\|throughput] \[--resume ID]`|full 5-stage tuning + micro-refinement + preset + `.md`|yes|
|`auto \[models] \[--max-context N] \[--kv gpu\|cpu\|both] \[--profile P] \[--workload W]`|smoke → precision → ladder → matrix → optimize → `.md`|yes|
|`fit \[models]`|max-context ladder per KV path → `-fit.md`|yes|
|`ctx <model> \[--min-context N] \[--max-context N] \[--step 500\|1000]`|geometric context sweep + bisect refinement → capacity vs recommendation|yes|
|`resume <run-id> \[--revalidate]`|resume from interrupt checkpoint without re-running completed work|no|
|`checkpoints`|list interrupt checkpoints available for resume|no|
|`param-matrix \[--record]`|parameter control matrix (+ DB snapshot)|no|
|`apply / restore / presets / runs`|presets and history management|apply/restore yes|
|`python hw\_monitor.py \[--model M]`|VRAM/RAM/swap snapshot; optional tok/s probe|only with `--model`|
|`python vram\_matrix.py <model>`|12-config ctx×flash×KV fit/speed matrix|yes|

Benchmark styles: `precise` (code/math temp 0.1), `balanced` (suite defaults),
`creative` (summarization temp 0.9). Every command first unloads stale models and
verifies empty state; every load/unload is server-state verified.

## Web UI

```bash
python -m lm\_optimizer.web\_main   # http://127.0.0.1:8080
```

Dashboard, history, results, settings pages + `/api/\*` (single prefix) + websocket
progress. Pages and read endpoints are smoke-tested (`check\_web.py`).

## How it works

1. **Prepare**: unload-all, snapshot free VRAM/RAM/swap, verify empty, detect hardware.
2. **Probe capabilities**: per-key safe probing (unknown keys 400 without loading) +
one echo-load of the smallest model with unload in `finally`.
3. **Smoke**: small-context load + short generation; failures get classified guidance
(flash/OOM/transient/...) and the model is skipped, never retried blindly.
4. **Ladder**: context escalated per KV path until refusal, then 1024-step bisect
(a PASS/FAIL pair never implies the PASS value is the maximum).
5. **Coarse search** (deterministic sampling, ≤20) → **refinement** → **stage 4**
(eval/physical batch, parallel, checkpoints one-at-a-time) → **validation**.
6. **Quality gate**: 6 heuristic checks × 5 tests; configs count only above the
profile threshold (0.95/0.97/0.99). Benchmarks run `reasoning="off"`
(auto-omitted where unsupported); empty outputs fail loudly.
7. **Report**: preset saved + per-model `.md` (best config with verified/default
verdicts, tried-values table, prune reasons, generation settings + source,
score breakdown, all attempts incl. failed, elapsed time).

## Verified LM Studio contract

Live-verified (bundled server, `lms` 1.3.3.0). Unknown keys 400 **without loading**,
so capabilities re-probe safely on every connect and adapt per version.

**Load** (`POST /api/v1/models/load`, flat params): `context\_length`,
`flash\_attention`, `offload\_kv\_cache\_to\_gpu`, `eval\_batch\_size`,
`physical\_batch\_size`, `parallel`, `context\_checkpoints`,
`reasoning\_budget\_message` (string), `num\_experts` (MoE), full
`speculative\_draft\_\*` family, `echo\_load\_config`. Requested-vs-applied is
compared on every load (mismatches recorded, never silent).

**Rejected via REST** (400): `gpu\_ratio` (use `lms load --gpu`, verified working
with REST benchmark), `rope\_\*`, `keep\_model\_in\_memory`, `try\_mmap`, `seed`,
`n\_threads`, KV-quant types, `unified\_kv\_cache`, `ttl`, `num\_layers`, `gpu`.

**Chat** (native `POST /api/v1/chat`, `input` string): `temperature`, `top\_p`,
`top\_k`, `min\_p`, `presence\_penalty`, `reasoning` (model-dependent sets!),
`max\_output\_tokens`. Rejected: penalties (except presence), `typical\_p`,
`mirostat\_\*`, `stop`, `seed`. `/api/v1/chat/completions` does not exist here.

Details: `docs/PARAMETERS.md`, generated `docs/LM\_STUDIO\_PARAMETER\_MATRIX.md`
(`scripts/generate\_parameter\_matrix.py`), `lm\_optimizer/services/parameter\_registry.py`
(101 parameters: REST/SDK/CLI/schema surfaces + lifecycle
Detected → Supported → Controllable → Applied → Verified).

## Hardware notes

* Designed hardware-agnostic (no hardcoded GPU): NVIDIA/CPU paths, ASCII output,
graceful degradation without `nvidia-smi`/psutil. Live-tested on RTX 3060 6 GB;
CPU-only paths are coded and unit-tested, not physically exercised.
* Remote LM Studio is detected and warned: local hardware must not score it.
* Timeouts scale with model size (loads 600 s, generations scale with max tokens).

## Development

```bash
pip install -e ".\[dev]"
python -m pytest -q                       # 233 tests
python scripts/generate\_parameter\_matrix.py
python .superpowers/sdd/reconstruction/check\_audit.py
```

## Project structure

```
lm\_optimizer/
  api/          FastAPI app, routes (single /api prefix), schemas, websocket
  benchmark/    suite (5 fixed tests) + runner (native chat)
   cli/          16 commands (status..param-matrix, incl. ctx/resume/checkpoints)
  database/     SQLite + migrations (winners, capability snapshots)
  discovery/    model inspection
  domain/       dataclasses (LoadConfiguration filters to server-supported keys)
  hardware/     detection (GPU/CPU/RAM)
  optimizer/    legacy engine (deprecated, see services/optimizer.py)
  profiles/     speed/balanced/context/quality weights
  scoring/      normalization + heuristic evaluator
   services/     clients, benchmark, search space, optimizer, matrix, fit,
                 context\_sweep, selection (5% non-inferiority), preheat
                 (LOAD/VERIFY/PREHEAT/MEASURE), workload
                 (interactive/throughput), run\_summary, reporting,
                 recommendations, generation defaults, hostguard,
                 lms\_cli, parameter\_registry, speculative
  storage/      checkpoints (legacy)
  ui/           templates + static JS (baseURL /api, websocket progress)
hw\_monitor.py vram\_matrix.py run-\*.bat scripts/ docs/ results/ (gitignored)
```

Runtime data (`data/\*.db`, `logs/`, `results/`, `.superpowers/`) is gitignored and
never pushed. No credentials exist in the repo (only `LM\_STUDIO\_URL` default).

## License

MIT — see `LICENSE`.

