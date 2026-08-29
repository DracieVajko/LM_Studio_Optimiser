# LM Studio Auto Optimizer

**Hardware-aware automatic benchmarking and runtime configuration optimization for LM Studio.**

Find empirically optimized configurations for your model and hardware based on speed, context, memory efficiency and correctness.

> ⚠️ **v1.0.0-beta — Untested Beta**
>
> Automated tests, type checks and static validation pass, but real-world end-to-end optimization against LM Studio hardware has not yet been validated for this release. Use with caution and verify generated configurations before relying on them.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Version](https://img.shields.io/badge/version-1.0.0--beta-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-61%20passed-brightgreen)
![Status](https://img.shields.io/badge/status-beta-orange)

## Why?

LM Studio exposes many runtime parameters:

* context length
* GPU offload
* Flash Attention
* KV cache placement
* evaluation batch size
* expert count
* RoPE parameters
* and other model/runtime-specific options

The best configuration depends on:

* GPU
* VRAM
* system RAM
* CPU
* model architecture
* quantization
* context length
* LM Studio capabilities

Manually finding the best combination can require many experiments.

LM Studio Auto Optimizer automates this process. It finds **empirically optimized LM Studio configurations based on your hardware, model and selected optimization profile** — not an “objectively best” or “universally optimal” configuration.

## How it works

```
             ┌─────────────────────┐
             │       Web UI        │
             │    / CLI interface  │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   Optimizer Engine  │
             │ Adaptive Search     │
             └──────────┬──────────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
     ┌───────────────┐     ┌───────────────┐
     │ Hardware      │     │ Model         │
     │ Detection     │     │ Discovery     │
     └───────────────┘     └───────────────┘
             │                     │
             └──────────┬──────────┘
                        ▼
             ┌─────────────────────┐
             │    LM Studio API    │
             └─────────────────────┘
```

1. Detect hardware
2. Inspect selected model
3. Discover supported LM Studio parameters
4. Capture baseline configuration
5. Generate candidate configurations
6. Run benchmark suite
7. Evaluate correctness
8. Refine promising configurations
9. Validate finalists
10. Store results and presets

## Features

| Feature                              | Status  |
| ------------------------------------ | ------- |
| Hardware-aware optimization          | ✅       |
| NVIDIA / AMD / Apple / CPU detection | ✅       |
| LM Studio capability discovery       | ✅       |
| Adaptive multi-stage search          | ✅       |
| Speed profile                        | ✅       |
| Balanced profile                     | ✅       |
| Context profile                      | ✅       |
| Quality profile                      | ✅       |
| Custom profile                       | ✅       |
| Baseline comparison                  | ✅       |
| Correctness heuristics               | ✅       |
| Pareto frontier                      | ✅       |
| SQLite history                       | ✅       |
| Presets                              | ✅       |
| Checkpoint / resume                  | ✅       |
| Web UI                               | ✅       |
| CLI                                  | ✅       |
| Remote LM Studio endpoint            | ✅       |
| LLM-as-a-judge                       | Planned |
| Multi-GPU optimization               | Planned |
| Docker GPU support                   | Planned |

## Profiles

Profiles change the optimizer's objective; they do not magically change the underlying model quality.

### Speed
Prioritizes generation and prompt processing speed.

### Balanced
General-purpose optimization balancing speed, context, memory and correctness.

### Context
Prioritizes usable context length while maintaining stable operation.

### Quality
Prioritizes correctness and reliable output over raw speed.

### Custom
Allows the user to define their own optimization weights. Validated to total 100%.

Weights and thresholds per profile are documented in `docs/OPTIMIZATION_METHOD.md`. For example, Speed uses `0.95` quality threshold while Quality uses `0.99`.

## Hardware support

Supported detection paths include:

* NVIDIA CUDA / nvidia-smi
* AMD ROCm
* Apple Metal
* CPU/RAM
* Vulkan where available

> The optimizer is designed to adapt to the detected hardware rather than relying on a specific GPU or VRAM capacity.

> Hardware detection and normalization logic are covered by automated tests across multiple simulated hardware configurations (4/6/12/24/48+GB, CPU-only, zero-range, etc.). Not every physical GPU has been individually tested — this is a beta.

## LM Studio configuration

Default:

```env
LM_STUDIO_URL=http://127.0.0.1:1234
```

This is the normal local LM Studio Developer API address. Change it in `.env`, the Web UI Settings, or via CLI override:

```bash
lm-optimizer status --url http://192.168.1.50:1234
```

`.env` example (`copy .env.example .env`):

```env
LM_STUDIO_URL=http://127.0.0.1:1234
WEB_HOST=127.0.0.1
WEB_PORT=8080
# DATABASE_PATH=./data/optimizer.db
```

For remote LM Studio, ensure network access is allowed (firewall/VPN) and **do not expose an unauthenticated LM Studio control API to the public internet** without understanding the security implications. The optimizer never stores credentials; only the URL is stored.

## Installation

```bash
git clone <repository>
cd lm-studio-auto-optimizer

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -e .
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Then:

```bash
lm-optimizer status
```

If this shows `⚠ LM Studio is not available — Open Settings to configure`, configure `LM_STUDIO_URL` and ensure LM Studio Developer API is enabled.

## First run

```
1. Start LM Studio
2. Enable Developer API (LM Studio → Settings → Developer)
3. Start the LM Studio server
4. Configure LM_STUDIO_URL (see above)
5. Open the optimizer
6. Select exactly one model
7. Select a profile
8. Review the search space
9. Run baseline
10. Start optimization
11. Review results
12. Apply a configuration only after reviewing it
```

Web UI:

```bash
python -m lm_optimizer.web_main
```

Default:

```
http://127.0.0.1:8080
```

The Web UI requires explicit model selection — the Start button remains disabled until one model is chosen. The pre-run review shows search space, estimated configs, and warns for large runs.

## Screenshots

> Screenshots will be added after physical LM Studio validation.

Placeholders:

* Dashboard
* Model selection
* Optimization review
* Live optimization
* Results
* Presets/history

## Results explanation

The optimizer compares:

* baseline
* candidate configurations
* validated configurations
* Pareto-optimal configurations

Example metrics:

```
Generation speed
Prompt processing speed
Estimated TTFT
Context length
VRAM/RAM usage
Correctness checks
Stability
Overall profile score
```

**TTFT is labeled `Estimated TTFT`** because the current implementation does not use streaming token timestamps; it estimates `total_time * 0.1` as documented in `docs/OPTIMIZATION_METHOD.md`.

The results page distinguishes baseline, best speed/balanced/context/quality and Pareto frontier, showing actual measured values only and the weighted `score_breakdown` explaining why a configuration won.

## Methodology

See [`docs/OPTIMIZATION_METHOD.md`](docs/OPTIMIZATION_METHOD.md) for full details:

* dynamically generates a search space from hardware/model/capabilities
* deterministically samples coarse candidates covering low/high per dimension
* refines promising configurations with interaction tests
* evaluates correctness via deterministic heuristics (`29/30 checks passed`)
* calculates profile-specific scores with run-relative/model-relative/hardware-relative normalization
* uses Pareto analysis for trade-offs
* validates finalists with repeated runs
* stores `benchmark_params`, hardware snapshot, and model identity for reproducibility

The method finds **empirically validated** configurations, not a mathematically proven global optimum.

## Known Limitations

### Beta status
This release has not yet completed real-world end-to-end validation on physical LM Studio hardware. Automated tests (61) and static checks pass, but hardware validation is pending.

### Benchmark noise
LLM inference performance can vary because of:

* thermal throttling
* background processes
* GPU state
* memory pressure
* OS scheduling
* LM Studio version
* model implementation
* backend differences

Median over repetitions and stability `1-CV` mitigate but do not eliminate variance.

### Correctness evaluation
Current correctness evaluation is heuristic. It is useful for detecting obvious regressions, malformed output and task failures, but it is not equivalent to human evaluation or a strong LLM judge (planned).

### LM Studio compatibility
Available runtime parameters depend on LM Studio and model/backend capabilities. Unsupported parameters are not optimized (hidden/disabled in UI with explanation).

### RoPE
RoPE modifications are experimental and disabled by default. When enabled, the run is labeled experimental and uses stricter quality validation.

### Remote servers
Network latency and server load can affect benchmark results. Prefer local `127.0.0.1` for most reproducible results.

## Safety

> This software can load and unload models and apply runtime configurations to LM Studio. Review configurations before applying them, especially when using remote LM Studio instances.

Recommend keeping the Web UI bound to:

```
127.0.0.1
```

unless you explicitly understand the networking/security implications (CORS restricted to `127.0.0.1:8080`/`localhost:8080` by default; see `SECURITY.md`).

## Project structure

```
lm_optimizer/
├── api/          FastAPI/WebSocket API
├── benchmark/    Benchmark definitions
├── cli/          CLI
├── database/     SQLite persistence
├── discovery/    Model/capability discovery
├── domain/       Core domain models
├── hardware/     Hardware detection
├── optimizer/    Optimization logic (deprecated legacy, see services/optimizer.py)
├── profiles/     Optimization profiles
├── scoring/      Normalization/correctness
├── services/     Application services (authoritative optimizer is services/optimizer.py)
├── storage/      Checkpoints/presets/results
└── ui/           Web interface
```

`services/optimizer.py` is the authoritative optimizer implementation. `optimizer/engine.py` and `ui/web.py` remain as **deprecated** wrappers for backward compatibility and will be removed in 2.0 — documented in code headers.

Runtime data lives in `data/` (gitignored): `data/optimizer.db`, `data/reports/`, `data/presets/`, `data/logs/`, `data/checkpoints/`. Legacy `config/`, `results/`, `logs/` are auto-migrated if present.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
ruff format .
mypy .  # strict; UI legacy may show untyped warnings, core passes with --ignore-missing-imports
python -m compileall lm_optimizer
```

CI runs without physical GPU requirements (`pytest`, `ruff`, `mypy`, `compileall` on ubuntu/windows/macos matrix, Python 3.11/3.12). GPU integration tests are optional/manual.

## Versioning

`v1.0.0-beta` — Untested Beta. Not stable, not production-ready, not fully hardware-validated. Use with caution.

* `pyproject.toml`: `1.0.0b0` (PEP 440 beta)
* `lm_optimizer/__init__.py`: `1.0.0-beta`
* `CHANGELOG.md`: `1.0.0-beta` entry
* `docs/RELEASE_AUDIT_v1.md`: audit for this beta

## License

MIT — see `LICENSE`.

## GitHub

**Description:** Hardware-aware automatic benchmark and optimization tool for LM Studio. Finds empirically optimized runtime configurations for speed, context, memory, and quality across different hardware.

**Topics:** `lm-studio`, `llm`, `llm-optimization`, `llm-benchmark`, `local-ai`, `local-llm`, `python`, `fastapi`, `gpu`, `nvidia`, `amd`, `apple-silicon`

## Acknowledgments

Built for LM Studio users who want to automate tuning without manual trial-and-error. See `CONTRIBUTING.md` for development guidelines and `SECURITY.md` for threat model.

---

*Is it vibe coded? Yes, yes it is. — built with a lot of iteration, care, and a little help from AI.*
