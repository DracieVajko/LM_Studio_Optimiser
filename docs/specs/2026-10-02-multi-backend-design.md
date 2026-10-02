# Multi-backend optimizers (Ollama + llama.cpp) — Design

Date: 2026-10-02. Status: approved by owner (Ollama first, adapters in 1 repo).

## Problem

LM Studio Optimiser tunes runtime configs only for LM Studio. The owner also
runs Ollama and llama-server and wants the same workflow (sweep → winner →
editable profile → apply) for both, with one shared UI/scoring/DB instead of
three copied projects.

## Goals

- One repo, backend switch (lm-studio | ollama | llama-cpp).
- Shared: benchmark suite, scoring, DB, web UI, sandbox, reporting.
- Per backend: client, capability discovery, parameter registry, profile exporter.
- Ollama: sweep via per-request `options`; winner rendered as editable
  `<model>-best.modelfile.md`; apply stage persists via `POST /api/create`.
- llama.cpp: sampling sweep via `/v1/chat/completions`; load params exported
  as server command / preset INI / `LLAMA_ARG_*` env (restart-applied).

## Non-goals

- No model downloading by the optimizer. No silent overwrite of existing
  Ollama models (new tagged model, e.g. `qwen3:opt`, only on confirm).
- No undocumented REST keys, no `llama.cpp overrides`-style hacks anywhere.
- No per-request load-param sweep on llama.cpp (API takes only `{"model"}`).

## Constraints

- TDD, 445+ pytest green, `node --check` on UI changes, never commit
  `results/` or `*.db`, fail-closed unload semantics per backend.
- Backend capabilities verified live against the actual server version;
  anything unverified is manual-only guidance, never claimed as tuned.
- `MAX_PLAN_PROBES = 25` discipline carries over to new backends.

## Approach

Introduce a `BackendClient` seam (list/discover, load, generate/chat,
unload, capabilities, metrics normalization); re-express `LMStudioClient` as
the first implementation without behavior change; add `OllamaClient`, then
`LlamaCppClient`; add per-backend registries reusing the registry schema;
add exporters (Modelfile `.md`, server-cmd/INI). Ollama phase first.

## Ollama mapping (verified vs docs)

- Discovery: `GET /api/tags`, details: `POST /api/show` (incl.
  `llama.context_length`, `parameter_size`).
- Sweep: `POST /api/chat|/api/generate` with `options` (`num_ctx`,
  `num_batch`, `num_thread`, `num_gpu`, `use_mmap`, sampling...).
- Metrics: `eval_count/eval_duration` (gen tok/s), `prompt_eval_*`,
  `load_duration`.
- Unload: `keep_alive: 0`. Persist: `POST /api/create {from, parameters}`.

## llama.cpp mapping (verified vs docs, live check pending)

- `GET /v1/models`, chat: `POST /v1/chat/completions`, `GET /props`
  (`POST /props` only with `--props`), `/slots`, `/metrics`.
- `POST /models/load|unload` take only `{"model"}` (router mode).
- Sweepable live: sampling params. Load params (`n_ctx`, `n_batch`,
  `n_ubatch`, `n_parallel`, `n_gpu_layers`, `cache_type_k/v`, `flash_attn`,
  `kv_unified`, `ctx_checkpoints`, RoPE/YaRN, threads): export only.
- Open: which server mode the owner runs (single vs `--models-dir`
  router) and its version — verify live before claiming channels.

## Open questions

- Ollama apply naming/tag convention (`<base>:opt` default?).
- Whether `num_ctx` changes force full reload cost accounting in scoring.
- llama.cpp preset INI vs full command export preference.
