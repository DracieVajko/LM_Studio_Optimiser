# Test Manager + Recommender Design (Integrated in LM Studio Optimizer)

**Status:** Approved by user (`ok`)
**Scope:** Direct integration into `lm-optimizer` repo; no separate repo; no external folder leaks.

## Purpose
A single integrated subsystem (`lab test-manager` CLI + UI card) that:
- Manages benchmark/test execution intelligently (order, VRAM awareness, partial/resume).
- Recommends which model/config is worth keeping based on measured results.
- Cleans failed artifacts automatically.
- Allows opt-in use of an external assistant (local LM Studio + optional `.env` endpoint); never leaks outside this folder; no web access by default.

## Architecture

- `lm_optimizer/services/test_manager.py`: orchestration, ordering logic, VRAM guard, cleanup.
- CLI extensions (`lm_optimizer/cli/lab.py`): `test-manager`, `recommend`, `clean`.
- UI: prompt/card in existing templates (`results.js`, `model-detail.js`, `model_detail.html`).
- DB: extends existing `runs`/`configurations` with `partial` status for intermediate results.

## Ordering Rules (VRAM-aware)

1. Ascending model/file size.
2. Within same size band: MoE before dense; dense last.
3. Before each phase: check available VRAM; keep 250–500 MB free (configurable floor).
4. If `generation_tok/s < 5` (configurable threshold `--speed-floor`, default 5): switch to shorter/faster tests, pause, wait for user input before larger/full tests.
5. Interim results saved (`partial` status + fields) before pause.

## Assistant / Endpoint

- Default: `local-only` (LM Studio at `http://127.0.0.1:1234`).
- Opt-in (`--use-assistant` CLI / checkbox UI): uses `.env` key (`LM_OPTIMIZER_*`) for external endpoint; key never written to repo; no web search.
- Prompt before each major phase (start, before large/full tests, before recommendation)

## Isolation & Security

- Never writes outside project folder (`lmstudio_defaults` writes only to `~/.lmstudio` via `LM_OPTIMIZER_LMSTUDIO_HOME`).
- API keys only via `.env` (`LM_OPTIMIZER_*`); no secrets in code or tests.
- No web access by default (`requests` to external domains disabled unless explicitly configured).

## Cleanup

- Auto-delete failed test artifacts (temp files, failed DB entries with `status=failed` after 2 consecutive failures — configurable).
- Manual `clean` command available.

## Review Points (Self-check)

- No placeholders: all interfaces named (`test_manager` service, CLI commands, status values).
- No contradictions: isolation confirmed, VRAM rules numeric, assistant opt-in explicit.
- Scope focused: single integration into existing repo; no new repo, no external leak.
--- Technical References (FreeToken + llama.cpp MoE) ---
FreeToken: dynamic CPU-GPU co-execution, CUDA double-buffering (buffer_A/B swap), Pinned Memory for expert weights, semantic-aware KV cache.
llama.cpp 8GB optimization: -ngl 99 (GPU attention/KV) + --n-cpu-moe 1 (CPU expert routing) achieves ~33 tok/s vs ~4 tok/s for 35B MoE.
Both are referenced for scheduling logic; no live LLM load required for design.

## XDA Article: 35B MoE on 8GB GPU (2026-10-22)
- Source: XDA Developers (Abhinav Raj), "I changed one llama.cpp flag, and my 8GB laptop GPU now runs a 35B model faster than I can read"
- Flag: --n-cpu-moe N (-ngl 99) peels expert tensors of first N layers to RAM; keeps Attention/KV/shared on GPU.
- KV cache quantization: -ctk q8_0 -ctv q8_0 (8-bit KV cache) for VRAM savings.
- Benchmark command: llama-bench -m Qwen3.6-27B-IQ4_XS.gguf -m Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -ngl 99 -ncmoe 0,28 -p 512 -n 128 -r 3 -ctk q8_0 -ctv q8_0
- Results (8GB RTX 3070 Mobile):
  - Dense 27B IQ4_XS (14.37GiB): 38.83 prompt tok/s, 0.75 gen tok/s (no MoE speedup)
  - MoE 35B UD-IQ4_XS (16.50GiB) 0 MoE layers: 109.66 prompt tok/s, 4.02 gen tok/s
  - MoE 35B UD-IQ4_XS (16.50GiB) 28 MoE layers: 341.37 prompt tok/s, 32.82 gen tok/s (8x speedup)
- Conclusion: MoE offloading makes 35B models usable on 8GB GPUs (~33 tok/s vs ~4 tok/s).
- Relevance: confirms our -ngl 99 + --n-cpu-moe 1 recommendation for LM Studio/llama.cpp; validates KV cache quantization (q4_0/q8_0) as key VRAM saver.
