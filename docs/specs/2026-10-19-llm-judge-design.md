# LLM Judge — Design Specification

**Date:** 2026-10-19
**Status:** Draft for review

---

## 1. Purpose

Provide an AI-powered "second opinion" that evaluates optimization results and recommends:
- **Per-model winner**: Best runtime configuration within a single model's run
- **Overall winner**: Best model+config combination across all tested models for a given use-case
- **Task-specific guidance**: What works best for coding / chat / reasoning / structured output
- **Avoid list**: Models/configs that underperform on this hardware

The judge is **advisory only** — the user always decides.

---

## 2. Scope

### In Scope
- Web UI: "Ask Judge" button on Results page + checkbox selection
- API: `POST /api/judge` (submit), `GET /api/judge/history` (list), `GET /api/judge/{id}` (detail)
- Three backends: LOCAL (LM Studio), API (OpenRouter-compatible), HEURISTIC (rule-based)
- Use-case templates: `coding`, `chat`, `reasoning`, `structured`, `general`, `custom`
- Persisted verdicts in DB (`judge_verdicts` table)
- Prompt strategy: summary table for ALL configs + full detail for top-N (default 3)

### Out of Scope
- Auto-applying judge recommendations
- Fine-tuning judge models
- Multi-turn judge conversations
- Benchmarking judge accuracy
- **Granular test selection for benchmark runs** (separate feature, see §14)

---

## 3. Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Web UI        │     │   API Layer      │     │  ModelJudge     │
│  (Results page) │────▶│  POST /api/judge │────▶│  (services)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                    ┌──────────────────┐                 │
                    │   PromptBuilder  │◀────────────────┘
                    │  (summary + topN)│
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌─────────┐   ┌───────────┐  ┌────────────┐
         │ Local   │   │ API       │  │ Heuristic  │
         │ Judge   │   │ Judge     │  │ Judge      │
         │(LM Std) │   │(OpenRouter)│  │(rules)     │
         └─────────┘   └───────────┘  └────────────┘
```

### Components

| Component | Location | Responsibility |
|---|---|---|
| `JudgeService` | `services/judge_service.py` | Orchestrates: loads data → builds prompt → calls backend → saves verdict |
| `PromptBuilder` | `services/judge_prompt.py` | Builds LLM prompt from run/config data |
| `LocalJudge` | `services/model_judge.py` | Calls LM Studio `/v1/chat/completions` with loaded model |
| `APIJudge` | `services/model_judge.py` | Calls OpenRouter/OpenAI-compatible endpoint |
| `HeuristicJudge` | `services/model_judge.py` | Rule-based fallback (no AI) |
| `JudgeVerdict` table | `database/manager.py` | Persists verdicts with metadata |

---

## 4. Data Flow

### 4.1 Input: `JudgeRequest`

```json
{
  "run_ids": ["uuid-1", "uuid-2"],           // for cross-model; optional if model_ids given
  "model_ids": ["qwen3.8-9b-distill", ...], // alternative: list of models (latest run each)
  "config_ids": ["uuid-a", "uuid-b"],       // optional: specific configs (checkboxes)
  "use_case": "coding",                     // coding | chat | reasoning | structured | general | custom
  "custom_prompt": "...",                   // if use_case == "custom"
  "priority": "balanced",                   // speed | quality | balanced | context
  "backend": "local",                       // local | api | heuristic
  "judge_model": "qwen3.8-27b-turbo...",    // for LOCAL: model key; for API: model name
  "api_key": "...",                         // for API backend
  "api_base": "https://openrouter.ai/api/v1", // for API backend
  "top_n_detail": 3,                        // how many configs get full prompt/output detail
  "include_reasoning_matrix": true,         // include reasoning=off/low/medium/high/xhigh comparison
  "fastest_context_preset": false           // filter to fastest config at max context per model
}
```

### 4.2 Data Loading (read-only)

For each `run_id`:
- Run metadata: model, status, quality_threshold, started_at
- All configurations with `status IN ('passed', 'quality_failed', 'load_failed')`
- For each config: `config_json`, `metrics_json`, `quality_json`, `stability_score`, `peak_vram_gb`, `score`
- From `metrics_json`: `prompt_text`, `thinking_text`, `output_text` (if present)

### 4.3 Prompt Construction

**Structure:**
```
SYSTEM: You are an expert LLM optimization analyst. Evaluate configurations for {use_case} on {hardware}.

HARDWARE: RTX 3060 Laptop 6GB VRAM, 32GB RAM

SUMMARY TABLE (ALL configs):
| Model | Config | ctx | tok/s | prompt tok/s | TTFT | Quality | VRAM | Stability | Key Tests |
|-------|--------|-----|-------|--------------|------|---------|------|-----------|-----------|
| ...   | ...    | ... | ...   | ...          | ...  | ...     | ...  | ...       | ...       |

TOP {top_n} CONFIGS — FULL DETAIL:
### Config {id} ({model})
- Runtime: gpu={gpu_ratio}, batch={batch}, flash={flash}, kv={kv}, parallel={parallel}, checkpoints={ckpt}
- Metrics (avg over {n} tests):
    gen_tok_s: {x}, prompt_tok_s: {y}, ttft_ms: {z}
- Quality: overall={q}, checks={passed}/{total}
    coding_task: {c}, structured_output: {s}, medium_reasoning: {r}, long_context: {l}
- Stability: {stab}, Peak VRAM: {vram} GB
- Sample Prompt: {prompt_text[:500]}
- Sample Output: {output_text[:500]}

REASONING SETTINGS MATRIX (if available in run):
For each config tested with reasoning={off, low, medium, high, xhigh}:
| Config | reasoning | tok/s | Quality | coding_task | structured_output | medium_reasoning |
|--------|-----------|-------|---------|-------------|-------------------|------------------|
| ...    | off       | ...   | ...     | ...         | ...               | ...              |
| ...    | low       | ...   | ...     | ...         | ...               | ...              |
| ...    | medium    | ...   | ...     | ...         | ...               | ...              |
| ...    | high      | ...   | ...     | ...         | ...               | ...              |
| ...    | xhigh     | ...   | ...     | ...         | ...               | ...              |

TASK: Return JSON verdict (schema below).
```

### 4.4 Output: `JudgeVerdict`

```json
{
  "per_model_winner": {
    "model_id": "qwen3.8-9b-distill",
    "config_id": "abc-123",
    "reasoning": "Best quality/speed tradeoff for coding at 12.7 tok/s, 0.983 quality"
  },
  "overall_winner": {
    "model_id": "mirothinker-v1.5-30b-q2ks",
    "config_id": "def-456",
    "reasoning": "Highest tok/s (20.5) with quality 0.978; Q2_K_S quantization wins on 6GB"
  },
  "task_breakdown": {
    "coding": {"winner": "mirothinker...", "runner_up": "qwen3.8-9b...", "reasoning": "..."},
    "chat": {"winner": "qwen3.8-9b...", "runner_up": "gemma-4-26b...", "reasoning": "..."},
    "reasoning": {"winner": "qwen3.6-35b...", "runner_up": "mirothinker...", "reasoning": "..."},
    "structured": {"winner": "qwen3.8-9b...", "runner_up": "gpt-oss-20b...", "reasoning": "..."}
  },
  "avoid": [
    {"model_id": "qwen/qwen3.8-27b", "reason": "MTP load succeeds but 1.1 tok/s; IQ4_XS too slow"},
    {"model_id": "mistralai/ministral-3-14b", "reason": "Quality 0.77-0.82 at ctx=32768; needs lower ctx"}
  ],
  "warnings": [
    "Models >15GB at ctx=32768 risk BSOD on this GPU (TDR)",
    "gpt-oss-20b MXFP4 quantization slower than Q2_K_S 30B"
  ],
  "confidence": 0.87,
  "metadata": {
    "backend": "local",
    "judge_model": "qwen3.8-27b-turbo-fable...",
    "use_case": "coding",
    "priority": "balanced",
    "run_ids": ["..."],
    "configs_evaluated": 42,
    "prompt_tokens": 18432,
    "completion_tokens": 1024,
    "created_at": "2026-10-19T20:30:00Z"
  }
}
```

---

## 5. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/judge` | Submit judge request, returns `judge_id` (async) or verdict (sync) |
| `GET` | `/api/judge/{id}` | Get verdict by ID |
| `GET` | `/api/judge/history` | List recent verdicts (limit=20) |
| `GET` | `/api/judge/config/{config_id}` | Get verdicts referencing a specific config |

### Request: `POST /api/judge`

Body: `JudgeRequest` (see 4.1)

Response (async):
```json
{ "judge_id": "uuid", "status": "pending", "poll_url": "/api/judge/{id}" }
```

Response (sync, if `wait=true`):
```json
{ "judge_id": "uuid", "status": "completed", "verdict": {...} }
```

---

## 6. Database Schema

```sql
CREATE TABLE judge_verdicts (
    id TEXT PRIMARY KEY,
    run_ids_json TEXT NOT NULL,           -- JSON array of run IDs
    config_ids_json TEXT,                 -- JSON array of config IDs (if filtered)
    use_case TEXT NOT NULL,
    custom_prompt TEXT,
    priority TEXT NOT NULL,
    backend TEXT NOT NULL,                -- local | api | heuristic
    judge_model TEXT NOT NULL,
    verdict_json TEXT NOT NULL,           -- full JudgeVerdict JSON
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'completed', -- pending | completed | failed
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_judge_verdicts_created ON judge_verdicts(created_at DESC);
CREATE INDEX idx_judge_verdicts_run ON judge_verdicts(run_ids_json); -- for LIKE queries
```

---

## 7. UI Design

### 6.1 Results Page Additions

**Location:** `lm_optimizer/ui/static/js/results.js` + template

**Elements:**
- **Header bar** (above configurations table):
  - Checkbox column in config table (header: select all visible)
  - Button: `🤖 Ask Judge` (disabled until ≥1 checkbox checked OR no selection = all)
  - Button: `📜 Judge History` (opens modal with past verdicts)

- **Judge Modal** (opens on "Ask Judge"):
  - Use-case dropdown: `coding` `chat` `reasoning` `structured` `general` `custom`
  - Custom prompt textarea (shown if `custom` selected)
  - Priority radio: `speed` `quality` `balanced` `context`
  - Backend radio: `Local (LM Studio)` `API (OpenRouter)` `Heuristic (fast)`
  - Judge model select (populated from `/api/models` for LOCAL; text input for API)
  - API key / base URL inputs (shown if API selected)
  - Top-N detail slider: 1–10 (default 3)
  - **Checkbox: "Include reasoning matrix" (default on)** — adds off/low/medium/high/xhigh comparison table
  - **Checkbox: "Fastest @ max context preset"** — filters to best config at highest context per model
  - `Run Judge` button → shows loading spinner → renders verdict

- **Verdict Modal** (read-only display):
  - Tabs: `Overall` | `Per-Model` | `By Task` | `Avoid` | `Raw JSON`
  - Confidence badge (green/yellow/red)
  - "Save / Dismiss" buttons

### 6.2 Aesthetic Direction

Per existing UI: **Technical/Analytical** — clean tables, monospace metrics, high information density. Matches `results.js` style.

### 6.3 State Coverage

| State | Handling |
|---|---|
| No runs in DB | Button disabled, tooltip "Run optimizations first" |
| Judge in progress | Spinner, "Evaluating..." toast, disable button |
| Backend error | Toast with error, fallback to heuristic offer |
| Empty verdict | "No clear winner" with reasoning |
| Long prompt (>context) | Auto-reduce `top_n_detail`, show warning |

---

## 8. Backend Implementations

### 8.1 LocalJudge (LM Studio)

- Calls `POST http://127.0.0.1:1234/v1/chat/completions`
- Uses currently loaded model (or `judge_model` if specified and loaded)
- Temperature 0.1, max_tokens 2048
- JSON mode enforced via `response_format: {type: "json_object"}`

### 8.2 APIJudge (OpenRouter-compatible)

- Configurable `api_base` (default: `https://openrouter.ai/api/v1`)
- `Authorization: Bearer {api_key}`
- Same payload as LocalJudge
- Supports any OpenRouter model (e.g., `anthropic/claude-3.5-sonnet`, `google/gemini-2.0-flash`)

### 8.3 HeuristicJudge (rule-based, no AI)

Scoring formula per config:
```
score = (norm_tok_s * w_speed) + (norm_quality * w_quality) + (norm_stability * w_stability)
```
Weights by priority:
- `speed`: 0.6 / 0.2 / 0.2
- `quality`: 0.2 / 0.6 / 0.2
- `balanced`: 0.33 / 0.34 / 0.33
- `context`: 0.2 / 0.3 / 0.2 + bonus for higher ctx

Returns same `JudgeVerdict` schema with `confidence: 0.6` and `reasoning: "Heuristic scoring: ..."`.

---

## 9. Error Handling

| Failure | Handling |
|---|---|
| LM Studio not running | Return error, suggest starting LM Studio or switching backend |
| Model not loaded (LOCAL) | Return error, suggest loading model or using API |
| API timeout (30s) | Retry once, then fallback to heuristic |
| JSON parse failure | Retry with stricter prompt, then fallback |
| Prompt too long | Auto-reduce `top_n_detail`, rebuild, retry |
| DB write failure | Log error, return verdict anyway (in-memory) |

---

## 10. Testing

### Unit Tests (`tests/test_judge_*.py`)

- `PromptBuilder`: builds valid JSON, respects `top_n_detail`, truncates long outputs
- `HeuristicJudge`: produces valid verdict, weights match priority
- `LocalJudge`/`APIJudge`: mock HTTP, verify payload shape
- `JudgeService`: integrates data loading → prompt → backend → DB save

### Integration Test

- Spin up test DB with 2 runs × 5 configs each
- Call `POST /api/judge` with HEURISTIC backend
- Verify verdict structure + DB row created

### Manual Test Checklist

- [ ] "Ask Judge" with no selection → evaluates all configs in run
- [ ] Checkbox selection → evaluates only checked configs
- [ ] Cross-model (multiple run_ids) → overall winner + per-model
- [ ] LOCAL backend with loaded model → works
- [ ] API backend with OpenRouter key → works
- [ ] HEURISTIC → instant, no network
- [ ] History modal shows past verdicts
- [ ] Long outputs truncated in prompt (not crashing)

---

## 11. Implementation Plan (Phased)

### Phase 1: Core Service (1-2 days)
- [ ] `services/judge_prompt.py` — PromptBuilder class (with reasoning matrix support)
- [ ] Extend `model_judge.py`: `LocalJudge`, `APIJudge` implementations
- [ ] `services/judge_service.py` — orchestration + DB persistence
- [ ] DB migration: add `judge_verdicts` table
- [ ] Unit tests for PromptBuilder + HeuristicJudge

### Phase 1b: Reasoning Matrix Data Loader (0.5 day)
- [ ] Query logic: group configs by base_runtime + reasoning setting
- [ ] Detect runs that have reasoning sweep (multiple configs differing only in `generation.reasoning`)
- [ ] Build matrix table for prompt (§4.3)

### Phase 2: API Endpoints (0.5 day)
- [ ] `POST /api/judge`, `GET /api/judge/{id}`, `GET /api/judge/history`
- [ ] Request validation (Pydantic models)
- [ ] Async execution (background task) + sync `wait=true` option

### Phase 3: UI Integration (1 day)
- [ ] Results page: checkbox column + "Ask Judge" button
- [ ] Judge Modal with all settings
- [ ] Verdict Modal with tabs
- [ ] History Modal
- [ ] Toast notifications for async flow

### Phase 4: Polish (0.5 day)
- [ ] Prompt token estimation + auto-reduce
- [ ] Better error messages per backend
- [ ] Documentation in README / UI help tooltip

---

## 12. Open Questions for Review

1. **Async vs Sync**: Default async with polling, or sync with timeout? (Current: async + `wait=true` option)
2. **Prompt token budget**: Hard limit? (Current: auto-reduce `top_n_detail` until fits 8k)
3. **Judge model for LOCAL**: Use currently loaded model, or allow loading a specific judge model? (Current: allow specifying, warn if not loaded)
4. **Cross-model without run_ids**: Accept `model_ids` and auto-pick latest successful run each? (Current: yes)
5. **Verdict versioning**: Store prompt + response for reproducibility? (Current: yes, in `verdict_json` metadata)

---

## 14. Granular Test Selection (UI + CLI)

**Separate but related feature**: Allow users to run specific subsets of benchmark tests instead of the full suite.

### 14.1 Test Categories

| Category | Tests | Purpose |
|---|---|---|
| `speed` | `short_instruction` | Quick tok/s measurement |
| `context` | `long_context` | Context window handling |
| `reasoning` | `medium_reasoning` | Step-by-step reasoning |
| `coding` | `coding_task` | Code generation |
| `format` | `structured_output` | JSON/structured output |
| `all` | All 5 | Full quality suite |

### 14.2 CLI Options

All granular modes follow the `*-only` pattern; each composes the required
`--skip-phases` set automatically (speed-only skips quality/recovery/
validation/context, quality-only and reasoning-only skip speed/.../context,
MTP-only keeps the speed phase but filters its plan to the speculative
probes + S0 budget anchor). `resume` keeps the composed skips.

```bash
# Run only the speed probes at a pinned context (required)
lm_optimizer optimize <model> --speed-only --context 8192

# Run the quality battery on the default (baseline) runtime config
lm_optimizer optimize <model> --quality-only

# Run only the speculative decoding / MTP probes (requires --context)
lm_optimizer optimize <model> --mtp-only --context 8192

# Run only the reasoning tests, each at off/low/medium/high/xhigh
lm_optimizer optimize <model> --reasoning-only --max-context 16384
```

Notes:
- `--mtp-test` and `--reasoning-sweep` remain accepted aliases of
  `--mtp-only` / `--reasoning-only` (added 2026-10-20 for the consistent
  `*-only` naming).
- The modes are mutually exclusive; `reasoning_sweep=True` makes the case
  battery reasoning-only (5 level variants per reasoning test) in
  `benchmark/suite.py`.
- Per-model family batch execution: `python tools/granular_suite.py`
  (sequential queue, per-step timeout + stall watchdog).

### 14.3 UI Additions

**Optimize Page** (new modal or expand existing):
- Test selection: checkboxes for each category (`speed`, `context`, `reasoning`, `coding`, `format`)
- Context size input for speed test: `--context 8192`
- Reasoning sweep toggle: runs all 5 reasoning levels
- MTP test toggle: enables speculative decoding test
- "Custom test" text input: comma-separated test names

**Results Page** (extend Judge modal):
- "Re-run specific test" button per config → opens modal with test selection
- Quick actions: "Re-run coding_task only", "Re-run at ctx=16384"

### 14.5 Implementation Notes

- Extend `create_benchmark_cases()` in `benchmark/suite.py` to accept `include_categories` filter
- Add `--test` / `--test-only` CLI args to `optimize` command
- Add test selection state to `OptimizationRun` (store which tests were run)
- Quality threshold still applies to selected tests only
- MTP test requires separate speculative decoding path (see `speculative.py`)
- Reasoning sweep: create 5 configs per base runtime with `generation.reasoning` = off/low/medium/high/xhigh

---

## 15. Dependencies

- No new Python dependencies (uses existing `httpx`, `sqlite3`, `json`)
- Frontend: vanilla JS, no new libraries
- LM Studio must be running for LOCAL backend
- OpenRouter API key for API backend (user-provided)

---

*End of specification. Please review and approve before implementation planning.*