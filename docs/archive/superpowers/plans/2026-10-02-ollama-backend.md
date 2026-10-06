# Ollama Backend (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ollama as a second backend behind a `BackendClient` seam, with options-based sweep, editable Modelfile `.md` export, and confirmed apply via `/api/create`.

**Architecture:** New code lives in `lm_optimizer/backends/` (split-ready packages). `LMStudioClient` stays where it is (move deferred to the llama.cpp phase to avoid churning 480 tests' imports); it gains interface conformance. Ollama sweep varies per-request `options`; persistence is a confirmed `/api/create` of a new tag.

**Tech Stack:** Python, httpx (mocked in tests), pytest, typer CLI.

**Spec:** `docs/specs/2026-10-02-multi-backend-design.md` (Ollama mapping + split-ready constraint).

## Global Constraints

- TDD: failing test first for every task.
- Full pytest green before every commit; `node --check` if UI JS touched.
- Never commit `results/` or `*.db`.
- No cross-backend imports; backends talk to core only via `BackendClient`.
- Ollama capabilities from the static known-`options` table + live `/api/show`; unknown keys are silently ignored server-side, so there is no 400-probe equivalent — never claim probing that does not exist.
- Apply (`/api/create`) only with explicit user confirm, only to a NEW tag (`<base>:opt` default), never overwrite.
- Fail-closed unload semantics (`keep_alive: 0` verified empty afterwards).

## Review Focus

- Ollama server unreachable mid-sweep; a reasonable person expects recorded failure + continued sweep, never a hang.
- `/api/create` name colliding with an existing model; expected behavior is refuse-before-overwrite, not silent replace.
- User-edited `.md` with invalid PARAMETER lines; expected behavior is a parse error naming the line, not a half-applied model.
- `num_ctx` option forcing a multi-minute reload on big models; expected behavior is the load cost counted in metrics, not hidden.
- Backend switch with a run DB from another backend; expected behavior is runs stay backend-tagged and never mixed in comparisons.

---

### Task 1: BackendClient seam + LM Studio conformance

**Files:**
- Create: `lm_optimizer/backends/__init__.py`
- Create: `lm_optimizer/backends/base.py`
- Modify: `lm_optimizer/services/lm_studio.py` (conformance only, no behavior change)
- Test: `lm_optimizer/tests/test_backend_seam.py` (create)

**Interfaces:**
- Consumes: existing `LMStudioClient` public surface (connect, list_models, get_model, load_model, chat_completion, ensure_unloaded, capabilities, close).
- Produces: `BackendClient` Protocol with `backend_name: str`, async `connect()`, `list_models()`, `load_model(model_id, config)`, `generate(model_id, prompt, options)`, `unload(model_id)`, `close()`; `capabilities` as property-or-method (LM Studio exposes a property — generic call sites must not blindly use parens); `assert_conforms(client) -> None` helper used by all backend tests.

- [ ] **Step 1: Write the failing test**

```python
def test_lm_studio_conforms_to_backend_client():
    assert_conforms(LMStudioClient())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_backend_seam.py -v`
Expected: FAIL with "name not defined".

- [ ] **Step 3: Implement `BackendClient` Protocol + `assert_conforms()` in `lm_optimizer/backends/base.py`, adapt `LMStudioClient` method aliases only if names differ**

No logic changes; aliases/shims only.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest lm_optimizer/tests/test_backend_seam.py lm_optimizer/tests/test_core.py -v`
Expected: PASS, plus full suite still green.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/backends lm_optimizer/services/lm_studio.py lm_optimizer/tests/test_backend_seam.py
git commit -m "feat: add BackendClient seam with LM Studio conformance"
```

### Task 2: OllamaClient (mocked httpx)

**Files:**
- Create: `lm_optimizer/backends/ollama/__init__.py`
- Create: `lm_optimizer/backends/ollama/client.py`
- Test: `lm_optimizer/tests/test_ollama_client.py` (create)

**Interfaces:**
- Consumes: `BackendClient`, `assert_conforms` from Task 1.
- Produces: `OllamaClient(base_url="http://127.0.0.1:11434")` implementing `BackendClient`; `OllamaMetrics` normalized to `{gen_tok_s, prompt_tok_s, load_ms}` computed as `eval_count/eval_duration*1e9` etc.; unload via `keep_alive: 0` + verify.

- [ ] **Step 1: Write the failing test**

```python
async def test_ollama_conforms_and_lists_models(mock_httpx_tags):
    assert_conforms(OllamaClient())
    assert [m.id for m in await OllamaClient().list_models()] == ["qwen3:8b"]

async def test_ollama_metrics_normalized(mock_httpx_generate):
    m = await OllamaClient().generate("qwen3:8b", "hi", {"num_ctx": 4096})
    assert m.gen_tok_s == pytest.approx(61.2, abs=0.1)
```

NOTE: seam methods are async (LM Studio reality); `asyncio_mode = "auto"` is set in pyproject, so plain `async def test_` works.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_ollama_client.py -v`
Expected: FAIL with "module not found".

- [ ] **Step 3: Implement `OllamaClient` in `lm_optimizer/backends/ollama/client.py`**

Endpoints: `GET /api/tags`, `POST /api/show`, `POST /api/generate` (stream=false), unload = generate with empty prompt + `keep_alive: 0` expecting `done_reason: "unload"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_ollama_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/backends/ollama lm_optimizer/tests/test_ollama_client.py
git commit -m "feat: add OllamaClient with metrics normalization"
```

### Task 3: Ollama registry + search space

**Files:**
- Create: `lm_optimizer/backends/ollama/registry.py`
- Modify: `lm_optimizer/services/search_space.py` (backend-aware branch, LM Studio path untouched)
- Test: `lm_optimizer/tests/test_ollama_registry.py` (create)

**Interfaces:**
- Consumes: `OllamaClient.capabilities` (property — access without parens) + `show()` raw details from Task 2.
- Produces: `OLLAMA_OPTIONS = ["num_ctx", "num_batch", "num_thread", "num_gpu", "use_mmap", "temperature", "top_k", "top_p", "min_p", "repeat_penalty"]`; `ollama_space(model_info, hardware) -> dict` with defaults `num_ctx=[2048,4096,8192]`, `num_batch=[64,256,512]`, `num_thread=[logical]`, `num_gpu=[0,max]`; sampling keys pass through to existing generation profiles.

- [ ] **Step 1: Write the failing test**

```python
def test_ollama_space_defaults():
    space = ollama_space(show_fixture("qwen3-8b"), hw_8thread())
    assert space["num_ctx"] == [2048, 4096, 8192]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_ollama_registry.py -v`
Expected: FAIL with "module not found".

- [ ] **Step 3: Implement registry + space branch**

Static table (no probing); `num_ctx` capped by `llama.context_length` from `/api/show`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_ollama_registry.py lm_optimizer/tests/test_stage4_grid.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/backends/ollama/registry.py lm_optimizer/services/search_space.py lm_optimizer/tests/test_ollama_registry.py
git commit -m "feat: add Ollama options registry and search space"
```

### Task 4: Modelfile export + apply

**Files:**
- Create: `lm_optimizer/backends/ollama/modelfile.py`
- Modify: `lm_optimizer/cli/main.py` (add `ollama-export` + `ollama-apply` commands)
- Test: `lm_optimizer/tests/test_ollama_modelfile.py` (create)

**Interfaces:**
- Consumes: winner config dict + `/api/show` base (`FROM` line) from Tasks 2-3.
- Produces: `render_modelfile(base: str, params: dict) -> str` (exact `FROM` + `PARAMETER k v` lines); `parse_modelfile(text: str) -> dict` raising `ValueError("line N: ...")` on invalid lines; `ollama-apply --file <md> --as <tag>` confirming + `POST /api/create`.

- [ ] **Step 1: Write the failing test**

```python
def test_roundtrip_modelfile():
    text = render_modelfile("qwen3:8b", {"num_ctx": 8192, "temperature": 0.7})
    assert parse_modelfile(text) == {"from": "qwen3:8b", "num_ctx": 8192, "temperature": 0.7}

def test_bad_line_names_number():
    with pytest.raises(ValueError, match="line 3"):
        parse_modelfile("FROM qwen3:8b\nPARAMETER nope\nPARAMETER num_ctx abc")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_ollama_modelfile.py -v`
Expected: FAIL with "module not found".

- [ ] **Step 3: Implement render/parse + CLI commands**

Apply refuses when the target tag already exists (list first); `--yes` required for non-interactive use.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_ollama_modelfile.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/backends/ollama/modelfile.py lm_optimizer/cli/main.py lm_optimizer/tests/test_ollama_modelfile.py
git commit -m "feat: add Modelfile export and confirmed apply"
```

### Task 5: Backend switch wiring

**Files:**
- Modify: `lm_optimizer/config.py` (add `backend: Literal["lm-studio","ollama","llama-cpp"] = "lm-studio"`, `ollama_base_url`, `llamacpp_base_url`)
- Modify: `lm_optimizer/cli/main.py` (`--backend` global option + client factory)
- Test: extend `lm_optimizer/tests/test_backend_seam.py`

**Interfaces:**
- Consumes: `OllamaClient` from Task 2.
- Produces: `get_backend_client(backend: str, base_url: str | None) -> BackendClient`; default `lm-studio` preserves current behavior byte-for-byte.

- [ ] **Step 1: Write the failing test**

```python
def test_backend_factory_defaults_lm_studio():
    assert isinstance(get_backend_client("lm-studio"), LMStudioClient)

def test_backend_factory_ollama():
    assert isinstance(get_backend_client("ollama"), OllamaClient)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest lm_optimizer/tests/test_backend_seam.py -v`
Expected: FAIL with "function not defined".

- [ ] **Step 3: Implement factory + config + `--backend` option**

llama-cpp maps to a `NotImplementedError("llama.cpp backend is phase 2")` stub — explicit, not silent.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest lm_optimizer/tests/test_backend_seam.py -v` + full suite
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lm_optimizer/config.py lm_optimizer/cli/main.py lm_optimizer/tests/test_backend_seam.py
git commit -m "feat: wire backend switch with lm-studio default"
```

### Task 6: Docs + live-verification checklist

**Files:**
- Modify: `docs/OPTIMIZATION_METHOD.md` (Ollama addendum)
- Create: `docs/OLLAMA_LIVE_CHECK.md`
- Test: none (docs only; suite must stay green)

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: documented owner-run checklist (Ollama version, `/api/tags` output, one sweep, one export-edit-apply cycle, capability table sign-off).

- [ ] **Step 1: Write the checklist draft**

Content decisions only: exact commands + expected outputs for a local Ollama instance.

- [ ] **Step 2: Verify suite still green**

Run: `pytest lm_optimizer/tests -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/OPTIMIZATION_METHOD.md docs/OLLAMA_LIVE_CHECK.md
git commit -m "docs: Ollama backend addendum and live checklist"
```
