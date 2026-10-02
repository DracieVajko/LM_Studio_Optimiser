# Ollama Backend — Live Verification Checklist (owner-run)

Phase-1 Ollama support is implemented and unit-tested (mocked httpx), but no
live Ollama server has been touched by CI. Work through this checklist on a
local machine with Ollama installed before trusting an Ollama sweep. Check
each box only with observed output in hand.

> Scope reminder: Phase 1 delivers the `BackendClient` seam, `OllamaClient`,
> the options registry + search-space branch, Modelfile export/apply, and the
> `--backend` switch. The sweep loop is not yet driven by
> `SearchSpaceGenerator.generate_ollama()` (no production caller), so this
> checklist verifies connectivity, measurement, and the export-edit-apply
> cycle — not a full optimizer run.

## 0. Prerequisites

- [ ] Ollama installed and serving: `ollama --version` prints a version;
      record it here: _______________
- [ ] At least one model pulled, e.g. `ollama pull qwen3:8b`
- [ ] Optimizer checkout on branch `feat/ollama-backend`, suite green
      (`pytest lm_optimizer/tests -q` → 513 passed)

## 1. Discovery — `/api/tags` output

- [ ] `curl -s http://127.0.0.1:11434/api/tags` returns `{"models": [...]}`;
      paste the model `name` list here: _______________
- [ ] `curl -s http://127.0.0.1:11434/api/version` returns a version matching
      `ollama --version` above (or note the mismatch).

## 2. Client smoke — list + show + generate

Run against the live server (Python REPL or a scratch script, never a test):

- [ ] `OllamaClient().list_models()` returns the same names as the
      `/api/tags` output in step 1.
- [ ] `await client.show("<tag>")` returns `modelfile` + `model_info` keys;
      record `llama.context_length` here: _______________
- [ ] `await client.generate("<tag>", "Say hi in five words.", {})` returns
      text plus `gen_tok_s > 0`; record gen_tok_s: _______________
      and `load_ms` (first call includes the load cost — expected, not hidden).

## 3. Sweep-shaped check — per-request options

- [ ] Same prompt with `{"num_ctx": 2048}` vs `{"num_ctx": 4096}` both
      succeed; the 4096 call shows a larger `load_ms` (reload cost counted
      in metrics, per `OPTIMIZATION_METHOD.md` §14.2).
- [ ] An unknown option (e.g. `{"definitely_not_real": 1}`) is silently
      ignored by the server — generation still succeeds. (Confirms the
      static-table design: there is no probe channel.)

## 4. Export-edit-apply cycle

- [ ] `python -m lm_optimizer.cli.main ollama-export --model <tag> --param num_ctx=4096 -o /tmp/check.md`
      prints exactly `FROM <tag>` + `PARAMETER num_ctx 4096` lines.
- [ ] Edit `/tmp/check.md`: change `4096` → `8192`, add a bad line
      `PARAMETER num_ctx abc` on line 3. Run apply **expecting refusal**:
      `... ollama-apply --file /tmp/check.md --as <tag>:opt --yes`
      must fail with `Invalid Modelfile: line 3: ...` and create nothing
      (`ollama list` unchanged).
- [ ] Fix the file (remove the bad line). Run apply for real; confirm the
      interactive prompt names the new tag, or use `--yes`. New tag
      `<tag>:opt` (default `<name>:opt`) appears in `ollama list`.
- [ ] Re-run apply with the same `--as` tag: must refuse with
      `refusing: target tag ... already exists` (no overwrite).
- [ ] Clean up the check tag: `ollama rm <tag>:opt`.

## 5. Unload verification

- [ ] After generating with the check tag, `await client.unload("<tag>")`
      returns `True` and `GET /api/ps` no longer lists the model
      (fail-closed: `False` if still loaded).

## 6. Capability table sign-off

- [ ] `OLLAMA_OPTIONS` in `lm_optimizer/backends/ollama/registry.py` matches
      the live server's documented `options` keys for your Ollama version
      (10 keys: `num_ctx`, `num_batch`, `num_thread`, `num_gpu`, `use_mmap`,
      `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`).
      Note any drift here: _______________
- [ ] `num_ctx` default `[2048, 4096, 8192]` vs the `llama.context_length`
      recorded in step 2: values above the cap are dropped by
      `ollama_space` (filter-only — confirm the surviving list by calling
      it with the live `show()` output).

## 7. Backend-switch hygiene

- [ ] Default CLI behavior unchanged: commands without `--backend` use
      LM Studio exactly as before (byte-identical path).
- [ ] `--backend ollama` selects `OllamaClient`; `--backend llama-cpp`
      fails explicitly with `llama.cpp backend is phase 2` (not silently).
- [ ] Do not mix runs across backends in one comparison:
      `OptimizationRun` has no backend field yet (only capability snapshots
      are backend-tagged). Keep Ollama and LM Studio runs separate until
      per-run tagging lands.

## Sign-off

- Owner: _______________  Date: _______________
- Ollama version (from step 0): _______________
- Suite still green after any live-driven fixes (`pytest lm_optimizer/tests -q`): _______________
