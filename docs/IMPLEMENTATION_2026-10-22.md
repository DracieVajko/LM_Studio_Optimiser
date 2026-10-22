# IMPLEMENTAČNÝ SUMÁR — 2026-10-22 (večerná vlna)

## Čo bolo doplnené v tejto vlne (P1–P6 gap closure)

| Gap z predchádzajúceho sumáru | Riešenie | Testy |
|---|---|---|
| P1: rotácia finalistov (ABC/BCA/CAB) | `AdaptiveOptimizer.rotate_order` + deterministický per-run offset v `_phase_quality` | 7 |
| P2: 1.5× bracketing v `ctx` command | `--factor`, `--round-to` + `context_sweep.geometric_contexts(factor=, round_to=)` | (existujúce) |
| P2: fill ratios + retrieval v CLI | `--retrieval/--no-retrieval`, `--detailed-fill`; retrieval validácia na najväčšom stable rungu | (existujúce) |
| P3: Quality Lab wiring do optimizer | `--quality-lab` → `_run_quality_lab` na každom PASSED finalistovi; hard gate veto | (existujúce + 2) |
| P4: browser validácia | `web_validator.py` (stdlib `html.parser` + `node --check`), napojený do `sandbox_runner` | 13 |
| P5: community DB import/export | `community_import.py` (strict validácia, dedup) + CLI `lab community-import/export` | 13 |
| P6: Experiment Builder | `/experiments` stránka + `experiments.js` (mode → model → budget → review → run) | 8 |
| P6: Model Detail page | `/models/<id>` + `model-detail.js` (identity, capabilities, presets, history, configs) | (v dashboard testoch) |
| P6: granular flagy cez web API | `compose_granular_skips` (shared CLI/web) + schéma polia | 6 |
| P6: Full Optimization orchestration | `lab full-optimize` (runtime → context → quality) | — |
| Cross-platform metrics | `metrics_providers.py` (Windows/Linux/macOS) s honest "unavailable" | 9 |
| Thermal/power monitoring | `_collect_thermal()` v benchmarku + polia v `BenchmarkMetrics` | (v metrics testoch) |

## Testy

```
1094 passed, 0 failed (predtým 1043)
```

Nové testovacie súbory:
- `tests/test_finalist_rotation.py` (7)
- `tests/test_web_validator.py` (13)
- `tests/test_community_import.py` (13)
- `tests/test_metrics_providers.py` (9)
- `tests/test_experiment_builder.py` (8)

## Čo ešte zostáva (honest list)

### Mimo rozsahu — rozhodnutie používateľa (2026-10-22)

| Položka | Dôvod |
|---|---|
| **Playwright / browser rendering** | sandbox zapisuje artefakty do `results/sandbox/<task>/<timestamp>/` — používateľ si výsledok pozrie sám; automatizácia prehliadača by pridala ťažkú závislosť za úsudok, ktorý človek spraví lepšie |
| **Blender render validácia** | rovnaký dôvod; Blender navyše nie je nainštalovaný |

### Doplnené v druhej vlne (2026-10-22 večer)

| Položka | Riešenie | Testy |
|---|---|---|
| **SHA256 model hash** | `model_hash.py` (streaming, cache podľa path+size+mtime, fuzzy resolve LM Studio cesty) + CLI `lab hash-model` / `lab hash-all` | 8 |
| **Artifact retention** | `apply_retention()` v `sandbox_runner.py` (keep_last_N, default 10) + timestampované run dirs v `lab sandbox` | 5 |
| **Open-web research sandbox** | `research_web.py` — fetch zdrojov, extrakcia textu, citation precision/recall, penalizácia vymyslených citácií | 11 |
| **HF katalóg discovery** | `hf_catalog.py` — search GGUF repos, parsovanie params/quant, odhad veľkosti, merge s curated katalógom + CLI `lab catalogue-sync` | 19 |

### Už nie je "missing"

## Commity (vetva `feat/single-model-guard`)

1. `06ebe48` — audit fixes, granular modes, cleanup tooling, P0–P6 lab workflows
2. `dabec9c` — handoff note + measured 16K comparison
3. `a7bb44d` — measured swift IQ2/IQ3 variants
4. `df1fc80` — gitignore data/ + targeted cleanup
5. `dbc2f2f` — ignore data/archive/
6. `5d1a34e` — context_explorer 1.5x, lab sandbox CLI, docs map
7. `a211e1c` — MTP comparison runner
8. `d6cada9` — detailed summary vs master spec
9. `f4f8b2a` (posledný) — complete the remaining P1–P6 gaps

## Namerané rýchlosti @16K (z DB, po targeted cleanup)

| Model | Quant | tok/s | N |
|---|---|---|---|
| mirothinker-v1.5-30b | Q2_K_S | 10.08 | 20 |
| qwen3.8-9b-distill | Q4_K_M | 8.99 | 17 |
| qwen/qwen3-coder-30b | Q4_K_M | 8.67 | 20 |
| gpt-oss-20b heretic | Q4_K_S | 7.14 | 16 |
| gemma-4-26b-a4b | Q4_0 | 6.70 | 15 |
| qwen/qwen3.6-35b-a3b | Q4_K_M | 9.43 | 20 (staršia kampaň) |
| bonsai | Q1_0 | 27.2 | 100+ (staršia kampaň) |

**Poznámka:** modely, ktoré si medzitým vymazal (swift, fable, hauhaucs, qwen base,
gpt-oss orig), boli z DB odstránené cez `tools/targeted_cleanup.py`; ich dáta sú
zálohované v `data/archive/cleanup_targeted_*/optimizer.db.snapshot`.

## Ako pokračovať

```bash
# 1. overenie stavu
python -m pytest -q                 # 1094 passed

# 2. nové merania (keď sa stiahnu modely)
python tools/speed16k_newmodels.py  # alebo konkrétny model:
python -m lm_optimizer optimize <model> --speed-only --context 16384 --yes --non-interactive

# 3. finálna tabuľka
python tools/final_report.py --context 16384

# 4. nové lab workflowy
python -m lm_optimizer lab recommend
python -m lm_optimizer lab context-explorer <model> --factor 1.5
python -m lm_optimizer lab quality-lab <model> --seeds 3
python -m lm_optimizer lab duel <a> <b> --type micro_quality
python -m lm_optimizer lab sandbox --task web_dashboard --model <model>
python -m lm_optimizer lab full-optimize <model>
```

## Web UI

| Stránka | Účel |
|---|---|
| `/` | Dashboard (models, presets, hardware) |
| `/history` | História behov |
| `/sandbox` | Prompt-based duel |
| `/experiments` | **Experiment Builder wizard** (mode → model → budget → review → run) |
| `/lab` | Recommender, Quality tasks, Duel types |
| `/models/<id>` | **Model Detail** (identity, capabilities, presets, history, configs) |
| `/results/<id>` | Výsledky behu + Pareto |

Všetky stránky majú boot reporter (JS chyba = viditeľná hláška, nie prázdna stránka)
a cache-busted assety (`?v=8`).