# LM STUDIO OPTIMISER — MASTER PLAN & CHECKLIST

**Zdroj:** `Delete after everything from this folder is done\` + research 2026-10-21
**Hardvér:** RTX 3060 Laptop 6 GB VRAM, Ryzen 7 5800H, 32 GB single-channel RAM
**Stav:** 2026-10-22 — P0–P6 hotové, 1151 testov prechádza

---

## SPEED @16384

> **Poznámka:** merania prebiehali pod záťažou (ffmpeg na CPU + AI upscale na RTX).
> Čísla sú **relatívne**, nie absolútne — viď `docs/MEASUREMENT_CONTAMINATION.md`.

| Model | Quant / Veľkosť | Median tok/s | N | Verdikt |
|---|---|---|---|---|
| **bonsai** | Q1_0, 4.4 GB | **27.2** | 100+ | KEEP (tools/odysseus) |
| **mirothinker v1.5 30B** | Q2_K_S, 10.7 GB | **10.08** | 20 | KEEP — reasoning |
| **qwen3.6-35b-a3b** | Q4_K_M, 22.1 GB | **9.43** | 20 | KEEP — MoE |
| **qwen3.8-9b-distill** | Q4_K_M, 5.8 GB | **8.99** | 17 | KEEP — celý v VRAM |
| **gpt-oss orig** | MXFP4, 12.1 GB | **8.76** | 20 | KEEP — reasoning |
| **qwen3-coder-30b** | Q4_K_M, 18.6 GB | **8.67** | 20 | KEEP — coding MoE |
| **gpt-oss heretic** | Q4_K_S, 14.7 GB | **7.14** | 16 | KEEP — uncensored |
| **gemma-4-26b-a4b** | Q4_0, 15.6 GB | **6.70** | 15 | KEEP — MoE |
| **swift (Q3_K_S)** | Q3_K_S, 13.9 GB | **2.16** | 16 | KEEP — najrýchlejší 27B |
| qwen base | Q4_K_M, 17.7 GB | 1.83 | 16 | DELETE |
| fable | IQ4_XS, 18.9 GB | 1.70 | 16 | DELETE |
| **swift@q2_k** | Q2_K, 11.96 GB | **1.49** | 17 | info — pomalší ako Q3_K_S |
| hauhaucs | IQ4_XS, 16.6 GB | 1.22 | 16 | DELETE |
| swift@iq2_xs | IQ2_XS, 10.2 GB | 1.18 | 17 | DELETE |
| swift@iq3_xs | IQ3_XS, 13.9 GB | 0.78 | 17 | DELETE |
| **swift@q3_k_m** | Q3_K_M, 14.54 GB | ~0.4 (kontaminované) | beží | info |
| swift@iq4_xs | IQ4_XS, 16.6 GB | nezmestí sa | – | DELETE |

### Kľúčové zistenia

- **K-kvanty > I-kvanty** na tomto HW: Q3_K_S **2.16** vs IQ3_XS **0.78** vs IQ2_XS **1.18**.
  I-kvanty majú drahšiu dequantizáciu; na CPU-bound hybridnom offloade to bolí.
- **Q2_K (1.49) je pomalší ako Q3_K_S (2.16)** — nižší quant nie je automaticky rýchlejší.
- **27B dense qweny (1.2–2.2) sú o rád pomalšie** než MoE/9B (6.7–10.1) a bonsai (27).
- Rozdiel je rádový, takže **keep/delete rozhodnutie je stabilné** aj pod záťažou.

---

## IMPLEMENTAČNÝ STAV (P0–P6)

| Fáza | Moduly | Testy | Stav |
|---|---|---|---|
| **P0** Correctness | `domain/models.py` (MTP + thermal polia), `benchmark.py` (`_extract_mtp_metrics`, `_collect_thermal`), `cli/main.py` (rc=3), `lm_studio.py` (capability discovery) | ✅ | hotovo |
| **P1** Runtime | `lm_studio_capabilities.py` (KV quant/CPU threads/mmap = nepodporované cez REST → runtime facts), rotácia finalistov (`rotate_order`) | 7 | hotovo |
| **P2** Context | `context_explorer.py` (1.5× bracketing, fill ratios, retrieval, FAIL_TOO_SLOW ≠ OOM), `ctx --factor/--round-to/--retrieval/--detailed-fill` | 23 | hotovo |
| **P3** Quality Lab | `quality_lab.py` (12 kategórií, 13 validátorov, 7 hard gates, reliability), `optimize --quality-lab` | 26 | hotovo |
| **P4** Sandbox/Duel | `sandbox_lab.py`, `sandbox_runner.py` (isolated workdir, allowlist, repair loop, retention), `web_validator.py` (stdlib), `research_web.py` (open-web + citations), `duel.py` (blind A/B) | 55 | hotovo |
| **P5** Recommender | `recommender.py` (fit podľa veľkosti súboru, MoE aktívne parametre, lokálne merania > community), `community_benchmarks.py` (21 záznamov A–D), `community_import.py` (import/export), `hf_catalog.py` (HF discovery), `model_hash.py` (SHA256 + cache) | 61 | hotovo |
| **P6** Web UI | `/lab` (Recommender, Quality tasks, Duel types, **Sandbox**), `/experiments` (Experiment Builder), `/models/<id>` (Model Detail), `/results/<id>` (Pareto) | 15 | hotovo |

### CLI

```bash
# optimalizácia
lm-optimizer optimize <model> [--quality-lab] [--test-only --test NAME]
lm-optimizer optimize <model> --speed-only --context 16384
lm-optimizer lab full-optimize <model>

# lab workflowy
lm-optimizer lab context-explorer <model> --factor 1.5 --retrieval
lm-optimizer lab quality-lab <model> --seeds 3
lm-optimizer lab recommend [--min-tps N] [--vram-only]
lm-optimizer lab duel <a> <b> --type micro_quality
lm-optimizer lab sandbox --task web_dashboard --model <model>

# katalóg a identita modelov
lm-optimizer lab hash-model <model>
lm-optimizer lab hash-all
lm-optimizer lab community-import <file.json>
lm-optimizer lab community-export
lm-optimizer lab catalogue-sync --query "qwen gguf"
```

### API

```
GET  /api/lab/recommend?min_tps=&min_context=&vram_only=&use_case=
GET  /api/lab/quality-tasks
GET  /api/lab/duel-types
GET  /api/lab/sandbox-tasks
POST /api/lab/sandbox-run          {"task": "...", "model": "..."}
GET  /api/lab/sandbox-run/{job_id}
GET  /api/lab/recommend
```

---

## ČO JE MIMO ROZSAHU (rozhodnutie používateľa)

- **Playwright / browser rendering** — sandbox artefakty si používateľ pozrie sám
- **Blender render validácia** — rovnaký dôvod, Blender nie je nainštalovaný

Statické validátory (`web_validator.py`, `score_attempt`) zostávajú a odchytia
štrukturálne chyby.

---

## OTVORENÉ POLOŽKY

| Položka | Stav |
|---|---|
| Live validácia 8 workflowov na bonsai | 🔄 pripravené: `python tools/live_validation.py prism-ml/bonsai-27b` |
| `lab hash-all` | ⏳ 5–10 min |
| Zmazať `Delete after everything from this folder is done/` | ⏳ obsah archivovaný v `docs/` |
| Push na GitHub | ⏳ čaká na povolenie |
| Čisté merania bez záťaže | ⏳ po vypnutí ffmpeg/upscale |

---

## DOKUMENTÁCIA

| Súbor | Obsah |
|---|---|
| `docs/README.md` | mapa špecifikácie na kód + honest "not implemented" |
| `docs/SUMMARY_2026-10-22.md` | detailný sumár vs. master špecifikácia |
| `docs/IMPLEMENTATION_2026-10-22.md` | implementačný sumár vĺn |
| `docs/MEASUREMENT_CONTAMINATION.md` | poznámka o záťaži počas meraní |
| `docs/HANDOFF_2026-10-22.md` | handoff s komandmi |
| `docs/specs/`, `docs/agents/`, `docs/research/` | archivované dokumenty z "Delete after" folderu |
