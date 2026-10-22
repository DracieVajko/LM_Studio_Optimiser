# LM STUDIO OPTIMISER — MASTER PLAN & CHECKLIST
**Zdroj:** `Delete after everything from this folder is done\` + research 2026-10-21
**Hardvér:** RTX 3060 Laptop 6 GB VRAM, Ryzen 7 5800H, 32 GB single-channel RAM
**Stav:** 2026-10-22 00:40 — P0–P6 hotové, 1027 testov prechádza

---

## 📊 SPEED @16384 (KV kvantizácia ON, jednotka = median gen tok/s)

| Model | Quant / Veľkosť | Median | Min | Max | N | Verdikt |
|---|---|---|---|---|---|---|
| **bonsai** | Q1_0, 4.4 GB | **27.2** | 9.3 | 28.9 | 100+ | 🟢 **KEEP** — tools/odysseus (extrémna kvantizácia) |
| **mirothinker v1.5 30B** | Q2_K_S, 10.7 GB | **10.08** | 5.45 | 15.50 | 20 | 🟢 **KEEP** — 🆕 najrýchlejší reasoning model |
| **qwen3.8-9b-distill** | Q4_K_M, 5.8 GB | **8.99** | 4.96 | 13.59 | 17 | 🟢 **KEEP** — 🆕 zmestí sa celý do VRAM |
| **gpt-oss orig** | MXFP4, 11.3 GB | **8.76** | 3.12 | 12.58 | 20 | 🟢 **KEEP** — reasoning + quality 1.0 |
| **gpt-oss heretic** | Q4_K_S, 13.7 GB | **7.14** | 0.54 | 14.43 | 16 | 🟢 **KEEP** — uncensored |
| **swift** | Q3_K_S, 12.9 GB | **2.16** | 1.40 | 2.60 | 16 | 🟢 **KEEP** — najrýchlejší 27B dense |
| **qwen base** | Q4_K_M, 16.5 GB | **1.83** | 0.56 | 2.04 | 16 | 🔴 DELETE — medzik bez výhody |
| **fable** | IQ4_XS, 17.6 GB | **1.70** | 0.73 | 2.00 | 16 | 🔴 DELETE — ťažší a pomalší ako swift |
| **hauhaucs** | IQ4_XS, 16.6 GB | **1.22** | 0.98 | 1.58 | 16 | 🔴 DELETE — najpomalší 27B |
| gemma-4-26b-a4b | Q4_0, 15.6 GB | 🔄 beží | – | – | – | – |
| qwen3-coder-30b | Q4_K_M, 18.6 GB | ⏳ | – | – | – | – |
| qwen3.6-35b-a3b | Q4_K_M, 22.1 GB | ⏳ | – | – | – | – |
| **swift@iq2_xs** | IQ2_XS | 10.2 GB | **1.18** | 0.92 | 2.54 | 17 |
| **swift@iq3_xs** | IQ3_XS | 13.9 GB | **0.78** | 0.68 | 1.27 | 17 |
| **swift@iq4_xs** | IQ4_XS | 16.6 GB | 🔄 beží | – | – | – |

**Kľúčové zistenia:**
- **mirothinker 30B (Q2_K_S, 10.7 GB) = 10.08 tok/s @16K** — rýchlejší než gpt-oss, reasoning model, 30B-A3B MoE
- **qwen3.8-9b-distill (5.8 GB) = 8.99 tok/s @16K** — takmer ako gpt-oss, ale **zmestí sa celý do 6 GB VRAM**
- 27B qweny (1.2–2.2 tok/s) sú o rád pomalšie než MoE/9B modely — na tomto HW sú 27B dense prakticky nepoužiteľné

---

## 🏗️ IMPLEMENTAČNÝ STAV (P0–P6)

| Fáza | Modul | Testy | Stav |
|---|---|---|---|
| **P0** Correctness | `domain/models.py` (MTP polia), `benchmark.py` (`_extract_mtp_metrics`), `cli/main.py` (rc=3), `lm_studio.py` (capability discovery) | ✅ | **hotovo** |
| **P1** Runtime | `lm_studio_capabilities.py` — KV quant/CPU threads/mmap overené ako **NEPODPOROVANÉ** cez REST → evidované ako runtime facts | ✅ | **hotovo** |
| **P2** Context | `context_explorer.py` — 1.5× bracketing, fill ratios, retrieval, FAIL_TOO_SLOW | 23 | **hotovo** |
| **P3** Quality Lab | `quality_lab.py` — 12 kategórií, 13 validátorov, 7 hard gates, reliability | 26 | **hotovo** |
| **P4** Sandbox/Duel | `sandbox_lab.py` (web/research/blender + repair loop + security), `duel.py` (controlled/best-profile + blind) | 32 | **hotovo** |
| **P5** Recommender | `recommender.py` — katalóg, fit predikcia, similarity, community DB, lokálna kalibrácia | 15 | **hotovo** |
| **P6** Web UI | `lab.html` + `lab.js` + `/api/lab/*` (recommend, quality-tasks, duel-types) | 6 | **hotovo** |

### Nové CLI príkazy
```bash
lm-optimizer lab context-explorer <model> [--detailed] [--no-retrieval]
lm-optimizer lab quality-lab <model> [--category X] [--seeds N]
lm-optimizer lab recommend [--min-tps N] [--min-context N] [--vram-only] [--json]
lm-optimizer lab duel <model-a> <model-b> [--type T] [--mode M]
```

### Nové API endpointy
```
GET /api/lab/recommend?min_tps=&min_context=&vram_only=&use_case=
GET /api/lab/quality-tasks
GET /api/lab/duel-types
GET /lab   (UI stránka)
```

---

## 📊 SÚČASNÝ STAV TESTOV (@16K speed-only, KV-quantizácia zapnutá)

| Model | Stav | Median tok/s | Min | Max | N | Poznámka |
|---|---|---|---|---|---|---|
| **gpt-oss orig** | ✅ **DONE** | **8.76** | 3.12 | 12.58 | 20 | MXFP4, 11.3 GB |
| **gpt-oss heretic** | ✅ **DONE** | **7.14** | 0.54 | 14.43 | 16 | Q4_K_S, 13.7 GB |
| **swift** | ✅ **DONE** | **2.16** | 1.40 | 2.60 | 16 | Q3_K_S, 12.9 GB — najrýchlejší 27B |
| **qwen base** | 🔄 **BEŽÍ** | 1.95* | – | – | 1/16 | Q4_K_M, 16.5 GB — predpoklad ~1.8–2.0 |
| **fable** | ✅ **DONE** | **1.70** | 0.73 | 2.00 | 16 | IQ4_XS, 17.6 GB |
| **hauhaucs** | ✅ **DONE** | **1.22** | 0.98 | 1.58 | 16 | iq4_xs, MTP |
| **bonsai** | ✅ **DONE** | **27.2** | 9.3 | 28.9 | 100+ | Q1_0, 4.4 GB — len pre tools/odysseus |

*čiastočné (1 sonda pred krachom PC)

---

## 🎯 KEEP / DELETE ODPOVEDAŇ (pre 6 GB VRAM)

### 🟢 **KEEP — nechať nainštalované**
| Model | Dôvod |
|---|---|
| **gpt-oss orig** | **Absolútny víťaz rýchlosti (8.76 tok/s)**, menší (11.3 GB), MXFP4 efektívny, quality 1.0 validovaná |
| **gpt-oss heretic** | 7.14 tok/s, uncensored varianta, 13.7 GB — stále do VRAM s KV offload |
| **swift** | **Najrýchlejší 27B (2.16 tok/s)**, najmenší 27B (12.9 GB Q3_K_S), má KV-kvantizáciu |
| **bonsai** | Pre tools/odysseus utilities (27 tok/s, 4.4 GB) |

### 🔴 **DELETE / ARCHIVE — vymazať z LM Studio**
| Model | Dôvod |
|---|---|
| **fable** | 17.6 GB IQ4_XS — ťažší ako swift, **pomalší (1.70 vs 2.16)**, MTP nepridáva rýchlosť |
| **hauhaucs** | **Najpomalší 27B (1.22 tok/s)**, agresívny MTP na CPU nepomáha |
| **qwen base** | 16.5 GB Q4_K_M, ~1.95 tok/s — **medzi swift/fable, žiadna výhoda**, väčší ako swift |

---

## 🔬 RESEARCH: QWEN 3.8 27B VARIANTY A KONFIGURÁCIE

### Dostupné kvantizácie (bartowski / mradermacher / AtomicChat)

| Quant | Veľkosť | Kvalita (perplexity/KLD) | Odporúčanie pre 6 GB VRAM |
|---|---|---|---|
| **IQ3_XXS** | ~11.2 GB | 89.4% top-1 | **Fits do VRAM**, rýchly, akceptovateľná kvalita |
| **IQ3_XS** | ~12.8 GB | 89.4% | **Fits do VRAM**, lepšia kvalita ako Q3_K_M |
| **IQ3_M** | ~14.8 GB | 91.1% | Čiastočný offload, dobrá rovnováha |
| **Q3_K_S** | ~12.3 GB | 89.0% | **Fits do VRAM**, ale pomalší ako IQ3_XS |
| **Q3_K_M** | ~13.4 GB | 89.8% | Čiastočný offload, **pomalší ako IQ4_XS** (dequant overhead) |
| **IQ4_XS** ⭐ | **~15.4 GB** | **94.1%** | **DEFAULT PICK** — optimalizovaný size/quality/speed, **q4_0 KV cache** |
| **Q4_K_S** | ~16.3 GB | 94.8% | Čiastočný offload, dobrá kvalita |
| **Q4_K_M** | ~17.4 GB | 95.0% | Referenčná kvalita, **vyžaduje offload na 6 GB** |
| **Q5_K_M** | ~19.5 GB | 96.9% | Offload-heavy, pomalší |
| **Q6_K** | ~22.5 GB | 97.4% | Near-lossless, len pre 24 GB+ VRAM |

### Kľavé zistenia z benchmarkov (batiai / AtomicChat / SpecPicks):
1. **Q3 je pomalší ako IQ4** — dequantization overhead na Q3_K kerneloch (batiai: Q3_K_M 13.0 t/s vs IQ4_XS 15.5 t/s na M4 Max)
2. **IQ4_XS je sweet spot** — imatrix calibrated, lepšia kvalita ako Q3_K_M pri menšej veľkosti
3. **Na 6 GB VRAM sa len IQ3 varianty zmestia úplne do VRAM** — Q4_K_M (17.4 GB) vždy vyžaduje CPU offload
4. **MTP na dense 27B**: n=3 je optimal (73% acceptance, 31→40 tok/s na 24 GB VRAM), na 6 GB VRAM **MTP spomaľuje** (VRAM pressure + CPU offload)
5. **KV cache quantization** kritická: q4_0/q8_0 kv cache umožňuje 16K+ kontext na 16 GB VRAM

### Recommended GGUF repozitáre pre stiahnutie:
| Repo | Quanty | Poznámka |
|---|---|---|
| **bartowski/Qwen3.8-27B-GGUF** | Všetky (IQ3_XXS → Q6_K) | **Primárny zdroj**, imatrix, MTP podpora |
| **mradermacher/Qwen-3.8-27B-Uncensored-i1-GGUF** | Všetky | Uncensored, imatrix |
| **AtomicChat (Atomic Dynamic)** | AD-IQ3_S, AD-IQ4_XS, AD-Q4_K_M | Vlastné imatrix, dynamické per-tensor |
| **georuffus/Qwen3.8-27B-i1-IQ4_KS_KT-GGUF** | IQ4_KT | Pre 16 GB VRAM, q4_0 KV cache |
| **vmarcelo/Swift-Qwen3.8-27B-MIX_GGUF** | IQ4-MIX, IQ3-MIX, IQ2-MIX | Mixed-tensor, MTP, Swift fine-tune |

### Alternatívne modely pre 6 GB VRAM (z researchu):
| Model | Quant | Veľkosť | Verw. tok/s (est.) | Use case |
|---|---|---|---|---|
| **Qwen 8B** | IQ4_XS | 4.25 GB | 15–25 | **Najrýchlejší usable**, fits celý do VRAM |
| **Qwen 14B** | IQ4_XS | ~8.5 GB | 4–10 | Hybrid GPU+RAM, coding (Qwen Coder 14B) |
| **GPT-OSS 20B** | MXFP4 | 11.3 GB | **8–12** | **MoE, 3.6B aktívnych**, reasoning low/medium/high |
| **Qwen 35B-A3B** | IQ3_M | ~13 GB | 5–14 | **MoE, 3B aktívnych**, expert offload → lepšie ako dense 27B |
| **Gemma 3 27B** | IQ4_XS | 14.77 GB | 2–5 | Sliding window attention → dlhší kontext |

---

## ⚙️ OPTIMÁLNE KONFIGURÁCIE PRE 6 GB VRAM (z researchu)

### A. Swift (máme, testované) — **BEST 27B DENSE**
```yaml
model: swift-qwen3.8-27b (Q3_K_S, 12.9 GB)
context_length: 16384
gpu_ratio: 1.0
flash_attention: true
offload_kv_cache_to_gpu: true
k_cache_quantization: q8_0
v_cache_quantization: q4_0
eval_batch_size: 2048
cpu_threads: 8
mtp: OFF (testované — nepridáva rýchlosť na 6 GB)
median: 2.16 tok/s
```

### B. GPT-OSS 20B (orig + heretic) — **BEST OVERALL**
```yaml
model: openai/gpt-oss-20b (MXFP4, 11.3 GB)
context_length: 16384
gpu_ratio: 0.55
flash_attention: true
offload_kv_cache_to_gpu: true
k_cache_quantization: q8_0
v_cache_quantization: q4_0
eval_batch_size: 128
cpu_threads: 16
reasoning: low/medium/high (oddelené meranie)
median: 8.76 tok/s (low), 7.14 tok/s (heretic)
```

### C. Qwen 27B IQ3_XS (nový stiahnuť) — **VRAM-RESIDENT 27B**
```yaml
model: bartowski/Qwen3.8-27B-GGUF:IQ3_XS (12.8 GB)
context_length: 16384
gpu_ratio: 1.0
flash_attention: true
offload_kv_cache_to_gpu: true
k_cache_quantization: q8_0
v_cache_quantization: q4_0
eval_batch_size: 128
cpu_threads: 8
mtp: test OFF vs n=3 (predpoklad: OFF lepšie na 6 GB)
# očakávané: 2–4 tok/s, celý model v VRAM
```

### D. Qwen 8B IQ4_XS (nový stiahnuť) — **SPEED KING**
```yaml
model: bartowski/Qwen3-8B-GGUF:IQ4_XS (4.25 GB)
context_length: 8192
gpu_ratio: 1.0
flash_attention: true
offload_kv_cache_to_gpu: true
k_cache_quantization: q8_0
v_cache_quantization: q4_0
eval_batch_size: 256
cpu_threads: 8
# očakávané: 15–25 tok/s
```

### E. Qwen 35B-A3B IQ3_M (MoE) — **QUALITY/SPEED BALANCE**
```yaml
model: batiai/qwen3.6-35b:iq3 (13 GB)
context_length: 16384
gpu_ratio: adaptive
flash_attention: true
offload_kv_cache_to_gpu: true
k_cache_quantization: q8_0
v_cache_quantization: q4_0
eval_batch_size: 64
cpu_threads: 16
moe_expert_placement: cpu (routed experts)
mtp: test OFF vs n=3
# očakávané: 5–14 tok/s, 3B aktívnych parametrov
```

---

## 📋 "DELETE AFTER" FOLDER — PLÁN PREVZATIA

### Obsah folderu:
```
Delete after everything from this folder is done/
├── lm studio deep think.txt              (109 KB) — technický audit LM Studio API, MTP, KV cache
├── LM_STUDIO_OPTIMISER_MASTER_SPEC_SK.md (40 KB)  — Master product specification
└── OPENCODE_MASTER_PROMPT_LM_STUDIO_OPTIMISER_SK.md (23 KB) — Agent prompt pre audit/plán
```

### Akcie:
1. **Archivovať** — presunúť do `docs/archive/` (s timestampom)
2. **Integrovať do projektu:**
   - `LM_STUDIO_OPTIMISER_MASTER_SPEC_SK.md` → `docs/specs/master-spec.md` (hlavná špecifikácia)
   - `OPENCODE_MASTER_PROMPT...` → `docs/agents/opencode-audit-prompt.md` (pre budúce audity)
   - `lm studio deep think.txt` → `docs/research/lmstudio-api-deep-dive.md` (technický research)
3. **Pôvodný folder zmazať** po overení že všetko je v `docs/`

---

## 📅 IMPLEMENTAČNÝ PLÁN (P0–P6 z OpenCode promptu)

### **P0 — Correctness Foundation** (kritické, má byť hotové v existujúcom kóde)
- [x] Requested vs Applied config
- [x] Config deduplication
- [x] Real TTFT (streaming timestamp)
- [x] Prompt/decode split
- [x] Explicit failure taxonomy
- [x] Fingerprints (HW/SW/Model)
- [x] Checkpoint/resume consistency
- [ ] **Doplnit:** MTP metrics (predicted/accepted/speedup)

### **P1 — Runtime Search** (rozšírenie existujúceho optimizeru)
- [ ] KV K/V quant capabilities (llamaKCacheQuantizationType, llamaVCacheQuantizationType)
- [ ] CPU threads (strategické body: 1, 8, 16)
- [ ] mmap cold/warm test
- [x] MTP/draft metrics (predicted/accepted/acceptance_rate/speedup_ratio)
- [ ] Interaction matrix (všetky 7 povinných interakcií)
- [ ] Fast/Detailed workflow separation

### **P2 — Context Explorer**
- [ ] 1.5× geometric bracketing (nie power-of-two)
- [ ] Binary refinement na 512 tokenov
- [ ] Loadable vs Stable vs Speed-acceptable vs Retrieval-passing
- [ ] Fill ratios (0/50/90% fast, 0/25/50/75/90/98% detailed)
- [ ] Retrieval validation (needles na 5/25/50/75/95%)
- [x] FAIL_TOO_SLOW threshold (median < 0.95 = fail)

### **P3 — Quality Lab**
- [ ] Deterministic task model (instruction/coding/reasoning/JSON/Slovak)
- [ ] Hard gates (build, unit tests, schema, exit code)
- [ ] Validators (exact match, JSON Schema, compiler, unit tests)
- [ ] Reliability repeats (4 seedy: 42, 1337, 2026, 9999)
- [ ] Task-specific scoring + variance reporting

### **P4 — Sandbox/Duel**
- [ ] Web/HTML sandbox (Playwright, responsive, accessibility)
- [ ] Research sandbox (closed-package + open-web)
- [ ] Blender/3D sandbox (Minecraft Skyblock, Python script)
- [ ] Repair loop (max 2 opravy, penalizácia skóre)
- [ ] Blind duel (Candidate A/B, randomized order)

### **P5 — Model Recommender**
- [ ] Model catalog (repo, filename, SHA256, arch, quant, active params)
- [ ] Fit classes (GPU-resident, Hybrid-interactive, Hybrid-quality, Not recommended)
- [ ] Community benchmark DB (confidence A–D)
- [ ] Similarity scoring (GPU arch, VRAM, RAM channels, CPU, OS, backend)
- [ ] Local calibration priority > community prior

### **P6 — Web UI**
- [ ] Navigation split (Dashboard, Models, Benchmark, Optimize, Context, Quality, Sandbox, Duel, Recommendations, Results, Settings)
- [ ] Experiment Builder (wizard: mode → models → scope → search profile → budget → review → run)
- [ ] Model Detail (identity, capabilities, exact file/hash/quant, presets, history, graphs)
- [ ] Results/Pareto (quality vs decode TPS, bubble=VRAM, color=context)
- [ ] Sandbox cards / Duel screen

---

## 🗂️ ČO SA UŽ ROBÍ V BEŽIACOM KÓDE (z auditov)

### Hotové / Čiastočne implementované:
- ✅ Runtime Optimizer (adaptívne viacstupňové vyhľadávanie, Pareto)
- ✅ SQLite história + checkpoint/resume
- ✅ CLI + Web UI (dashboard, models, results)
- ✅ Hardware detection + capability discovery
- ✅ Granular test modes (--speed-only, --quality-only, --reasoning-only, --mtp-only)
- ✅ Context Explorer (grundy, binary refinement)
- ✅ MTP/speculative decoding podpora (config, metrics)
- ✅ KV cache quantization (K/V cache quant type)
- ✅ Streaming TTFT (timestamp prvého tokenu)
- ✅ Requested vs Applied config logging
- ✅ Explicit failure taxonomy
- ✅ Granular test selection (preflight validácie)

### Chýbajúce / Čiastočné (gap matrix):
| Oblasť | Stav | Akcia |
|---|---|---|
| **CPU threads** | ❌ MISSING | Pridať do load space |
| **mmap cold/warm** | ❌ MISSING | Pridať do load space |
| **MTP metrics** | ⚠️ PARTIAL | Doplniť acceptance/speedup do DB |
| **Interaction matrix** | ⚠️ PARTIAL | Overiť 7 interakcií v Fast mode |
| **Fill ratios** | ❌ MISSING | Context Explorer |
| **Retrieval validation** | ❌ MISSING | Context Explorer |
| **Quality Lab** | ❌ MISSING | Nový modul |
| **Sandbox runners** | ❌ MISSING | Nový modul |
| **Blind duel** | ❌ MISSING | Nový modul |
| **Community benchmark DB** | ❌ MISSING | Nový modul + schema |
| **Cross-platform metrics** | ⚠️ PARTIAL | Windows hotové, Linux/macOS chýbajú |

---

## 🚀 ĎALŠIE KROKY (HNEĎ PO DOBEHNUTÍ QWEN BASE)

### 1. Dokončenie testov (dnes večer)
- [ ] Počkať na dokončenie qwen base @16K (cca 2–3 h)
- [ ] Overiť výsledky všetkých 6 modelov
- [ ] Spustiť `cleanup.bat` (archivuje runs/benchmarks/logs/cache, **netýka sa modelov**)

### 2. Stiahnutie nových modelov (paralelne)
- [ ] `bartowski/Qwen3.8-27B-GGUF:IQ3_XS` (12.8 GB) — VRAM-resident 27B
- [ ] `bartowski/Qwen3-8B-GGUF:IQ4_XS` (4.25 GB) — Speed king
- [ ] `batiai/qwen3.6-35b:iq3` (13 GB) — MoE 35B-A3B
- [ ] `bartowski/Qwen3-8B-Coder-GGUF:IQ4_XS` — ak existuje

### 3. Commit zmien (po cleanup)
- [ ] Audit + granular + cleanup changes → commit na `feat/single-model-guard`
- [ ] Tag `v0.6.0-granular-cleanup`

### 4. P0–P2 implementácia (týždeň 1–2)
- [ ] P0: Doplniť MTP metrics do DB + testy
- [ ] P1: CPU threads, mmap, interaction matrix, Fast/Detailed split
- [ ] P2: Context Explorer — fill ratios, retrieval, 1.5× bracketing

### 5. Research-driven optimalizácie
- [ ] Pridať IQ3_XS / IQ4_XS varianty do model catalogu
- [ ] Pridať Qwen 8B / 14B / 35B-A3B do recommenderu
- [ ] KV cache quantization (q4_0/q8_0) ako default pre kontext >8K
- [ ] MTP test OFF vs n=3 pre dense 27B (automatické vyhodnotenie speedup > 1.05)

---

## ❓ OTÁZKY PRE ŤA (rozhodnutia, ktoré neviem urobiť sám)

1. **Qwen base @16K** — chceš počkať na úplných 16 prôb, alebo stačí 8+ (aktuálne 1/16, predpoklad ~1.9)?
2. **Stiahnutie IQ3_XS / 8B / 35B-A3B** — robiť teraz (paralelne s qwen base), alebo po cleanup?
3. **cleanup.bat** — spustiť hneď po dobehnutí qwen base, alebo počkať na tvoje potvrdenie?
4. **MTP test na swifte** — už máme data (OFF), chceš explicitne testovať n=3 pre porovnanie?
5. **Quality Lab** — začať implementovať hneď (P3), alebo najprv P1+P2?
6. **gpt-oss heretic reasoning** — chceš porovnať low/medium/high reasoning na heretic variant (aktuálne máme len orig)?

---

## 📌 HRUBÝ ČASOVÝ HARMONOGRAM

| Deň | Aktivity |
|---|---|
| **Dnes** | Dokončiť qwen base, cleanup.bat, commit, stiahnutie nových modelov |
| **Zajtra** | P0 dokončenie (MTP metrics), P1 začiatok (CPU threads, mmap) |
| **Deň 3** | P1 dokončenie (interaction matrix, Fast/Detailed), P2 začiatok |
| **Deň 4–5** | P2 Context Explorer (fill ratios, retrieval, 1.5× bracketing) |
| **Týždeň 2** | P3 Quality Lab (tasks, validators, hard gates) |
| **Týždeň 3** | P4 Sandbox (Web + Research), P5 Recommender (catalog, similarity) |
| **Týždeň 4** | P6 Web UI (navigation, builder, Pareto), integrácia, testy |

---

*Pripravené: 2026-10-21 20:25 | Beží: qwen base @16K (1/16 sondy) | Ďalšia kontrola: o 30 min*