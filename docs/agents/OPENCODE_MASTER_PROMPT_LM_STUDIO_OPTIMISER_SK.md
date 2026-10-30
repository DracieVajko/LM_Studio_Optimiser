# OpenCode Master Prompt — LM Studio Optimiser Audit, Architecture Plan and Implementation Roadmap

## Rola

Pracuješ ako senior Python softvérový architekt, performance engineer, benchmark metodik, cross-platform systems engineer a product engineer pre projekt **LM Studio Optimiser**.

Tvojou úlohou nie je bezhlavo prepisovať projekt ani okamžite implementovať všetky nápady. Najprv musíš dôkladne preskúmať existujúci repozitár, aktuálne necommitnuté zmeny, testy, databázové modely, API klientov, Web UI a existujúcu optimalizačnú metodiku. Až potom pripravíš presný gap audit a realizovateľný implementačný plán.

Tento prompt vložím do OpenCode spolu s prílohou:

`LM_STUDIO_OPTIMISER_MASTER_SPEC_SK.md`

Master špecifikácia je produktový a technický cieľ. Nie je dôkazom, že konkrétny endpoint alebo parameter existuje v aktuálne používanej verzii LM Studio. Všetky capability tvrdenia musíš overiť proti:

1. existujúcemu kódu projektu,
2. aktuálnym runtime odpovediam a capability discovery,
3. nainštalovanému LM Studio/SDK/API schema,
4. existujúcim testom a dokumentácii v repozitári.

Ak nájdeš rozdiel medzi špecifikáciou a reálnou capability, neprispôsobuj realitu špecifikácii. Rozdiel explicitne zdokumentuj a navrhni bezpečný fallback.

---

# 1. Hlavný cieľ

Rozšíriť existujúci LM Studio Optimiser na modulárnu experimentálnu platformu, ktorá dokáže:

- benchmarkovať aktuálnu konfiguráciu bez jej zmeny,
- optimalizovať runtime jedného modelu,
- presne hľadať maximálny aj prakticky použiteľný kontext,
- objektívne hodnotiť kvalitu jedného modelu,
- spúšťať samostatné praktické Sandbox úlohy,
- porovnávať dva modely alebo dva presety v blind dueli,
- spustiť úplný optimalizačný workflow kombinujúci vybrané moduly,
- odporúčať konkrétne modely a quanty podľa hardvéru a use case,
- pracovať na Windows, Linuxe a macOS,
- ukladať auditovateľné raw výsledky, requested/applied konfigurácie a failure reasons,
- vytvárať viac Pareto presetov namiesto jedného univerzálneho víťaza.

Existujúcu architektúru zachovaj a rozširuj. Neprepisuj fungujúce časti bez jasného dôvodu, testov a migračného plánu.

---

# 2. Referenčný hardvér a praktická realita

Referenčná zostava používateľa:

- NVIDIA RTX 3060 Laptop GPU, 6 GB VRAM,
- AMD Ryzen 7 5800H,
- 8 fyzických jadier / 16 threadov,
- 1 × 32 GB RAM, single-channel,
- NVMe SSD,
- Windows,
- LM Studio.

Používateľ lokálne pozoruje viac než 70 tok/s pri modeloch ako:

- Bonsai 27B Q1/Q+,
- Ministral 3B,
- Qwen 3.5 4B.

Z toho vyplýva:

- nepoužívaj hardcoded absolútne speed anchors,
- parameter count nie je spoľahlivý odhad fitu ani rýchlosti,
- rozlišuj total params, active params, bits/weight, GGUF size, architecture, KV cache a runtime overhead,
- komunitné benchmarky sú iba prior s confidence grade,
- lokálne empirické meranie má posledné slovo,
- single-channel RAM je významná pri partial CPU/RAM offloade,
- Windows shared GPU memory nesmie byť mylne označená ako čistý VRAM fit.

---

# 3. Povinný pracovný režim

## 3.1 Pracuj najprv v audit/planning režime

V prvej fáze:

- nič zásadné nerefactoruj,
- nemaž súbory,
- nemen databázové schémy,
- nepridávaj veľké závislosti,
- neprepisuj Web UI,
- nevytváraj paralelnú duplicitnú architektúru.

Najprv preskúmaj repozitár a vytvor audit.

## 3.2 Rešpektuj necommitnuté zmeny

Pred akoukoľvek zmenou preskúmaj:

```bash
git status --short
git diff --stat
git diff
git diff --cached
```

Necommitnuté zmeny považuj za rozpracovanú prácu používateľa.

Nesmieš:

- resetovať ich,
- checkoutovať ich preč,
- prepísať ich bez analýzy,
- predpokladať, že commitnutý `main` je aktuálnejší než lokálny stav.

V audite jasne rozlišuj:

- stav v HEAD,
- lokálne necommitnuté zmeny,
- nové untracked súbory,
- konflikty medzi nimi.

## 3.3 Nepýtaj sa otázky, ktoré vieš zistiť z repozitára

Najprv prehľadaj:

- README,
- docs,
- pyproject,
- requirements,
- config,
- tests,
- API klientov,
- domain modely,
- services,
- optimizer,
- scoring,
- database/storage,
- UI/API routes.

Otázku polož iba vtedy, ak rozhodnutie nemožno bezpečne odvodiť z kódu, dokumentácie alebo existujúcich konvencií.

## 3.4 Zachovaj traceability

Pri každom návrhu uveď:

- existujúci súbor/triedu/funkciu,
- zistený problém alebo medzeru,
- navrhovanú zmenu,
- dôvod,
- riziká,
- potrebné testy,
- migračné dopady.

---

# 4. Prvá povinná fáza: kompletný audit repozitára

## 4.1 Zmapuj architektúru

Vytvor mapu:

```text
entrypoints
CLI
Web UI
API routes
LM Studio client/runtime adapters
hardware detection
model discovery
capability discovery
candidate generation
optimizer orchestration
benchmark runner
quality/correctness logic
scoring/normalization
Pareto logic
database/storage
checkpoint/resume
reporting
presets
configuration
tests
CI
```

Pri každom module uveď:

- jeho zodpovednosť,
- hlavné triedy/funkcie,
- vstupy/výstupy,
- kto ho volá,
- či je testovaný,
- či sa prekrýva s iným modulom.

## 4.2 Audituj existujúcu optimalizačnú metodiku

Presne zisti:

- aké load parametre sa dnes optimalizujú,
- ako vzniká search space,
- ako funguje coarse search,
- ako funguje refinement,
- ako sa vyhodnocujú failures,
- ako sa meria TTFT,
- ako sa meria prompt/decode performance,
- ako sa deteguje VRAM/RAM,
- ako sa počíta quality score,
- ako sa počíta stability,
- ako funguje Pareto frontier,
- ako sa ukladá requested config,
- či sa ukladá applied config,
- ako funguje checkpoint/resume,
- čo sa deje po runtime crashi,
- ako sa rozlišuje OOM od timeoutu alebo pomalosti.

## 4.3 Audituj Web UI

Zisti:

- existujúcu navigáciu,
- hlavné obrazovky,
- formuláre pre spustenie optimalizácie,
- stavový manažment,
- progress reporting,
- výsledkové obrazovky,
- grafy,
- API kontrakty,
- možnosti resume/cancel,
- čo je funkčné a čo iba placeholder.

## 4.4 Audituj Sandbox

Používateľ uvádza, že už má pripravené tri testy alebo ich základy:

1. web/HTML stránka,
2. research/deep-research úloha,
3. 3D animácia/script s Minecraft-inspired Skyblock témou.

Nájdi ich presné súbory a zisti:

- či ide iba o prompty alebo plný runner,
- ako sa model spúšťa,
- kde vznikajú artefakty,
- či existujú validátory,
- či existuje repair loop,
- či existuje single-model režim,
- či existuje duel,
- či je hodnotenie automatické alebo manuálne,
- aký je security model sandboxu.

## 4.5 Audituj LM Studio capabilities

Zisti, čo projekt reálne používa:

- native REST v1,
- OpenAI-compatible API,
- Python SDK,
- TypeScript SDK,
- `lms` CLI,
- priame runtime/llama.cpp možnosti.

Pre každý parameter urč:

- supported,
- partially supported,
- GUI-only,
- SDK-only,
- CLI-only,
- unavailable,
- unknown and requires probing.

---

# 5. Požadovaný prvý výstup

Po audite vytvor jeden detailný dokument, napríklad:

`docs/OPENCODE_IMPLEMENTATION_AUDIT_AND_PLAN.md`

Musí obsahovať:

## 5.1 Executive summary

Stručne:

- čo už projekt robí dobre,
- najväčšie correctness problémy,
- najväčšie chýbajúce capability,
- odporúčané poradie implementácie.

## 5.2 Architecture map

Presné súbory, triedy, funkcie a závislosti.

## 5.3 Gap matrix

Pre každú požiadavku z master špecifikácie:

```text
IMPLEMENTED
PARTIAL
MISSING
CONFLICTING
UNVERIFIED
```

Ku každej položke:

- dôkaz v kóde,
- presná medzera,
- odporúčaná zmena.

## 5.4 Local changes analysis

Čo pridávajú necommitnuté zmeny a ako menia gap audit.

## 5.5 Risk register

Minimálne:

- databázové migrácie,
- async/concurrency,
- runtime crashes,
- cross-platform telemetry,
- sandbox security,
- API version drift,
- unsupported LM Studio fields,
- long-running workflow resume,
- thermal/noise contamination,
- UI/API compatibility.

## 5.6 Implementation roadmap

Fázy P0 až P6, každá s:

- scope,
- konkrétne súbory,
- nové/menené interfaces,
- databázové zmeny,
- testy,
- akceptačné kritériá,
- rollback/fallback.

## 5.7 Test plan

- unit,
- integration,
- fake/mock LM Studio,
- real runtime smoke test,
- Windows/Linux/macOS fallback,
- database migration,
- resumability,
- deterministic benchmark suite,
- sandbox isolation.

Po vytvorení auditu zatiaľ neimplementuj ďalšiu fázu, kým audit jasne neurčí bezpečné hranice. Ak je však úloha spustená v režime, kde sa očakáva pokračovanie bez ďalšieho potvrdenia, pokračuj iba P0 s malými, izolovanými a testovanými commit-ready zmenami.

---

# 6. Cieľová produktová architektúra

Režimy musia byť oddelené:

1. Quick Benchmark
2. Runtime Optimizer
3. Context Explorer
4. Quality Lab
5. Sandbox Lab
6. Model Duel
7. Full Optimization
8. Model Recommender
9. Results & Presets

Full Optimization iba orchestruje existujúce workflow. Nesmie vlastniť duplicitnú benchmark, scoring alebo context logiku.

Odporúčané workflow rozhranie:

```python
class Workflow(Protocol):
    def validate_request(self, request): ...
    def plan(self, request, capabilities): ...
    async def run(self, context): ...
    async def resume(self, checkpoint): ...
    async def cancel(self): ...
```

Toto je návrh, nie povinný názov. Použi existujúce projektové konvencie, ak už existujú vhodné abstractions.

---

# 7. Quick Benchmark požiadavky

Quick Benchmark iba meria, neoptimalizuje.

Povinné metriky:

- model identity a hash,
- requested config,
- applied config,
- prompt tokens,
- generated tokens,
- prompt TPS,
- decode TPS,
- skutočný TTFT,
- wall time,
- stop reason,
- VRAM/RAM/shared GPU memory,
- stability,
- MTP proposed/accepted tokens,
- thermal/power flags, ak dostupné.

Profily:

- Quick,
- Standard,
- Detailed.

Nepoužívaj estimated TTFT ako plnohodnotnú metriku. Ak skutočný TTFT nemožno získať, ulož `unavailable` alebo explicitný lower-confidence source.

---

# 8. Runtime Optimizer požiadavky

## 8.1 Objavované parametre

Capability discovery má podľa runtime podporovať čo najviac z:

### Load-time

- context length,
- GPU offload,
- eval batch,
- Flash Attention,
- KV placement,
- K-cache quantization,
- V-cache quantization,
- FP16 KV,
- keep model in memory,
- mmap,
- num experts iba ak relevantné,
- RoPE base/scale ako experimental,
- parallel/max concurrent predictions,
- model-specific fields.

### Inference-time

- temperature,
- top-p,
- top-k,
- min-p,
- repeat penalty,
- XTC,
- max tokens,
- stop strings,
- context overflow policy,
- structured output,
- reasoning effort,
- CPU threads,
- draft model,
- MTP.

## 8.2 Requested vs applied

Každý candidate musí uložiť:

```text
requested config
applied config
ignored fields
modified fields
unsupported fields
```

Rovnaké applied configy deduplikuj.

## 8.3 Fast search

Použi:

- capability pre-filter,
- memory estimate,
- baseline,
- coordinate/adaptive search,
- povinné interaction tests,
- sequential halving,
- top-3 repeated validation,
- Pareto presets.

## 8.4 Detailed search

Použi hierarchické exhaustive prehľadávanie platných a skutočne odlišných konfigurácií. Nie naivný úplný kartézsky súčin.

## 8.5 Povinné interakcie

- GPU offload × KV placement,
- Flash Attention × V-cache quant,
- context × KV quant,
- eval batch × context,
- MTP × GPU offload,
- MTP × context,
- threads × CPU offload,
- parallelism × context/memory.

---

# 9. Context Explorer požiadavky

Výsledok musí rozlišovať:

- max loadable,
- max stable,
- max nad speed threshold,
- max retrieval-passing,
- max quality-passing,
- recommended interactive.

## 9.1 Search

1. model/runtime max,
2. geometrický rast približne 1.5×,
3. nájsť PASS/FAIL bracket,
4. binárne spresniť na 512 alebo 1 024 tokenov,
5. validovať reálnym context fill.

Výsledok môže byť napríklad 76 288, nie iba power-of-two.

## 9.2 Too slow

Používateľský default:

```text
decode TPS < 1.0 = FAIL_TOO_SLOW
```

Použi hysterézu:

- median >= 1.05 PASS,
- 0.95–1.05 BORDERLINE,
- < 0.95 FAIL_TOO_SLOW.

Tento status nesmie byť OOM.

## 9.3 Fill ratios

Fast:

```text
0 %, 50 %, 90 %
```

Detailed:

```text
0 %, 25 %, 50 %, 75 %, 90 %, 98 %
```

## 9.4 Retrieval

Needles na viacerých pozíciách. Presná validácia odpovede. Samotné načítanie context allocation nestačí.

---

# 10. Failure model

Použi explicitné stavy:

```text
PASS
BORDERLINE
FAIL_OOM_LOAD
FAIL_OOM_INFERENCE
FAIL_TOO_SLOW
FAIL_TIMEOUT
FAIL_CRASH
FAIL_CONTEXT_OVERFLOW
FAIL_QUALITY
FAIL_UNSUPPORTED
FAIL_CONFIG_IGNORED
FAIL_OUTPUT_INVALID
FAIL_THERMAL_CONTAMINATION
FAIL_SWAP_CONTAMINATION
CANCELLED
INTERRUPTED_RESUMABLE
```

Každý failure musí mať:

- stage,
- requested/applied config,
- raw error,
- partial metrics,
- thresholds,
- pruning eligibility.

---

# 11. MTP/speculative decoding

Podpor režimy:

- disabled,
- external draft model,
- native MTP/NextN.

Meraj:

- predicted draft tokens,
- accepted draft tokens,
- acceptance rate,
- TPS with/without,
- speedup ratio,
- TTFT delta,
- VRAM/RAM delta,
- quality preservation.

Nikdy neodporúčaj MTP iba preto, že ho model podporuje.

---

# 12. Quality Lab

Quality Lab hodnotí jeden model, nie duel.

Samostatne zapínateľné kategórie:

- instruction following,
- coding,
- reasoning,
- JSON/structured,
- tool calling,
- Slovak,
- English,
- translation/grammar,
- summarization,
- long-context retrieval,
- hallucination resistance,
- research synthesis.

Poradie validácie:

1. hard gates,
2. deterministic validators,
3. task-specific rubric,
4. sekundárny blind judge, ak treba.

Preferované validátory:

- exact match,
- JSON Schema,
- unit tests,
- compiler/type checker,
- regex/AST,
- expected tool call,
- retrieval key.

Model nesmie hodnotiť vlastný výstup.

Reportuj:

- correctness,
- instruction following,
- completeness,
- reliability,
- maintainability,
- efficiency,
- mean/median/worst case,
- pass rate,
- variance.

---

# 13. Sandbox Lab

Každý task musí byť spustiteľný:

- jednotlivo pre jeden model,
- ako časť suite,
- ako samostatný duel,
- ako complete sandbox duel.

## 13.1 Web/HTML

Automaticky kontrolovať:

- build/run,
- required files,
- console errors,
- network policy,
- Playwright functionality,
- responsive screenshots,
- accessibility,
- overflow/overlap,
- code maintainability.

Ukladať:

- first-pass success,
- time/tokens to valid build,
- repair count,
- functional score,
- visual score.

## 13.2 Research

Podpor:

- closed-source controlled package,
- open-web agentic research.

Closed package má obsahovať relevantné, konfliktné, zastarané a irelevantné zdroje.

Hodnotiť:

- factuality,
- citation precision/recall,
- source quality,
- coverage,
- synthesis,
- conflict handling,
- unsupported claims.

## 13.3 Blender/3D

Minecraft-inspired Skyblock task cez Blender Python.

Validovať:

- exit code,
- `.blend`,
- required objects,
- materials,
- camera/light,
- keyframes,
- frame range,
- renders,
- non-black frames,
- visible motion,
- composition.

## 13.4 Repair loop

Max repair budget konfigurovateľný. Model pri oprave dostane iba relevantné logs/tests/screenshots, nie riešenie.

Samostatne reportuj first-pass a final success.

## 13.5 Security

- isolated workdir,
- network default off,
- timeout,
- resource limits,
- no host secrets,
- controlled executable access,
- artifact retention.

---

# 14. Model Duel

Typy:

- Micro Quality Duel,
- Web Duel,
- Research Duel,
- Blender Duel,
- Complete Sandbox Duel,
- Complete Model Duel.

## Controlled

Rovnaký context, sampler, seed, prompt, tools, budget a repair count.

## Best-profile

Každý model používa najlepší validovaný preset.

## Blind

Anonymizovať ako Candidate A/B. Skryť model, quant, parameter count, TPS a históriu.

Result:

- winner alebo tie,
- confidence,
- dimension winners,
- hard failures,
- performance/quality/efficiency delta.

---

# 15. Model Recommender

Nesmie byť iba VRAM kalkulačka.

Vstupy:

- hardware,
- RAM channels,
- backend/OS,
- min TPS,
- min context,
- VRAM-only/hybrid,
- use case,
- quality preference,
- required capabilities.

Výstupy:

- GPU-resident,
- hybrid interactive,
- hybrid quality-first,
- not recommended.

Community benchmark záznam musí mať confidence A–D podľa completeness. Lokálne kalibrovaný výsledok má vyššiu prioritu.

---

# 16. Cross-platform

Implementuj spoločné metric rozhranie a platform adapters.

## Windows

- dedicated VRAM,
- shared GPU memory,
- working set,
- commit/page file,
- WDDM,
- power profile,
- AC/battery.

## Linux

- RSS/PSS,
- swap,
- major page faults,
- GPU telemetry,
- CPU governor,
- headless mode.

## macOS

- unified memory,
- memory pressure,
- swap,
- Metal/MLX/llama.cpp backend distinction.

Chýbajúca hodnota = `null/unavailable`, nikdy vymyslená 0.

---

# 17. Web UI cieľ

Navrhni samostatnú navigáciu:

```text
Dashboard
Models
Benchmark
Optimize
Context
Quality
Sandbox
Duel
Recommendations
Results
Settings
```

Dashboard:

- LM Studio status,
- loaded model,
- current preset,
- resource usage,
- last run,
- quick actions,
- recent/resumable runs.

Model detail:

- identity/capabilities,
- exact file/hash/quant,
- presets,
- history,
- graphs,
- actions.

Experiment builder:

1. mode,
2. models,
3. scope,
4. search/test profile,
5. budget/thresholds,
6. review,
7. run/resume.

Sandbox UI:

- Web App,
- Research,
- 3D Animation,
- Run Single Model,
- Add to Duel,
- Configure,
- Previous Results.

Results:

- raw metrics,
- dimension scores,
- confidence,
- config diff,
- failures,
- Pareto,
- context boundary,
- preset export.

Najprv audituj existujúce UI. Nevytváraj nový frontend stack, ak nie je potrebný.

---

# 18. Databáza a kompatibilita

Minimálne entity/concepts:

- OptimizationRun,
- CandidateRun,
- ModelIdentity,
- HardwareFingerprint,
- SoftwareFingerprint,
- SandboxArtifact,
- Preset,
- BenchmarkSource/CommunityBenchmark.

Každá schema zmena potrebuje:

- migration,
- backward compatibility plan,
- migration tests,
- handling starých `null` polí,
- export/import stability.

Raw metrics nikdy nenahrádzaj iba agregovaným skóre.

---

# 19. Implementačné priority

## P0 Correctness foundation

- requested/applied config,
- deduplication,
- real TTFT,
- prompt/decode split,
- failure taxonomy,
- fingerprints,
- resume consistency.

## P1 Runtime search

- K/V quant capability,
- CPU threads,
- mmap cold/warm,
- MTP metrics,
- interaction matrix,
- Fast/Detailed separation.

## P2 Context

- 1.5× bracketing,
- binary refinement,
- loadable/stable/usable split,
- fill ratios,
- retrieval,
- too-slow threshold.

## P3 Quality Lab

- deterministic tasks,
- hard gates,
- validators,
- robustness,
- scoring.

## P4 Sandbox/Duel

- Web,
- Research,
- Blender,
- repair,
- blind duel,
- controlled/best-profile.

## P5 Recommender

- catalog,
- fit classes,
- benchmark confidence,
- similarity,
- local calibration.

## P6 Web UI

- navigation,
- builder,
- model detail,
- results/Pareto,
- sandbox/duel.

---

# 20. Kódové a testovacie pravidlá

- Dodržuj existujúci style, typing, linting a test framework.
- Nevytváraj giant god class.
- Oddeľ orchestration od runtime clients, metrics, scoring a persistence.
- Async lifecycle musí korektne riešiť cancel/timeouts/process cleanup.
- Testy nesmú vyžadovať reálnu GPU, pokiaľ nejde o explicitný hardware integration test.
- Vytvor fake LM Studio server/fixtures pre:
  - supported config,
  - ignored parameter,
  - normalized applied config,
  - OOM,
  - timeout,
  - malformed stream,
  - runtime crash,
  - MTP stats,
  - unavailable telemetry.
- Deterministické testy nesmú závisieť od poradia dictionary alebo náhodného sampling bez seedu.
- Každá nová failure vetva potrebuje test.
- Každý pruning rule potrebuje test, že neodstráni platný priestor pri zmene memory topology.
- Každý workflow potrebuje cancel/resume test.

---

# 21. Čo nesmieš urobiť

- Nesmieš zahodiť necommitnuté zmeny.
- Nesmieš tvrdiť, že GUI parameter je dostupný cez API bez overenia.
- Nesmieš hardcodovať jednu GPU alebo 6 GB limit.
- Nesmieš používať parameter count ako jediný fit predictor.
- Nesmieš miešať prompt TPS a decode TPS.
- Nesmieš počítať fake TTFT z total time.
- Nesmieš označiť TOO_SLOW ako OOM.
- Nesmieš vyhlásiť maximálny context iba preto, že sa alokoval.
- Nesmieš dať quality score nad hard functional failure.
- Nesmieš nechať testovaný model hodnotiť vlastný výstup.
- Nesmieš automaticky považovať MTP za výhodu.
- Nesmieš považovať Windows shared GPU memory za dedicated VRAM.
- Nesmieš vytvoriť duplicitný benchmark pipeline pre Duel/Sandbox.
- Nesmieš implementovať celý projekt v jednom neauditovateľnom kroku.

---

# 22. Formát odpovede po audite

Odpovedz v tejto štruktúre:

```markdown
# LM Studio Optimiser Audit and Plan

## 1. Repository state
## 2. Uncommitted changes
## 3. Current architecture
## 4. Existing optimizer workflow
## 5. Existing LM Studio capabilities
## 6. Existing benchmark/scoring correctness
## 7. Existing Sandbox and UI
## 8. Gap matrix
## 9. Critical correctness defects
## 10. Proposed target architecture
## 11. Database/API migrations
## 12. P0–P6 roadmap
## 13. Test strategy
## 14. Cross-platform risks
## 15. Recommended first implementation slice
## 16. Files expected to change
## 17. Acceptance criteria
```

Pri každom zistení cituj konkrétny path a symbol, napríklad:

```text
lm_optimizer/services/benchmark.py::BenchmarkRunner.run_case
```

Ak je niečo iba odhad, označ to. Nevymýšľaj existujúce súbory alebo funkcie.

---

# 23. Odporúčaný prvý implementation slice po audite

Ak audit potvrdí predpoklady, prvý slice má byť malý correctness foundation:

1. zaviesť requested vs applied config model,
2. ukladať config diff,
3. odstrániť alebo degradovať estimated TTFT,
4. oddeliť prompt/decode metrics,
5. zaviesť explicitnú failure taxonomy,
6. doplniť test fixtures,
7. zachovať spätnú kompatibilitu DB/API,
8. aktualizovať report bez veľkého UI redesignu.

Tento slice musí byť commit-ready a plne testovaný pred Context Explorer/Sandbox rozšíreniami.

Ak audit ukáže, že časť už existuje v necommitnutých zmenách, nereimplementuj ju. Dokonči alebo oprav existujúce riešenie.

---

# 24. Definition of Done pre celý cieľ

Projekt je hotový až vtedy, keď:

- každý run má requested/applied config,
- TTFT má pravdivý source,
- prompt/decode sú oddelené,
- failures sú explicitné,
- Fast a Detailed workflow sú samostatné,
- context search nájde ne-power-of-two hranicu,
- context je validovaný reálnym fill a retrieval,
- MTP má baseline a speedup,
- Quality Lab má hard gates,
- Web/Research/Blender sa dajú spustiť samostatne,
- Duel podporuje jeden task aj complete suite,
- controlled a best-profile sú oddelené,
- blind judging funguje,
- cross-platform metrics majú fallback,
- Full Optimization iba orchestruje,
- výsledkom sú Pareto presety,
- Recommender odlišuje community prior a local evidence,
- všetko je resumable, auditovateľné a testované.

---

# 25. Začni teraz

1. Prečítaj `LM_STUDIO_OPTIMISER_MASTER_SPEC_SK.md`.
2. Preskúmaj celý repozitár.
3. Preskúmaj HEAD aj všetky necommitnuté zmeny.
4. Spusti existujúce bezpečné statické kontroly a testy, ktoré nemenia dáta.
5. Vytvor `docs/OPENCODE_IMPLEMENTATION_AUDIT_AND_PLAN.md`.
6. V odpovedi zhrň najdôležitejšie zistenia, riziká a odporúčaný prvý slice.
7. Nezačni veľký refaktor pred dokončením auditu.
