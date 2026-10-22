# LM Studio Optimiser — Master Product and Implementation Specification

## 0\. Účel dokumentu

Tento dokument je hlavná špecifikácia ďalšieho vývoja projektu **LM Studio Optimiser**. Má slúžiť ako kompletný kontext pre OpenCode/AI agenta aj pre človeka, ktorý bude projekt analyzovať, plánovať alebo implementovať.

Dokument nevychádza z predstavy jednoduchého benchmark skriptu. Cieľom je vytvoriť hardvérovo uvedomelý, empirický a cross-platform optimalizačný nástroj pre LM Studio, ktorý bude vedieť:

* zistiť vlastnosti hardvéru, operačného systému, LM Studio a runtime,
* objaviť parametre, ktoré sú skutočne podporované konkrétnou verziou LM Studio/runtime,
* benchmarkovať aktuálnu konfiguráciu bez jej zmeny,
* optimalizovať runtime jedného modelu,
* nájsť maximálny aj prakticky použiteľný kontext,
* hodnotiť kvalitu modelu automatickými a sandbox testami,
* robiť anonymizované duely dvoch modelov alebo dvoch presetov,
* odporúčať vhodné modely podľa hardvéru a účelu,
* vytvárať viac druhov odporúčaných presetov namiesto jedného univerzálneho „víťaza“,
* ukladať úplný audit každej požadovanej a reálne aplikovanej konfigurácie,
* fungovať na Windows, Linuxe aj macOS s backendovo špecifickým meraním,
* minimalizovať čas testovania bez toho, aby sa prehliadli relevantné kombinácie.

Kľúčový princíp projektu:

> Neexistuje univerzálne najlepší model ani univerzálne najlepšia konfigurácia. Existuje empiricky najlepší model a konfigurácia pre konkrétny hardvér, runtime, backend, účel, kontext, požadovanú kvalitu, pamäťový limit a minimálnu akceptovateľnú rýchlosť.

\---

## 1\. Kontext projektu a cieľ používateľa

Repozitár: `DracieVajko/LM\_Studio\_Optimiser`

Projekt už má alebo deklaruje modulárnu architektúru s vrstvami pre API, benchmarky, CLI, databázu, discovery, domain modely, hardware detection, optimizer, scoring, services, storage, testy a Web UI. Existuje adaptívne viacstupňové vyhľadávanie, profily Speed/Balanced/Context/Quality/Custom, hardvérovo relatívne skórovanie, Pareto frontier, SQLite história, checkpoint/resume, CLI a Web UI.

Tieto existujúce časti sa nemajú zahodiť ani nahradiť úplne novým riešením. Nové funkcie sa majú integrovať do existujúcej architektúry, používať spoločné služby a zachovať deterministické, auditovateľné a resumable workflow.

Používateľov referenčný hardvér:

* GPU: NVIDIA RTX 3060 Laptop, 6 GB VRAM,
* CPU: AMD Ryzen 7 5800H, 8 fyzických jadier / 16 threadov,
* RAM: 1 × 32 GB, single-channel,
* disk: NVMe SSD,
* typické prostredie: Windows a LM Studio,
* relevantné modely: Bonsai 27B Q1/Q+, Ministral 3B, Qwen 3.5 4B, väčšie Qwen modely, GPT-OSS 20B, Qwen 27B, Qwen Coder 30B-A3B, Qwen 35B-A3B, Gemma 27B a ďalšie alternatívy.

Dôležitá lokálna realita: Bonsai 27B Q1, Ministral 3B a Qwen 3.5 4B dosahujú na tomto hardvéri viac než 70 tok/s. Projekt preto nesmie používať zastarané alebo všeobecné predpoklady typu „7B na RTX 3060 dosiahne približne 20 tok/s“ ako pevný scoring anchor. Komunitné benchmarky majú byť iba prior/reference s explicitnou dôveryhodnosťou. Lokálne meranie má vždy posledné slovo.

Bonsai 27B demonštruje, že samotný parameter count nestačí. Nástroj musí oddeliť:

* celkový počet parametrov,
* aktívny počet parametrov pri MoE,
* bits per weight,
* presnú veľkosť GGUF,
* architektúru,
* reálnu pamäť pri načítaní,
* KV cache,
* runtime overhead,
* kvalitu po kvantizácii,
* reálny prompt/decode výkon.

\---

## 2\. Produktové pojmy, ktoré sa nesmú miešať

### 2.1 Benchmark

Benchmark iba zmeria aktuálnu alebo explicitne zvolenú konfiguráciu. Nehľadá automaticky lepšiu konfiguráciu.

### 2.2 Optimize

Optimize prehľadáva runtime konfigurácie jedného konkrétneho modelu/quant variantu a hľadá vhodné Pareto profily.

### 2.3 Evaluate / Quality Lab

Evaluate hodnotí schopnosti a kvalitu jedného modelu na automaticky validovaných úlohách. Nemá byť zamieňané s runtime optimalizáciou.

### 2.4 Sandbox Lab

Sandbox spúšťa väčšie praktické agentické úlohy, napríklad web, research alebo Blender animáciu, vrátane build/test/repair workflow.

### 2.5 Duel

Duel porovnáva dva modely, dva quanty alebo dva presety na rovnakých úlohách. Môže byť controlled alebo best-profile.

### 2.6 Context Explorer

Context Explorer cielene hľadá loadable, stable, speed-acceptable, quality-passing a odporúčaný kontext.

### 2.7 Full Optimization

Full Optimization je orchestration workflow nad existujúcimi modulmi. Nesmie duplikovať ich logiku.

### 2.8 Model Recommender

Recommender navrhuje vhodné presné modely/quanty/presety podľa hardvéru a účelu. Jeho odporúčanie sa má následne lokálne overiť benchmarkom.

\---

## 3\. Navrhovaná hlavná navigácia a samostatné režimy

Aplikácia má mať tieto samostatné režimy:

1. Dashboard
2. Models
3. Quick Benchmark
4. Runtime Optimizer
5. Context Explorer
6. Quality Lab
7. Sandbox Lab
8. Model Duel
9. Full Optimization
10. Model Recommender
11. Results \& Presets
12. Settings

Každý režim musí byť spustiteľný samostatne. Používateľ musí vedieť napríklad:

* iba zmerať Bonsai bez optimalizácie,
* iba optimalizovať load/runtime parametre Qwen modelu,
* iba nájsť maximálny kontext,
* iba spustiť HTML sandbox,
* spustiť Research alebo Blender sandbox samostatne,
* urobiť HTML duel dvoch modelov,
* spustiť kompletný sandbox duel,
* urobiť controlled duel alebo best-profile duel,
* spustiť kompletnú optimalizáciu bez sandboxu,
* voliteľne pridať sandbox úlohy do Full Optimization.

\---

## 4\. Quick Benchmark

### 4.1 Účel

Rýchlo zmerať aktuálne správanie načítaného modelu/configu bez automatického hľadania lepších parametrov.

### 4.2 Povinné metriky

* requested config snapshot,
* applied config snapshot,
* load time, ak sa model explicitne načítal,
* prompt token count,
* generated token count,
* prompt processing tok/s,
* decode/generation tok/s,
* skutočný TTFT,
* total wall time,
* stop reason,
* peak VRAM,
* peak RAM,
* shared GPU memory na Windows,
* stability/variability,
* MTP/draft proposed and accepted tokens, ak dostupné,
* MTP acceptance rate,
* teploty/clocks/throttling, ak dostupné.

### 4.3 Profily

#### Quick

* 1 warmup,
* 2 merané behy,
* krátky prompt,
* približne 256 output tokenov,
* bez rozsiahleho quality testu.

#### Standard

* 1 warmup,
* 3 merané behy,
* short/medium/long prompt,
* 256–512 output tokenov,
* základný correctness gate.

#### Detailed

* viac prompt lengths,
* viac context fill ratios,
* 5 opakovaní,
* thermal/stability monitoring,
* report mediánu, priemeru, minima, maxima, štandardnej odchýlky a CV.

\---

## 5\. Kompletný konfiguračný povrch LM Studio, ktorý má optimizer objavovať

Implementácia nesmie predpokladať, že každý parameter funguje v každej verzii, backende alebo architektúre. Capability discovery má určovať, čo možno použiť.

### 5.1 Model management a server

* list models,
* model metadata,
* model key a instance id,
* model load,
* model unload,
* download a download status, ak sa pridá integrácia,
* explicitný load vs JIT load,
* TTL,
* Auto-Evict,
* server authentication/token, ak je zapnutý,
* local vs remote endpoint,
* headless llmster,
* loaded instances,
* max concurrent predictions / parallel requests,
* continuous batching,
* model variants,
* reasoning capabilities,
* vision/tool-use capabilities,
* GGUF vs MLX.

Počas benchmarku má byť preferovaný explicitný load/unload. JIT, TTL alebo Auto-Evict nesmú nepozorovane vložiť load time do inference testu alebo unloadnúť model počas runu.

### 5.2 Load-time parametre

Minimálne objavovať a podľa capability testovať:

* `contextLength` / `context\_length`,
* GPU offload ratio alebo ekvivalent,
* multi-GPU distribution, ak bude podporované,
* `evalBatchSize`,
* Flash Attention,
* offload KV cache to GPU,
* K-cache quantization,
* V-cache quantization,
* FP16 KV cache compatibility,
* `keepModelInMemory`,
* `tryMmap`,
* seed relevantný pre load/runtime,
* `numExperts` iba tam, kde dáva architektonický zmysel,
* RoPE frequency base,
* RoPE frequency scale,
* parallel/max concurrent predictions,
* vision projector prítomnosť a vplyv,
* model-specific load fields,
* backend-specific fields.

Závislosti:

* V-cache quantization môže vyžadovať Flash Attention.
* `numExperts` sa nesmie slepo meniť na dense modeli.
* RoPE je experimentálny quality-impacting parameter.
* multi-GPU parametre sa nesmú ponúkať na single-GPU zariadení.
* vision projector sa nemá načítavať pri text-only benchmarku, pokiaľ to runtime/model nevyžaduje.

### 5.3 Inference-time parametre

* temperature,
* top-p,
* top-k,
* min-p,
* repeat penalty,
* XTC probability,
* XTC threshold,
* max output tokens,
* stop strings,
* tool-call stop strings,
* context overflow policy:

  * stop at limit,
  * truncate middle,
  * rolling window,
* structured output / JSON schema,
* reasoning level:

  * off,
  * on,
  * low,
  * medium,
  * high,
* CPU threads, ak runtime/SDK umožňuje per-prediction nastavenie,
* draft model,
* MTP/speculative decoding,
* system prompt,
* streaming,
* integrations/tools iba v samostatných agentických testoch.

Sampling parametre sa nemajú miešať do rovnakého primárneho search space s load parametrami. Menia správanie a kvalitu výstupu.

### 5.4 MTP a speculative decoding

Podporiť tri režimy:

* disabled,
* external compatible draft model,
* native MTP/NextN, ak ho runtime/model podporuje.

Ukladať:

* draft model id/hash,
* spec mode,
* proposed/predicted draft tokens,
* accepted draft tokens,
* acceptance rate,
* baseline TPS,
* speculative TPS,
* speedup ratio,
* VRAM delta,
* RAM delta,
* TTFT delta,
* kvalitatívny rozdiel,
* runtime/backend version.

MTP sa nesmie automaticky považovať za výhodu. Pri niektorých dense modeloch môže pomôcť, pri MoE alebo nevhodnom hardvéri môže spomaliť. Musí existovať OFF baseline.

### 5.5 Requested vs Applied Config

Každý run musí uložiť:

* requested config,
* applied config,
* ignored parameters,
* runtime-modified parameters,
* unsupported parameters,
* config hash.

Ak dva requested configy skončia rovnakým applied configom, ich benchmark sa má deduplikovať.

Ak runtime ignoruje kritický parameter, run má dostať napríklad status `FAIL\_CONFIG\_IGNORED` alebo warning podľa režimu.

\---

## 6\. Parameter priority a oddelené optimalizačné fázy

### 6.1 Memory/fit parametre

Testovať najskôr:

* context length,
* GPU offload,
* KV placement,
* K-cache quant,
* V-cache quant,
* Flash Attention,
* eval batch size.

### 6.2 Performance parametre

Po nájdení stabilného memory profilu:

* CPU threads,
* mmap,
* MTP/draft,
* parallelism/continuous batching,
* keep-in-memory,
* eventuálne platform-specific backend settings.

### 6.3 Quality-impacting parametre

Oddelený search alebo Quality Lab:

* sampler,
* reasoning effort,
* RoPE,
* structured decoding,
* context overflow policy,
* tool setup.

### 6.4 Povinné interakcie

Aj Fast režim musí overiť minimálne:

* GPU offload × KV placement,
* Flash Attention × V-cache quantization,
* context × KV quantization,
* eval batch × context,
* MTP × GPU offload,
* MTP × context,
* CPU threads × partial CPU offload,
* parallelism × context/memory.

\---

## 7\. Runtime Optimizer

### 7.1 Účel

Nájsť najlepšiu runtime konfiguráciu jedného modelu/quant súboru pri zvolenom cieli.

### 7.2 Ciele/profily

* Fastest single-user,
* Lowest TTFT,
* Highest prompt throughput,
* Highest server throughput,
* Balanced,
* Largest usable context,
* Lowest VRAM,
* Lowest total memory,
* Best quality-preserving,
* Custom weighted profile.

### 7.3 Fast režim

Workflow:

1. Capability discovery.
2. Hardware/runtime fingerprint.
3. Memory pre-filter (`estimate-only`, ak dostupné).
4. Baseline.
5. Greedy/adaptive coordinate search:

   * GPU offload,
   * Flash Attention,
   * KV placement,
   * K/V quant,
   * eval batch,
   * CPU threads,
   * mmap,
   * MTP,
   * context ceiling.
6. Povinné interaction verification.
7. Sequential halving kandidátov.
8. Top 3 finalist validation.
9. Pareto profiling.
10. Save presets.

Finalisti majú byť rotovaní v poradí, napríklad ABC/BCA/CAB, aby neskorší kandidát nebol systematicky poškodený thermal throttlingom.

### 7.4 Detailed režim

Hierarchické exhaustive prehľadávanie:

1. exhaustive search platných memory topológií,
2. prerezanie infeasible a dominated konfigurácií,
3. exhaustive performance sweep na Pareto kandidátoch,
4. quality-preservation gates,
5. context ceiling search pre odlišné Pareto profily,
6. rozsiahle finalist repetitions,
7. stability a thermal validation.

„Exhaustive“ znamená všetky relevantné a reálne odlišné applied konfigurácie, nie slepý kartézsky súčin duplicitných/ignorovaných/neplatných konfigurácií.

\---

## 8\. Context Explorer

### 8.1 Kontextové výsledky musia byť rozdelené

Pre každý model/config reportovať:

* Max loadable context,
* Max inference-stable context,
* Max context s decode TPS nad limitom,
* Max prompt-processing acceptable context,
* Max retrieval-quality-passing context,
* Max overall quality-passing context,
* Recommended interactive context.

Príklad reportu:

```text
Max loadable context:      110 592
Max stable context:         98 304
Max >= 1 tok/s context:     76 288
Max retrieval-passing:      65 536
Recommended interactive:    32 768
```

### 8.2 Konfigurácia hraníc

Predvolený používateľský limit:

* decode TPS < 1.0 = príliš pomalé,
* ale tento výsledok nesmie byť označený ako OOM.

Hysteréza:

* PASS: median >= 1.05 tok/s,
* BORDERLINE: 0.95–1.05 tok/s,
* FAIL\_TOO\_SLOW: median < 0.95 tok/s.

Borderline sa má zopakovať viackrát.

### 8.3 Search algoritmus

#### Krok 1: hard model maximum

Získať max context deklarovaný modelom/runtime. Ísť nad tento limit iba v explicitnom experimentálnom RoPE režime.

#### Krok 2: exponential/geometric bracketing

Nepoužívať iba power-of-two. Použiť napríklad faktor 1.5 a zaokrúhlenie na 256 alebo 512 tokenov:

```text
4 096
6 144
9 216
13 824
20 736
31 104
46 592
69 888
104 960
```

#### Krok 3: nájsť posledný PASS a prvý FAIL

FAIL môže byť OOM, timeout, crash alebo TOO\_SLOW.

#### Krok 4: binary refinement

Spresniť interval na nastaviteľné rozlíšenie, napríklad 512 alebo 1 024 tokenov. Výsledok môže byť 76 288, nie iba 64K/128K.

### 8.4 Dve úrovne kontextového testu

#### Allocation-only

* load modelu so zvolenou context allocation,
* verify applied context,
* krátky prompt,
* 32–64 output tokenov.

#### Real fill validation

Iba kandidáti blízko hranice alebo relevantné body:

* 0 %,
* 25 %,
* 50 %,
* 75 %,
* 90 %,
* 98 % fill.

Fast režim používa minimálne 0/50/90 %. Detailed režim používa širšiu maticu.

### 8.5 Kontekstový filler

Dlhý prompt nemá byť len opakovaná veta. Má byť deterministicky generovaný, tokenovo kontrolovaný a primerane reprezentovať:

* prose,
* code,
* structured records,
* mixed task.

Ukladať:

* allocated context,
* actual input tokens,
* fill ratio,
* target needle position,
* prompt TPS,
* TTFT,
* decode TPS,
* memory metrics.

### 8.6 Retrieval validation

Vkladať kontrolné údaje do pozícií napríklad 5/25/50/75/95 %. Model musí vrátiť presnú odpoveď. Nesmie sa tvrdiť, že 90K context je „quality usable“, ak sa iba načíta, ale nevie z neho spoľahlivo vybrať informáciu.

\---

## 9\. Failure taxonomy

Použiť explicitné statusy:

* PASS,
* BORDERLINE,
* FAIL\_OOM\_LOAD,
* FAIL\_OOM\_INFERENCE,
* FAIL\_TOO\_SLOW,
* FAIL\_TIMEOUT,
* FAIL\_CRASH,
* FAIL\_CONTEXT\_OVERFLOW,
* FAIL\_QUALITY,
* FAIL\_UNSUPPORTED,
* FAIL\_CONFIG\_IGNORED,
* FAIL\_OUTPUT\_INVALID,
* FAIL\_THERMAL\_CONTAMINATION,
* FAIL\_SWAP\_CONTAMINATION,
* CANCELLED,
* INTERRUPTED\_RESUMABLE.

Každý fail uloží:

* fázu,
* requested/applied config,
* posledné dostupné metriky,
* stdout/stderr alebo API error,
* threshold a observed value,
* platform/backend info,
* či možno zlyhanie použiť na pruning.

\---

## 10\. Zrýchlenie optimizeru bez vynechania relevantných možností

### 10.1 Pre-filter

Použiť memory estimate, capabilities a model metadata na odstránenie očividne nemožných kombinácií. Estimate nie je náhrada reálneho testu.

### 10.2 Cache

Cache key má obsahovať:

* model SHA256,
* model key a file size,
* runtime/LM Studio version,
* OS,
* backend,
* GPU driver,
* hardware fingerprint,
* requested config,
* applied config,
* benchmark suite version,
* task version,
* power profile,
* thermal policy.

### 10.3 Deduplicate applied configs

Ak runtime znormalizuje viac requested configs do rovnakej applied config, benchmark vykonať iba raz.

### 10.4 Failure-region pruning

OOM alebo monotónny fit failure môže prerezávať vyššie memory demands iba v rámci rovnakej topológie. Zmena KV quant, KV placement alebo Flash Attention vytvára novú oblasť.

### 10.5 Sequential halving

* všetci kandidáti dostanú krátky test,
* vyradiť jasne horšiu časť,
* zvyšným dať opakovania,
* finalistom plnú validáciu.

Nevyraďovať kandidáta pri rozdiele v rámci meracieho šumu.

### 10.6 Early abort

Zastaviť a logovať, ak:

* rolling TPS zostáva hlboko pod prahom po warmup intervale,
* OOM,
* runtime hang,
* output loop/repetition,
* memory safety limit,
* swap thrashing,
* thermal throttling mimo povoleného limitu.

### 10.7 Resume/checkpoints

Všetky workflow majú byť resumable po:

* zatvorení UI,
* výpadku servera,
* runtime crashi,
* reštarte OS, ak sú potrebné artefakty zachované.

\---

## 11\. Skutočné meranie namiesto chybných proxy metrík

### 11.1 TTFT

Nepoužívať odhad typu `total\_time \* 0.1`. Použiť:

* server/SDK reported TTFT, alebo
* timestamp prvého streamovaného text tokenu.

Ukladať `ttft\_source`:

* stream\_measured,
* server\_reported,
* estimated,
* unavailable.

Estimated údaj nesmie mať rovnakú dôveryhodnosť ani scoring weight.

### 11.2 Prompt vs decode

Oddeliť:

* prompt processing tok/s,
* decode tok/s,
* TTFT,
* full end-to-end time.

Pri reasoning modeli oddeliť:

* raw generation výkon,
* reasoning token count,
* visible answer token count,
* time to usable final answer.

### 11.3 MTP

Samostatne ukladať acceptance a speedup. Vysoká acceptance bez speedup nie je víťazstvo.

### 11.4 Thermal a power

Na notebooku:

* AC power status,
* power profile,
* GPU/CPU clocks,
* temperature,
* throttle indicator.

Kontaminované runs označiť a nevybrať automaticky ako winner.

\---

## 12\. Quality Lab — hodnotenie kvality jedného modelu

### 12.1 Quality dimensions

Predvolená všeobecná rubrika:

* Correctness: 35 %,
* Instruction following: 20 %,
* Completeness: 15 %,
* Robustness: 10 %,
* Efficiency: 10 %,
* Maintainability: 10 %.

Váhy majú byť task-specific.

### 12.2 Hard gates pred subjektívnym scoringom

Najprv:

* build,
* unit tests,
* schema validation,
* runtime exit code,
* required files,
* exact required functionality.

Nefunkčný výstup nesmie vyhrať iba vizuálnym dojmom.

### 12.3 Kategórie Quality Lab

Samostatne zapínateľné:

* instruction following,
* coding,
* reasoning,
* structured JSON,
* tool calling,
* Slovak language,
* English language,
* translation/grammar,
* summarization,
* long-context retrieval,
* hallucination resistance,
* research synthesis.

### 12.4 Automatické validátory

Preferovať:

* exact match,
* JSON Schema,
* compiler/type checker,
* unit tests,
* regex/AST constraints,
* expected tool name/arguments,
* retrieval key,
* deterministic file checks.

LLM judge má byť sekundárny, nie jediný autoritatívny hodnotiteľ.

### 12.5 Robustness

Testovať:

* temperature 0/fixed seed,
* model-recommended profile,
* viac seedov, napríklad 42/1337/2026.

Uložiť:

* mean,
* median,
* worst case,
* standard deviation,
* success rate.

Stabilný model s 84/85/83 môže byť prakticky lepší než model s 95/61/88.

\---

## 13\. Sandbox Lab

Sandbox má byť samostatný produktový modul. Tri základné tasky musia byť spustiteľné:

* každý samostatne pre jeden model,
* vybraná kombinácia,
* kompletná suite,
* duel pre konkrétny task,
* kompletný duel.

### 13.1 Web/HTML Sandbox

Task má mať presné acceptance criteria, napríklad:

* responzívny dashboard,
* štyri metriky,
* aspoň tri modely,
* filter,
* detail modelu,
* light/dark theme,
* mobilné a desktop rozlíšenie,
* bez nepovoleného externého API,
* plne spustiteľný zo sandboxu.

Automatické hodnotenie:

* build/runtime status,
* required files,
* console errors,
* network policy,
* Playwright functional tests,
* responsive screenshoty,
* accessibility,
* visual overflow/overlap,
* maintainability.

Metriky:

* first-pass success,
* time to first valid build,
* repair count,
* tokens to success,
* final functional score,
* visual score,
* code quality.

### 13.2 Research Sandbox

Dve verzie:

#### Closed-source package

Vopred pripravené dokumenty:

* primárne/oficiálne zdroje,
* technické články,
* komunitné reporty,
* zastaraný zdroj,
* konfliktný zdroj,
* irelevantný dokument.

Hodnotiť:

* factuality,
* citation precision,
* citation recall,
* source quality,
* coverage,
* synthesis,
* conflict handling,
* recommendation quality,
* unsupported claims.

Penalizovať vymyslené zdroje a neexistujúce citácie.

#### Open-web research

Všetci kandidáti dostanú rovnaké nástroje, query budget a čas/token budget. Výsledok je menej reprodukovateľný, ale prakticky agentický.

### 13.3 Blender/3D animation Sandbox

Štandardizovať napríklad Blender Python task s Minecraft-inspired Skyblock scénou:

* floating block island,
* grass/dirt/stone materiály,
* strom,
* chest,
* block avatar,
* svetlo,
* camera orbit,
* presný frame range,
* uloženie `.blend`,
* render kontrolných snímok,
* bez manuálnych krokov.

Automatický validator:

* script exit code,
* `.blend` existuje a znovu sa otvorí,
* required objects,
* materials,
* camera,
* lights,
* keyframes,
* frame range,
* rendered frames,
* frames nie sú čierne,
* kamera nemieri mimo scény,
* začiatok/stred/koniec nie sú identické.

Skóre:

* execution/render,
* required elements,
* animation correctness,
* composition/visibility,
* visual quality,
* code quality.

### 13.4 Repair loop

* attempt 1: task,
* pri fail: model dostane stdout/stderr, failing tests, relevantný screenshot a zostávajúci repair budget,
* maximálne nastaviteľný počet opráv, napríklad 2.

Ukladať:

* first-pass success,
* final success,
* repair count,
* cumulative tokens,
* wall time,
* skóre po každom pokuse.

Možná penalizácia functional score:

* first pass: 100 %,
* po jednej oprave: 90 %,
* po dvoch: 80 %.

Penalty musí byť configurable a samostatne reportovaná.

### 13.5 Sandbox security

* izolovaný filesystem/workdir,
* network default off, explicitne povoliť pre open-web task,
* process timeout,
* CPU/RAM/disk limity,
* žiadne host secrets,
* whitelist executables alebo containerization,
* čistenie artefaktov podľa retention policy,
* nezobrazovať model-generated commands bez sanitizácie.

\---

## 14\. Model Duel

### 14.1 Typy

* Micro Quality Duel,
* HTML/Web Duel,
* Research Duel,
* Blender Duel,
* Complete Sandbox Duel,
* Complete Model Duel.

### 14.2 Controlled Duel

Rovnaké:

* context,
* sampling,
* seed,
* prompt,
* tools,
* time/token budget,
* repair count,
* sandbox permissions.

Odpovedá: ktorý model je lepší za rovnakých podmienok?

### 14.3 Best-profile Duel

Každý model používa vlastný najlepší uložený preset a odporúčané sampler/thinking nastavenia.

Odpovedá: ktorý model poskytuje lepší reálny výsledok pri svojom najlepšom známom nastavení?

### 14.4 Blind evaluation

Výstupy anonymizovať ako Candidate A/B. Judge nemá poznať:

* model,
* parameter count,
* quant,
* speed,
* predchádzajúce skóre.

Poradie A/B randomizovať alebo vyvážiť.

### 14.5 Výsledok

Ukladať:

* winner/tie,
* confidence,
* dimension winners,
* critical failures,
* hard-gate status,
* performance difference,
* quality difference,
* efficiency difference.

Počiatočné interpretácie:

* rozdiel < 3 body = tie,
* 3–7 = weak win,
* > 7 = clear win.

Prahy majú byť configurable.

\---

## 15\. Full Optimization

### 15.1 Fast Full Optimize

1. Hardware discovery.
2. Runtime/capability discovery.
3. Quick baseline.
4. Fast runtime optimization.
5. Fast context search.
6. Compact Quality Lab.
7. Final validation.
8. Save recommended presets.
9. Sandbox je voliteľný.

### 15.2 Detailed Full Optimize

1. Preflight a fingerprint.
2. Capabilities.
3. Memory estimates.
4. Detailed Runtime Optimizer.
5. Detailed Context Explorer.
6. Full Quality Lab.
7. Voliteľné Sandbox tasky.
8. Interaction retest.
9. Top finalist repeated validation.
10. Pareto frontier.
11. Save multiple presets.

### 15.3 Výsledné presety

* Fastest,
* Lowest TTFT,
* Best Balanced,
* Best Quality,
* Largest Usable Context,
* Lowest VRAM,
* Lowest RAM,
* Best Coding,
* Best Research,
* Best Creative/Sandbox,
* Best Server Throughput.

\---

## 16\. Model Recommender

### 16.1 Vstupy

* hardware fingerprint,
* VRAM,
* RAM capacity a channels/bandwidth,
* CPU,
* OS/backend,
* use case,
* minimum TPS,
* minimum context,
* VRAM-only vs hybrid allowed,
* quality preference,
* max model/download size,
* vision/tool/reasoning requirements.

### 16.2 Výstupné triedy

* GPU-resident,
* Hybrid interactive,
* Hybrid quality-first,
* Not recommended / insufficient confidence.

### 16.3 Presný modelový záznam

* repository,
* filename,
* SHA256,
* architecture,
* total params,
* active params,
* quant,
* bits/weight,
* file size,
* max context,
* capabilities,
* expected fit,
* source/confidence,
* suggested starting config.

### 16.4 Community benchmark database

Každý záznam:

* presný hardware,
* VRAM,
* CPU,
* RAM capacity/channels,
* OS,
* power profile,
* LM Studio/runtime/backend version,
* model repository/file/hash,
* quant,
* context,
* GPU offload,
* Flash Attention,
* KV placement/quant,
* eval batch,
* threads,
* mmap,
* MTP,
* prompt/decode TPS,
* TTFT,
* memory,
* benchmark methodology,
* source,
* confidence grade.

Confidence:

* A: raw JSON + exact hash + full settings,
* B: complete command/config + benchmark output,
* C: hardware/model/quant/context/TPS,
* D: community claim with missing settings.

Lokálne výsledky majú vyššiu prioritu než komunitný prior.

### 16.5 Similarity

Zohľadniť:

* GPU architecture,
* VRAM,
* GPU bandwidth/TDP,
* CPU,
* RAM channels/bandwidth,
* OS,
* backend,
* runtime version.

Pri modeli spilling to RAM zvýšiť váhu RAM bandwidth/channel similarity.

\---

## 17\. Cross-platform pravidlá

### 17.1 Všeobecný fingerprint

* OS/version,
* LM Studio version,
* runtime version,
* backend,
* GPU/VRAM,
* driver,
* CUDA/Vulkan/Metal/MLX,
* CPU,
* RAM capacity/channels,
* disk,
* power profile,
* model hash.

### 17.2 Windows

Monitorovať:

* dedicated GPU memory,
* shared GPU memory,
* process working set,
* commit size,
* page file/swap,
* power plan,
* AC/battery,
* WDDM správanie.

Konfigurácia, ktorá „prejde“ iba vďaka shared GPU memory a prudko spomalí, nesmie byť považovaná za čistý VRAM fit.

### 17.3 Linux

Monitorovať:

* RSS/PSS,
* swap,
* major page faults,
* GPU telemetry,
* CPU governor,
* thermal throttling,
* headless llmster/systemd mode.

### 17.4 macOS

* unified memory namiesto oddelenej RAM/VRAM,
* memory pressure,
* swap,
* Metal vs MLX vs llama.cpp,
* backendy nemiešať v jednom comparison bucket bez označenia.

### 17.5 Metric adapters

Použiť spoločné rozhranie s platform adapters:

* WindowsMetricsProvider,
* LinuxMetricsProvider,
* MacMetricsProvider.

Chýbajúca metrika má byť `null/unavailable`, nie vymyslená nula.

\---

## 18\. Scoring a Pareto

### 18.1 Nezlučovať všetko do jedného čísla

Reportovať minimálne:

* Performance,
* Quality,
* Reliability,
* Context,
* Memory,
* Efficiency,
* Sandbox.

Voliteľný composite score až nad nimi.

### 18.2 Performance

* decode TPS,
* prompt TPS,
* TTFT,
* end-to-end time,
* server aggregate throughput pri throughput profile.

### 18.3 Quality

Task-specific validators, hard gates a rubric.

### 18.4 Reliability

* pass rate,
* worst case,
* CV/variance,
* crash count,
* invalid output count,
* repair dependency.

### 18.5 Efficiency

* VRAM,
* RAM,
* shared memory/swap,
* generated tokens,
* repair count,
* energy, ak dostupná.

### 18.6 Pareto frontier

Neexistuje jediný winner. Zachovať konfigurácie, ktoré nie sú dominované vo všetkých relevantných rozmeroch.

Príklady vizualizácie:

* quality vs decode TPS,
* bubble size = VRAM,
* color = context,
* quality vs time-to-success,
* context vs memory.

\---

## 19\. Web UI návrh

### 19.1 Dashboard

Karty:

* LM Studio status,
* loaded model,
* current preset,
* VRAM/RAM,
* last benchmark,
* best known config.

Quick actions:

* Quick Benchmark,
* Optimize Loaded Model,
* Test Maximum Context,
* Run Quality Check,
* Start Duel.

Recent runs:

* mode,
* model,
* status,
* score,
* TPS,
* context,
* Open Report,
* Resume.

### 19.2 Model Detail

* Overview,
* capabilities,
* exact file/hash/quant,
* presets,
* history,
* graphs,
* actions:

  * Benchmark,
  * Optimize,
  * Context Test,
  * Quality Test,
  * Sandbox,
  * Add to Duel.

### 19.3 Experiment Builder

Wizard:

1. Mode.
2. Model/model pair.
3. Scope/test checkboxes.
4. Search mode.
5. Budget and thresholds.
6. Review requested vs fixed dimensions.
7. Run/resume.

Namiesto sľubovania presného času ukázať:

* max candidates,
* repetitions,
* token budget,
* repair budget,
* stop threshold,
* safety limits,
* checkpoint status.

### 19.4 Sandbox UI

Karty:

* Web App,
* Research,
* 3D Animation.

Každá:

* Run Single Model,
* Add to Duel,
* Configure Task,
* Previous Results.

### 19.5 Results

* dimension scores,
* raw metrics,
* confidence,
* source of each metric,
* requested/applied config diff,
* fail reasons,
* Pareto plots,
* context boundary chart,
* compare presets,
* save/export preset/report.

\---

## 20\. Navrhovaná interná architektúra

```text
lm\_optimizer/
├─ workflows/
│  ├─ quick\_benchmark.py
│  ├─ runtime\_optimization.py
│  ├─ context\_explorer.py
│  ├─ quality\_lab.py
│  ├─ sandbox\_suite.py
│  ├─ model\_duel.py
│  ├─ model\_recommender.py
│  └─ full\_optimization.py
├─ runtime/
│  ├─ capabilities.py
│  ├─ lmstudio\_rest.py
│  ├─ lmstudio\_sdk.py
│  ├─ lms\_cli.py
│  ├─ loader.py
│  └─ applied\_config.py
├─ search/
│  ├─ fast\_search.py
│  ├─ detailed\_search.py
│  ├─ sequential\_halving.py
│  ├─ pareto.py
│  ├─ pruning.py
│  └─ deduplication.py
├─ context/
│  ├─ search.py
│  ├─ fillers.py
│  ├─ retrieval.py
│  ├─ thresholds.py
│  └─ validators.py
├─ quality/
│  ├─ tasks/
│  ├─ validators/
│  ├─ scoring.py
│  └─ reliability.py
├─ sandbox/
│  ├─ web/
│  ├─ research/
│  ├─ blender/
│  ├─ runner.py
│  ├─ repair\_loop.py
│  └─ isolation.py
├─ duel/
│  ├─ controller.py
│  ├─ anonymizer.py
│  ├─ blind\_judge.py
│  └─ comparison.py
├─ recommender/
│  ├─ catalog.py
│  ├─ fit\_predictor.py
│  ├─ similarity.py
│  └─ local\_calibration.py
├─ metrics/
│  ├─ common.py
│  ├─ windows.py
│  ├─ linux.py
│  ├─ macos.py
│  └─ thermal.py
└─ ui/
   ├─ dashboard/
   ├─ models/
   ├─ experiments/
   ├─ sandbox/
   ├─ duel/
   ├─ recommender/
   └─ results/
```

Spoločné služby:

* ModelLoader,
* CapabilityDiscovery,
* BenchmarkRunner,
* MetricsCollector,
* ConfigVerifier,
* ResultStore,
* QualityValidator,
* HardwareMonitor,
* CheckpointManager,
* ArtifactManager.

Duel, Sandbox, Quick Benchmark a Optimizer musia používať rovnaký metric pipeline.

\---

## 21\. Databázový model — minimum

### OptimizationRun

* id,
* workflow type,
* mode,
* status,
* model identity/hash,
* hardware fingerprint,
* software fingerprint,
* task/suite version,
* timestamps,
* checkpoint,
* thresholds,
* benchmark params.

### CandidateRun

* requested config,
* applied config,
* config diff,
* repetition,
* prompt/task id,
* seed,
* status/fail reason,
* raw metrics,
* quality metrics,
* thermal/memory flags.

### ModelIdentity

* repository,
* key,
* filename,
* SHA256,
* architecture,
* quant,
* bits/weight,
* total/active params,
* size,
* max context,
* capabilities.

### HardwareFingerprint

* OS,
* CPU,
* GPU(s),
* VRAM,
* RAM,
* channels,
* disk,
* drivers,
* backend,
* power profile.

### SandboxArtifact

* task id/version,
* created files,
* logs,
* screenshots/renders,
* test reports,
* repair history,
* judge results.

### Preset

* target model/hash,
* profile type,
* load config,
* inference config,
* validated context,
* confidence,
* source run ids.

\---

## 22\. Preset/workflow príklady

### Quick Performance

```yaml
workflow: quick\_benchmark
warmups: 1
repetitions: 2
output\_tokens: 256
quality\_checks: false
sandbox: false
```

### Runtime Fast

```yaml
workflow: runtime\_optimization
search\_mode: fast
context\_mode: fixed
context\_length: 8192
finalist\_repetitions: 5
quality\_gate: compact
```

### Context Maximum

```yaml
workflow: context\_explorer
growth\_factor: 1.5
refinement\_resolution: 512
minimum\_decode\_tps: 1.0
fill\_ratios: \[0.0, 0.5, 0.9]
retrieval\_validation: true
```

### HTML only

```yaml
workflow: sandbox
tasks: \[web]
repair\_attempts: 2
visual\_validation: true
functional\_tests: true
```

### Duel HTML only

```yaml
workflow: duel
models: \[model\_a, model\_b]
tasks: \[sandbox\_web]
mode: controlled
blind: true
```

### Full Detailed

```yaml
workflow: full\_optimization
runtime: detailed
context: detailed
quality: full
sandbox:
  web: true
  research: true
  blender: true
save\_presets: true
```

\---

## 23\. Implementačné priority

### P0 — správnosť merania a audit

1. Requested vs applied config.
2. Config deduplication.
3. Skutočný TTFT.
4. Prompt/decode separation.
5. Explicit fail taxonomy.
6. Model/runtime/hardware fingerprint.
7. Checkpoint/resume consistency.

### P1 — runtime optimalizácia

1. KV K/V quant capabilities.
2. CPU threads.
3. mmap cold/warm test.
4. MTP/draft metrics.
5. interaction matrix.
6. Fast/Detailed workflow separation.

### P2 — kontext

1. 1.5× bracketing.
2. binary refinement.
3. loadable vs stable vs speed-acceptable.
4. fill ratios.
5. retrieval validation.
6. FAIL\_TOO\_SLOW threshold.

### P3 — Quality Lab

1. deterministic task model.
2. hard gates.
3. validators.
4. reliability repeats.
5. task-specific scoring.

### P4 — Sandbox/Duel

1. Web sandbox.
2. Research package.
3. Blender sandbox.
4. repair loop.
5. blind duel.
6. controlled vs best-profile.

### P5 — Recommender a community data

1. model catalog.
2. fit classes.
3. benchmark schema/confidence.
4. similarity score.
5. local calibration.

### P6 — Web UI

1. navigation split.
2. experiment builder.
3. model detail.
4. context charts.
5. results/Pareto.
6. Sandbox cards/Duel screen.

\---

## 24\. Akceptačné kritériá

Projekt je pripravený na dôveryhodné používanie, keď:

1. Každý benchmark vie ukázať requested aj applied config.
2. TTFT je meraný alebo označený ako nedostupný, nie falošne odhadnutý.
3. Prompt TPS a decode TPS sú oddelené.
4. OOM, too slow, timeout, crash a quality failure sú odlišné stavy.
5. Context result nerozpráva iba o alokácii, ale aj o reálnom fill a retrieval.
6. Context search môže nájsť hodnotu ako 76K, nie iba power-of-two.
7. Fast mode prerezáva iba bezpečne a validuje interakcie.
8. Detailed mode pokrýva všetky odlišné podporované konfigurácie.
9. MTP má OFF baseline, acceptance a reálny speedup.
10. Quality score nevie prehlasovať hard functional failure.
11. Sandbox tasky sa dajú spustiť jednotlivo aj ako suite.
12. Duel sa dá spustiť pre jeden task aj complete suite.
13. Controlled a best-profile duel sú oddelené.
14. Judge je blind a model nehodnotí sám seba.
15. Windows/Linux/macOS používajú platformové metric adapters.
16. Chýbajúca metrika je `unavailable`, nie nula.
17. Výsledky sú resumable a cache je fingerprintovaná.
18. Full Optimization iba orchestruje moduly.
19. Výsledkom je viac Pareto presetov, nie jeden absolútny winner.
20. Recommender jasne odlíši community prior od lokálne nameraného výsledku.

\---

## 25\. Inštrukcie pre OpenCode/AI agenta

1. Najprv urob audit existujúceho repozitára a necommitnutých zmien.
2. Neimplementuj duplicitný systém, ak už podobná služba existuje.
3. Zachovaj existujúce verejné API a databázovú kompatibilitu, pokiaľ nie je zdokumentovaná migrácia.
4. Každý návrh mapuj na konkrétne existujúce moduly/súbory.
5. Jasne rozlišuj:

   * čo už existuje,
   * čo je čiastočne implementované,
   * čo chýba,
   * čo treba refaktorovať.
6. Najprv priprav implementačný plán po menších fázach.
7. Pri každej fáze požaduj alebo vytvor:

   * unit tests,
   * integration tests,
   * schema/database migration tests,
   * cross-platform fallback behavior.
8. Nepredpokladaj parameter support. Použi capability discovery.
9. Nehardcoduj konkrétnu GPU, VRAM ani absolútnu referenčnú rýchlosť.
10. Neukladaj iba výsledné skóre; zachovaj raw metrics.
11. Neoptimalizuj všetky rozmery naraz.
12. Nezamieňaj kvalitu, rýchlosť a praktickú použiteľnosť.
13. Všetky experimental parametre jasne označ.
14. Každý pruning krok musí mať zdokumentovaný dôvod a musí byť auditovateľný.
15. Pri neistote preferuj čiastočnú, testovanú implementáciu pred veľkým neovereným refaktorom.

\---

## 26\. Očakávaný prvý výstup OpenCode

Po načítaní tohto dokumentu má OpenCode najprv vytvoriť audit a plán, nie okamžite meniť všetko naraz.

Požadovaný výstup:

1. Súčasná architektúra repozitára.
2. Zoznam relevantných existujúcich súborov/tried/funkcií.
3. Mapa tejto špecifikácie na existujúci kód.
4. Gap analysis:

   * implemented,
   * partial,
   * missing,
   * conflicting.
5. Identifikácia rizikových miest a technického dlhu.
6. Návrh dátových modelov a migrácií.
7. Návrh workflow interfaces.
8. Fázovaný implementation plan podľa P0–P6.
9. Test plan pre každú fázu.
10. Explicitný zoznam otázok iba tam, kde sa nedá bezpečne rozhodnúť z repozitára.

Agent nesmie tvrdiť, že parameter alebo endpoint existuje iba na základe tejto špecifikácie. Musí ho overiť proti aktuálnemu kódu, používanej LM Studio verzii, runtime capabilities alebo oficiálnemu SDK/API schema.

\---

## 27\. Konečná vízia

Výsledný LM Studio Optimiser nemá byť obyčajný „VRAM calculator“ ani skript, ktorý otestuje niekoľko powers-of-two. Má byť experimentálna platforma, ktorá vie odpovedať:

* Aký model je najlepší pre môj hardvér a môj účel?
* Ktorý quant je výhodnejší pri reálnej kvalite?
* Aké runtime nastavenia sú najrýchlejšie?
* Aký je najväčší kontext, ktorý sa načíta?
* Aký je najväčší kontext, ktorý zostáva nad 1 tok/s?
* Aký je najväčší kontext, pri ktorom model ešte správne využíva informácie?
* Zrýchľuje MTP tento model na tomto stroji, alebo ho spomaľuje?
* Je lepší Bonsai 27B Q1, Qwen 3.5 4B alebo Ministral 3B pre coding, research, web a 3D tvorbu?
* Dokáže model vytvoriť funkčný projekt na prvý pokus?
* Dokáže sa opraviť po chybovom reporte?
* Ktorý preset je najlepší pre interaktívny chat, coding, research, veľký context alebo server throughput?

Každá odpoveď musí byť založená na auditovateľnom lokálnom meraní, jasnej metodike, platformovom fingerprintingu, raw dátach, quality gates a dôveryhodnosti zdrojov.

