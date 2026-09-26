# Benchmarky do detailu: čo sa testuje, ako a prečo

> CURRENT: od Phase A/B stratégie beží najprv lacný speed probe (kap. 9)
> a plný 5-testový suite až na finalistoch. Popis 5 testov nižšie platí
> pre QUALITY fázu bez zmien. Aktuálna stratégia: `docs/OPTIMIZATION_STRATEGY.md`.

Zdroj pravdy: `lm_optimizer/benchmark/suite.py`, `lm_optimizer/services/benchmark.py`,
`lm_optimizer/services/quality.py`, `lm_optimizer/services/optimizer.py`.
Bez spustených modelov — len čítanie kódu.

## 1. Život jedného kandidáta (presný postup)

Každá testovaná konfigurácia (`LoadConfiguration`: context, gpu_ratio, flash,
KV on/off, eval/physical batch, parallel, checkpoints, MoE) prejde týmto:

| # | Krok | Čo sa deje | Nastavenia |
|---|---|---|---|
| 0 | Príprava hostiteľa | unload všetkého, snapshot VRAM/RAM/swap, overenie prázdneho servera | `prepare_host()` |
| 1 | LOAD | načítanie modelu danou konfiguráciou (REST, alebo `lms load --gpu` pri `--gpu-via-cli`) | timeout až 600 s |
| 2 | VERIFY | echo-load konfigurácia vs. aplikovaná (`MATCH/PARTIAL/MISMATCH/UNKNOWN`) | `echo_load_config=true` |
| 3 | PREHEAT | 1× overovací + N× zahrievací chat (výsledky sa ZAHADZUJÚ) | prompt "Say hi in 5 words.", max 30 tok., temp 0.3; meria sa len `warmup_time_ms` |
| 4 | MERANÉ BEHY | každý z 5 testov × `repetitions` (default 3) | teploty a max_tokens podľa testu (kap. 5) |
| 5 | UNLOAD | vyloženie + overenie prázdneho servera | vždy vo `finally` |
| 6 | AGREGÁCIA | z opakovaní medián; kvalita z najkompletnejšej vzorky | medián rýchlostí, stabilita = 1 − CV |
| 7 | QUALITY GATE | 6 heuristických kontrol, priemer ≥ threshold profilu (speed 0.95 / balanced 0.97 / quality 0.99), inak `QUALITY_FAILED` bez skóre | `QualityConfig(minimum_score)` |
| 8 | SKÓRE | vážený súčet normalizovaných zložiek × váhy profilu | 7 komponentov (kap. 4) |

Pri zlyhaní loadu: stav `LOAD_FAILED`, `score=None`, `quality=None`, do histórie áno, do výberu víťaza nikdy.

## 2. Čo sa mení v ktorej fáze

| Fáza | Mení sa | Nemení sa |
|---|---|---|
| Coarse (≤20) | context, gpu_ratio, flash, KV, eval batch | physical/parallel/checkpoints |
| Refinement | okolie víťazov + flash/KV interakcie | — |
| Stage 4 | po jednom: eval batch, physical batch, parallel, checkpoints | ostatné |
| Micro (≤6) | susedstvo podľa profilu | teplota NIKDY |
| Validation (5×) | nič — ten istý config opakovane | všetko |
| `ctx` sweep | len context (KV cesta zvlášť) | runtime |

Teplota sa neladí nikdy: je fixná na test (0.0–0.3, creative výnimka 0.9) a
precision probe (0.2 vs default) je len informatívny.

## 3. Čo sa počas testu sleduje (každý jeden beh)

Z `BenchmarkMetrics` + obálky výsledku:

- `load_time_ms` — len samotný load (warmup ho nekontaminuje)
- `warmup_time_ms` — zvlášť, do metrík nevstupuje
- `prompt_tokens / completion_tokens / total_tokens` — zo servera (`usage`)
- `prompt_tok_s` — prompt_tokens / prompt_processing_ms
- `generation_tok_s` — zo servera (`tokens_per_second`), inak completion/generation_ms
- `estimated_ttft_ms` — reálny `time_to_first_token_seconds` zo servera, inak heuristika 10 % celkového času (preto "estimated")
- `prompt_processing_ms / generation_ms` — rozdelenie celkového času
- `peak_vram_gb / peak_ram_gb` — nvidia-smi + psutil pred/po (total − min free); `None` = nemerateľné, nikdy odhad
- `output_text` — plný text odpovede (v DB skrátený), `error`, `success`
- Eventy: `LOAD_REQUESTED/SUCCEEDED/FAILED`, `PREHEAT_STARTED` (+ms), `BENCHMARK_*`, `QUALITY_*`, `CONFIG_ELIGIBLE/REJECTED`
- Verifikácia: kanál (`REST`/`CLI`), `requested vs applied`, `MATCH/...`

## 4. Speed testy — mechanika (`_run_single_case`)

1. Stopky `perf_counter()` → `POST /api/v1/chat` (prompt testu, jeho teplota, jeho `max_tokens`, `reasoning="off"`).
2. Prázdny výstup + reasoning off → **presne jeden** retry s `reasoning="on"` (niektoré modely bez myslenia neodpovedajú); stále prázdne → `success=False, "Empty output"` (nekazí štatistiku absurdnými tok/s).
3. Čas sa delí: ak server dal `tokens_per_second`, `generation_ms = completion / tok_s`, zvyšok je prompt; inak pomerné delenie podľa tokenov.
4. TTFT: reálne číslo zo servera, inak 10 % času (označené estimated).
5. Agregácia: **medián** cez opakovania (odolný voči výkyvom); stabilita = `1 − stdev/mean` rýchlostí; text pre kvalitu = **najkompletnejšia** vzorka (nie medián — jeden useknutý beh nesmie zabiť dobrý config).

## 5. Accuracy testy — mechanika (`QualityEvaluator`)

Šesť dimenzií 0.0–1.0, `overall` = priemer; **check = dimenzia ≥ 0.9**
(`28/30` = 28 prejdených kontrol z 30). Agregát = priemer cez testy,
`súčet passed / súčet total`. Evaluátor je deterministický (žiadny random);
variabilita medzi behmi je sampling modelu, nie chyba merania.

**`structured_output` (format):** strihne ``` fence; neplatný JSON → okamžite
`overall=0.0`. Inak: povinné kľúče `name/age/skills/address` (chýbajúci −0.25
každý), typy (`age` int, `skills` list, `address` dict, inak 0.5), koniec na
`}` (inak 0.5).

**`coding_task` (coding):** strihne fence; `def find_duplicates` v kóde
(0/1); zmienka `O(n)/O(1)/linear/constant` (1.0/0.7); žiadne `class /`
`if __name__` (1.0/0.7); riadny koniec (1.0/0.5); `coding_correctness` =
`task_completion`.

**Všeobecné (`instruction/reasoning/context`):** dĺžka vs `min_tokens`
(pomer, max 1.0); reasoning: musí obsahovať niečo z
`difference/pattern/add/sequence/72` (inak 0.5); context: kľúčové slová
`solar/wind/hydro/geothermal/biomass`, skóre `min(1.0, found/5*1.5)`;
instruction factual vždy 1.0. Riadny koniec: posledný znak po ostrihaní
markdownu v `. ! ? ) " ' ] }` (inak 0.5). **Repetícia:** 2–4-gramy slov
(bez `*` `` ` `` `# _ ~ > |`); ak unikátnych < 70 % → `no_malformed = 0.0`.

## 6. Presné prompty a nastavenia (doslovne zo `suite.py`)

Styly menia len teploty: precise `{coding: 0.1, reasoning: 0.1}`,
creative `{context: 0.9}`, balanced bezo zmeny. `max_tokens` sa škáluje
`--max-tokens-scale` (min 32) a stropuje `context_length // 4`.
Poctivo: `seed = 42` je len zaznamenaný v `benchmark_params`, REST ho
odmieta, takže sa neposiela — determinizmus stojí na fixných nízkych
teplotách. Rovnako `stop_sequences` sú definované v suite, ale server
`stop` odmieta, preto sa neposielajú a fence sa strihá v evaluátore.

**T1 `short_instruction` [instruction]** — temp 0.3, max 256:
> Write a concise explanation of how a hash table works in 3-4 sentences.
Účel: základná inštrukcia + plynulosť; lacný (krátky).

**T2 `medium_reasoning` [reasoning]** — temp 0.3, max 768:
> You are given a sequence: 2, 6, 12, 20, 30, 42, 56. What is the next number
> in the sequence? Explain your reasoning step by step.
Účel: reťazec úvah + správna odpoveď (72); kľúčové slová vyššie.

**T3 `long_context` [context]** — temp 0.3, max 1024:
> Below is a document about renewable energy. Please read it carefully and
> answer the question at the end. DOCUMENT: [~350-slovný dokument o solárnej,
> veternej, vodnej, geotermálnej energii a biomase + úložiská, sieť, politiky,
> náklady, trendy] QUESTION: Summarize the main renewable energy sources,
> their key advantages, primary challenges, and two future trends mentioned
> in the document.
Účel: porozumenie dlhému vstupu + pokrytie kľúčových slov; najdrahší test
(1024 tokenov) — pri 1 tok/s modeloch ~17 min/opakovanie.

**T4 `coding_task` [coding]** — temp 0.1, max 768, stop `["```", "def ", "class "]`:
> Write a Python function `find_duplicates(nums: list[int]) -> list[int]`
> that returns all duplicate integers in a list. The function should:
> 1. Run in O(n) time complexity
> 2. Use O(1) extra space (excluding output)
> 3. Handle negative numbers
> 4. Return duplicates in ascending order
> Provide only the function definition with docstring.
Účel: kódová korektnosť; nízka teplota = determinizmus. (Server `stop`
odmieta, preto sa fence strihá v evaluátore.)

**T5 `structured_output` [format]** — temp 0.0, max 256, stop `["}"]`:
> Output a JSON object with the following structure exactly:
> { "name": "string", "age": integer, "skills": ["string", "string", "string"],
>   "address": { "city": "string", "country": "string" } }
> Use realistic data for a software engineer. No extra text, just the JSON.
Účel: striktný formát; teplota 0.0 = maximálna determinovanosť.

**Precision probe** (len `auto --precision`): T4+T2+`structured_output` pri
suite teplote vs 0.2, jeden load — informatívne, do skóre nevstupuje.

## 7. Ostatné behy (nastavenia)

- **smoke_test**: load + 1 chat ("Say hi in 5 words.", 30 tok., 0.3) + unload.
- **measure_throughput**: 1 load + N súbežných 32-tokenových chatov
  (`asyncio.gather`); agregát = tokeny / wall-clock.
- **matrix.run_one**: 1 load + 1×30-token chat (bunka ctx×flash×KV).
- **fit ladder / ctx sweep**: smoke na bod + `ctx` bisect refinement (krok
  500/1000, max 8 sond).
- **speculative.ab_compare**: opt-in, draft vs baseline na víťaznej konfigurácii.

## 8. Cenový model a páky pre pomalé modely

Cena kandidáta ≈ load + preheat + `repetitions` × Σ(max_tokens) / tok_s.
Pri 1 tok/s: T3 sám ~17 min × 3 opakovania ≈ 1 h/kandidát.
Páky: `--repetitions 1 --validation 1` (3–5× dole),
`--max-tokens-scale 0.5|0.25` (s rizikom truncation-failov, varovanie sa
vypíše), užší `--min/max-context`, `--dry-run` ukáže počet kandidátov vopred.
Report (`-best.md`) obsahuje každú tabuľku vyššie + per-test riadky s teplotou,
tokenmi, rýchlosťami a kvalitou.

## 9. Speed probe (Phase A) — proti čomu sa 5-testový suite nebeží

- Jeden prompt: "Explain in one or two short sentences what a hash table is.",
  max 64 tokenov, teplota 0.1, reasoning off, minimálny preheat.
- Žiadna quality evaluácia. Výsledok nesie `generation.phase = "SPEED"`,
  skóre ostáva None až po prejdenú kvalitu.
- Opakovania podľa budget triedy z prvého (S0) merania: FAST 3, NORMAL 2,
  SLOW 1, VERY_SLOW 1. Pomalé modely sa nikdy neodmietajú.
- Kontext: `speed_context` (user cap alebo 2048) pre všetky sondy;
  `quality_context` (user cap alebo min(limit, 8192)) pre finalistov.
  Kontext do Phase-A skóre nevstupuje (váha 0, renormalizácia zvyšku).
- Throughput sondy (parallel) merajú agregát do separátneho bloku
  `generation.throughput`; primárna metrika ostáva single-stream tok/s.
