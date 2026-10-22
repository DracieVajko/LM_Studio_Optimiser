# Merania @16K — poznámka o kontaminácii

**Dátum:** 2026-10-22

## ⚠️ Dôležité: merania vykonané počas záťaže v pozadí

Posledné merania (swift Q2_K, Q3_K_M, a čiastočne ostatné) boli vykonané počas:

- **ffmpeg na CPU** (video processing)
- **AI upscale na RTX 3060** (GPU záťaž)

To znamená, že **namerané tok/s sú podhodnotené** oproti čistému stavu. Čísla
sú informatívne pre **relatívne porovnanie** (všetky merania v tej istej
záťaži), ale **nie sú absolútne** — čistý stav by dal vyššie hodnoty.

Toto je presne prípad, ktorý špecifikácia nazýva **kontaminácia merania**
(§11.4: "thermal/power contamination") — výsledok sa musí označiť, nie
prezentovať ako čistý.

## Ako to ovplyvňuje rozhodnutia

| Rozhodnutie | Dopad |
|---|---|
| **Relatívne poradie modelov** | ✅ platí — všetky merania v rovnakej záťaži |
| **Absolútne tok/s** | ❌ podhodnotené — čistý stav dá viac |
| **Max context / kvalita** | ✅ neovplyvnené (nie je to speed metrika) |
| **Keep/delete rozhodnutie** | ✅ platí — rozdiely sú rádové (1.5 vs 9 vs 27 tok/s), nie percentá |

## Pre čisté merania

Ak budeš chcieť presné čísla:

1. vypni ffmpeg a AI upscale
2. `python tools/speed16k_newmodels.py` (alebo konkrétny model)
3. `python tools/final_report.py --context 16384`

Rozdiel sa prejaví najmä pri modeloch, ktoré sú **blízko hranice** (napr.
swift Q3 vs Q2) — tam môže záťaž zmeniť poradie. Pri modeloch s rádovým
rozdielom (bonsai 27 vs qwen 9b 9 vs swift 2) je poradie stabilné.

## Live validácia (prebieha)

Live validácia na **bonsai** (najrýchlejší model, 27 tok/s) testuje **či
workflowy fungujú**, nie presné čísla. Kontaminácia tu nevadí — ide o
funkčnosť, nie o benchmark.
