# Plán: Swift q2_k vs q4_k_s vs referenčné modely

Dátum: 2026-10-26
HW: RTX 3060 Laptop 6 GB VRAM, 31 GB RAM, LM Studio s KV cache + 8 threads
Cieľ: porovnať kvalitu (nie len rýchlosť) dvoch Swift K-quantov proti referenčnému modelu.

## Prečo tento plán

Doterajšie merania sú speed-only. Reasoning sweepy ukázali rýchlosť po leveloch, ale
quality bolo `1.000` s `0/0 checks` (heuristika nemala pravidlá pre reasoning varianty).
To je od 2026-10-26 opravené: reasoning levely teraz používajú kritériá základného
promptu a neznáme testy už nedostanú falošných 1.000. Tento plán preto meria
**skutočné checks**, nie len tok/s.

## Modely a roly

| Rola | Model | Prečo |
|---|---|---|
| Kandidát A | `swift-qwen3.8-27b@q2_k` | najrýchlejší Swift (3.3 tok/s @4K) |
| Kandidát B | `swift-qwen3.8-27b@q4_k_s` | vyššia kvantizácia, 2.7 tok/s @4K |
| Referencia (rýchla) | `qwen/qwen3-coder-30b` | 12.3 tok/s @4K, coding-heavy |
| Referencia (univerzálna) | `openai-gpt-oss-20b-...-heretic` | 10.5–11.3 tok/s |

q3_k_m je vyradený (najpomalší, 1.9–2.0 tok/s).

## Metodika

### Fixné podmienky
- Jeden model naraz (job lock + host guard).
- `--max-context 8192` pre všetky behy (quality context = 8192; speed context = 4096).
- `--preset quick` = 1 rep v kvalite, 2 finalisti, cap 12 probes.
- Žiadny sandbox, žiadny context sweep.
- Pred každým behom over `check_loaded6.py` = nič načítané.

### Fáza 1 — quality bateria (hlavné porovnanie)
Pre každý model:

```
python -m lm_optimizer optimize <model> \
  --preset quick --max-context 8192 \
  --yes --non-interactive
```

Toto pustí speed (cap 12) → quality (5 kategórií, 1 rep) → validation (2 reps).
Zber: `overall`, `checks_passed/checks_total`, gen tok/s, TTFT, VRAM, status.

Odhad: coder ~40 min, gpt-oss ~45 min, q2 ~3 h, q4 ~3.5 h.

### Fáza 2 — reasoning s reálnymi checks (voliteľné, ak Fáza 1 nestačí)
Pre Swift varianty znovu reasoning sweep s opraveným evaluatorom:

```
python -m lm_optimizer optimize swift-qwen3.8-27b@q2_k \
  --reasoning-only --max-context 8192 --yes --non-interactive
```

(analogicky q4_k_s). Tentoraz musí mať každý level `checks_total > 0`.

### Fáza 3 — vyhodnotenie
Víťaz = najvyšší `overall` pri splnení:
1. speed floor 2.0 tok/s (median generation),
2. `checks_total > 0` (nesmieme porovnávať neoverené),
3. stability ≥ 0.85.

Tie-break: vyšší gen tok/s. Ak referenčný model vyhrá, Swift varianty idú na DELETE.

## Kritériá úspechu
- Každý beh má `resolved_settings`, `speed_floor` blok a `quality.checks_total > 0`.
- Žiadny beh neskončí `load_failed` viac ako raz (inak rerun).
- Reporty v `results/` obsahujú per-config tabuľku a quality checks.

## Riziká
- Heuristická kvalita je len proxy (keyword/JSON/štruktúra), nie MMLU. Neprezentovať ako „IQ".
- Pri 2–3 tok/s trvá quality bateria hodiny; teplo môže skresliť poradie — preto fixné poradie a jedna opakovacia sada.
- Iné agent-session na stroji môžu načítať coder-30b; watchdog ho vyhostí, ale probe môže padnúť → taký beh zopakovať.
- q2 vs q4 rýchlostné poradie je na tomto HW obrátené oproti bežným očakávaniam; dôvod je spill do RAM, nie kvantizácia.

## Výstup
- `results/*.md` reporty pre 4 behy.
- `docs/EXAMPLE_RESULTS.md` doplnený o tabuľku Fázy 1.
- Finálne KEEP/DELETE rozhodnutie s odkazom na konkrétne run ID.
