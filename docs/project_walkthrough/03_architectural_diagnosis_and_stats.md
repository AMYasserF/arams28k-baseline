[← Previous: Chapter 02](02_benchmark_hygiene_audit.md) | [Master Index](00_INDEX.md) | [Next: Chapter 04 — Step 2 Kraken Calibration →](04_step2_kraken_zero_training_fix.md)

---

# Chapter 03 — Architectural Diagnosis & Statistical Hardening (Step 1B)

> **Chapter Takeaway:** Rigorous paired statistics (`10,000` bootstrap resamples, Wilcoxon tests, and Holm–Bonferroni correction) prove the **CER-vs-WER Paradox** is real across every script: **Kraken** wins Strict CER (`22.22%` vs. `25.22%`, $p = 2.20 \times 10^{-181}$) while **HATFormer** wins True Corpus WER (`56.68%` vs. `59.31%`, $p = 2.86 \times 10^{-20}$). Edit decompositions explain *why*: Kraken suffers CTC deletions in Ruq'ah (`40.0%` of its errors) and lacks a language model, whereas HATFormer suffers `+61.0%` spurious insertions in Maghrebi. After removing `page_034`, physical damage (`AraMS-Restore`) significantly elevates gradual CER ($p < 0.001$)—nearly doubling Kraken's lead over HATFormer—without triggering catastrophic collapses, while a Baseline Oracle (`19.89%` CER / `50.53%` WER) confirms high complementarity.

---

## 1. Why We Did This (Reason & Question)

Looking at Chapter 01's averages wasn't enough for a peer-reviewed paper. We needed to know:
1. **Is the CER-vs-WER Paradox statistically robust** across all three manuscript traditions, or could a handful of skewed long/short lines explain it?
2. **What is the exact physical mechanism** behind each model's errors (deletions vs. substitutions vs. insertions, and orthographic spelling variants)?
3. **Once the `page_034` confounder is removed**, how does real physical manuscript damage (`177` lines in `AraMS-Restore`) affect CNN+CTC vs. Vision Transformer attention?
4. **If we had an ideal selector (Oracle)** picking the better model per line, how low could baseline errors go?

---

## 2. What We Did & How (Step-by-Step + Code & File References)

Executed across [`EDA/comparative_eda.py`](../../EDA/comparative_eda.py) and [`EDA/paper1_stats_audit.py`](../../EDA/paper1_stats_audit.py), generating [`EDA/reports/report_4_kraken_vs_hatformer.md`](../../EDA/reports/report_4_kraken_vs_hatformer.md) and [`EDA/reports/paper1_statistical_tables.md`](../../EDA/reports/paper1_statistical_tables.md):

1. **Statistical Hardening (Plain-English Guide to Our 3 Tests):**
   - **Corpus-Weighted Paired Bootstrap (`n_boot = 10,000, seed = 42`):** Because character/word counts vary per line, standard analytical formulas for line-averaged CER CIs can mislead. So `10,000` times in a row, we sampled `6,744` paired `(Kraken, HATFormer)` lines with replacement and recomputed total corpus edits ÷ total reference length inside every resample. The middle `95%` of deltas gives the **95% CI** (Confidence Interval — range proving the difference isn't sampling luck).
   - **Two-Sided Paired Wilcoxon Signed-Rank Test:** HTR line error rates are heavily skewed (lots of lines near `5–15%`, a long tail above `50%`), violating the bell-curve assumption of a paired t-test. The non-parametric Wilcoxon test checks whether per-line error ranks consistently favor one model on the exact same lines.
   - **Holm–Bonferroni Correction (`m = 8` tests):** When checking `4` splits (`book_03`, `book_05`, `book_09`, `OVERALL`) × `2` metrics (`CER`, `True WER`), running `8` separate hypothesis tests increases false-alarm risk. Holm–Bonferroni adjusts p-values sequentially from smallest to largest to keep family-wise false-discovery probability strictly under `5%`.
2. **Character Edit Decomposition & Orthography Folding:** Enumerated exact Levenshtein `insert`, `delete`, and `replace` operations via `rapidfuzz.distance.Levenshtein.opcodes`, and recomputed CER after **Orthographic Normalization** (`أ/إ/آ→ا`, `ة→ه`, `ى→ي`).
3. **De-Confounded Physical Damage Audit (`AraMS-Restore`):** Excluded all `20` shifted lines of `book_09_page_034` from Naskh (`1,056 - 20 = 1,036`: `n_dam = 177` damaged vs. `n_clean = 859` undamaged), then ran **Mann–Whitney U tests** (unpaired rank-shift test for gradual CER increases), **two-sided Fisher's Exact tests** (checking if `CER > 50%` collapses happen disproportionately on damaged lines), and an analytic **80% statistical power** curve.
4. **Baseline Oracle & 2×2 Rescue Matrix:** Evaluated best-of-two per-line selection across all `6,744` test lines.

---

## 3. Results & Visual Evidence

![Baseline CER vs WER Paradox Chart](assets/chart_01_baseline_paradox.png)

### Finding 1 — Statistical Proof of the Baseline CER-vs-WER Paradox (`N = 6,744`)

From [`EDA/reports/paper1_statistical_tables.md`](../../EDA/reports/paper1_statistical_tables.md) ($\Delta = \text{HATFormer} - \text{Kraken}$, so negative $\Delta$ = HATFormer has lower error):

| Split / Book | Strict CER $\Delta$ `[95% CI]` | Ortho CER $\Delta$ `[95% CI]` | True Corpus WER $\Delta$ `[95% CI]` | Holm Adj $p$ (`CER`) | Holm Adj $p$ (`True WER`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `book_03` (Maghrebi) | **`+5.54 pp`** `[+5.20, +5.89]` | **`+5.05 pp`** `[+4.72, +5.38]` | **`-2.23 pp`** `[-2.89, -1.58]` | `5.256e-170` | `1.622e-10` |
| `book_05` (Ruq'ah) | **`+0.50 pp`** `[+0.06, +0.93]` | `+0.12 pp` `[-0.31, +0.54]` | **`-1.71 pp`** `[-2.61, -0.82]` | `2.952e-20` | `1.617e-01` *(NS rank)* |
| `book_09` (Naskh) | **`+1.62 pp`** `[+1.05, +2.20]` | **`+1.54 pp`** `[+0.98, +2.10]` | **`-5.12 pp`** `[-6.19, -4.06]` | `4.135e-11` | `4.385e-19` |
| **OVERALL (`6,744`)** | **`+2.99 pp`** `[+2.73, +3.25]` | **`+2.62 pp`** `[+2.37, +2.87]` | **`-2.63 pp`** `[-3.12, -2.16]` | **`2.204e-181`** | **`2.863e-20`** |

- Overall, **Kraken beats HATFormer on `4,112` lines (`61.0%`) by CER** (ties `542`, loses `2,090`), yet **HATFormer beats Kraken on Corpus WER in every single script** (`-2.63 pp` overall, bootstrap CI `[-3.12, -2.16]`). *(Note: on Ruq'ah `book_05`, Corpus WER favors HATFormer `-1.71 pp` via rescuing catastrophic long lines, while per-line Wilcoxon rank test is non-significant at `0.1617` because clean lines tie often.)*

### Finding 2 — Why They Trade Leads: Opposite Architectural Failure Modes
1. **Kraken CTC Blank Deletions vs. Linguistic Blindness:** In Ruq'ah (`book_05`), **`40.0%`** of all Kraken character edits are **deletions** (`6,538` missing characters vs. `4,210` in HATFormer — HATFormer cuts Ruq'ah deletions by **`-35.6%`** and rescues **`32 / 42` (`76.2%`)** of Kraken's catastrophic Ruq'ah lines). Without any language model, Kraken also splits or misspells intact Arabic roots (`59.31%` WER despite `22.22%` CER).
2. **HATFormer Maghrebi Insertions:** On dense Maghrebi calligraphy (`book_03`), HATFormer raw insertions surge **`+61.0%`** (`11,637 → 18,745`) as its cross-attention decoder guesses fluent but ungrounded Arabic phrases (`528` catastrophic lines vs. `199` for Kraken).
3. **Orthographic Sensitivity (`~20%` of Ruq'ah Error):** Folding non-semantic spelling variants drops Ruq'ah CER below `10%` for **both** models (`11.25% → 9.19%` for Kraken, `-18.3%` relative; `11.75% → 9.31%` for HATFormer, `-20.8%` relative). Notice from [`report_4`](../../EDA/reports/report_4_kraken_vs_hatformer.md) Section 9 that in Ruq'ah, **`أ → ا`** happens `991` times in HATFormer vs. `440` times in Kraken, showing RoBERTa's strong corpus spelling prior overrides rare hamzas!

### Finding 3 — De-Confounded `AraMS-Restore` Physical Damage (`book_09`, `n_dam=177` vs. `n_clean=859`)

![De-Confounded Naskh Damage Comparison](assets/chart_02_naskh_damage_deconfounded.png)

Excluding `book_09_page_034` reveals the clean effect of real physical stains and bleed-through:
- **Gradual CER Shifts Up Significantly:** Kraken Mean CER climbs `20.51% → 23.13%` (Median `19.28% → 21.92%`, two-sided Mann–Whitney U `p = 1.44e-04`); HATFormer Mean CER climbs `21.94% → 25.95%` (Median `20.83% → 25.35%`, `p = 1.00e-06`). Sensitivity check excluding blank crop `006514` (`n_clean=858`) stays equally significant (`1.28e-04` and `8.63e-07`).
- **Why Kraken Resists Stains Better:** Kraken's Corpus CER lead over HATFormer **nearly doubles** under physical degradation—from **`-1.43 pp`** on undamaged Naskh (`20.51%` vs. `21.94%`) to **`-2.76 pp` Corpus / `-2.81 pp` Mean** on physically damaged Naskh (`23.19%` vs. `25.95%`), because overlapping small `3×3` CNN convolutions tolerate localized ink stains better than `16×16` ViT patches.
- **No Catastrophic Surge:** Binary `>50% CER` collapses do **not** increase under damage (`1.13%` damaged vs. `1.05%` Kraken / `0.93%` HATFormer clean; Fisher `p = 1.000` / `0.683`, with `>80%` analytic power to detect surges `>= 4.4%`), proving polygon-masked line crops successfully isolate stains. Yet on those `177` damaged lines, HATFormer *still* won True WER (`56.45%` vs. `59.64%` Corpus WER, Wilcoxon `p = 0.0341`), confirming its LM prior helps reconstruct damaged words.

### Finding 4 — Baseline Oracle Ceiling (`N = 6,744`)
- Picking the better baseline prediction per line drops overall **Corpus CER to `19.89%`** (`28.89%` Maghrebi, **`8.55%` Ruq'ah**, `19.58%` Naskh), **CER-selected Corpus WER to `52.99%`**, **WER-selected Ceiling to `50.53%`**, and total catastrophic lines down from `272 / 572` to **`178`**.

---

## 4. Conclusion (What This Proves)

1. Both models fail in **complementary ways**: Kraken's CTC backbone preserves spatial accuracy and stain resilience (`22.22%` CER) but suffers from **missing an Arabic language model** (`59.31%` WER) and CTC deletion collapse in Ruq'ah (`40.0%` deletions). HATFormer captures fluent Arabic phrasing (`56.68%` WER) and saves Ruq'ah lines, at the cost of visual instability on dense Maghrebi (`36.39%` CER).
2. A baseline Oracle already breaks below `20%` CER (`19.89%`) and nearly halves Ruq'ah error (`8.55%`), proving huge upside if each model's weakness is fixed first.

---

## 5. How This Affects the Next Step

Before training expensive new neural architectures, Step 1B gave us two crystal-clear **zero-training decoding hypotheses** to test in **Step 2 (Tier 1A)**:
1. **Track B (Kraken):** What happens if we replace Kraken's greedy column picker with a **beam search + leak-free 5-gram Arabic character LM + blank/length calibration** (`bw`, `bp`, `lb`, `lm`)? That experiment is documented in [Chapter 04](04_step2_kraken_zero_training_fix.md).
2. **Track A (HATFormer):** What happens if we add targeted decoding guards (**Blank-Gate**, **Width-Cap**, **Repetition Penalty**, **Loop Fallback**) to curb HATFormer's Maghrebi hallucinations? That experiment is documented in [Chapter 05](05_step2_hatformer_saturation_and_visual_gap.md).

---

[← Previous: Chapter 02](02_benchmark_hygiene_audit.md) | [Master Index](00_INDEX.md) | [Next: Chapter 04 — Step 2 Kraken Calibration →](04_step2_kraken_zero_training_fix.md)
