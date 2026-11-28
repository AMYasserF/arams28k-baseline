[← Previous: Chapter 03](03_architectural_diagnosis_and_stats.md) | [Master Index](00_INDEX.md) | [Next: Chapter 05 — HATFormer Saturation →](05_step2_hatformer_saturation_and_visual_gap.md)

---

# Chapter 04 — Step 2 Track B: Kraken Zero-Training Calibration

> **Chapter Takeaway:** Without retraining a single neural weight, adding **CTC Prefix Beam Search (`beam_width=25`)**, a strictly **leak-free 5-gram Kneser–Ney character Language Model (`lm=0.45`)**, and blank/length calibration (`bp=0.20, lb=0.30`) tuned solely on validation (`N=1,486`) transforms Kraken: on the official `6,744`-line test split, **Strict CER drops `22.22% → 19.86%` (`-2.37 pp`)** and **True Corpus WER plunges `59.31% → 49.25%` (`-10.06 pp`)**. This achieves a **complete Paradox Reversal**: single-model Kraken Tier 1A now outperforms HATFormer (`01_plus_blank_gate`) on **both** CER and True Word-Level WER across **all 3 scripts** (all `8` Holm–Bonferroni tests significant at $p \le 6.71 \times 10^{-19}$), even beating the old Baseline Oracle!

---

## 1. Why We Did This (Reason & Question)

Chapter 03 diagnosed two specific bottlenecks holding Kraken back (`22.22%` CER, `59.31%` WER):
1. **Greedy Decoding & Linguistic Blindness:** Baseline Kraken picks only the top letter independently at every image frame, ignoring whether neighboring letters form a real Arabic word—creating `59.31%` WER despite `22.22%` CER.
2. **CTC Blank Dominance (`40%` deletions in Ruq'ah):** Because CTC uses blank separator tokens between characters, faint strokes or diacritics often get collapsed to blanks and deleted.

We asked: **How far can pure decoding-time calibration (beam search + statistical language model + blank/length tuning, zero neural retraining) push Kraken—and can it eliminate HATFormer's word-level WER lead?**

---

## 2. What We Did & How (Step-by-Step + Code & File References)

All Track B code and artifacts live in [`scripts/models/kraken/`](../../scripts/models/kraken/), [`kraken/`](../../kraken/), [`runs/tier1a_kraken_val_ext/`](../../runs/tier1a_kraken_val_ext/), and [`runs/tier1a_kraken_test/`](../../runs/tier1a_kraken_test/), summarized in [`report_5_kraken_ctc_tier1a_calibration.md`](../../EDA/reports/report_5_kraken_ctc_tier1a_calibration.md).

### Step A — Train a Leak-Free 5-Gram Kneser–Ney Character LM (`train_char_lm.py`)
- Trained via [`scripts/models/kraken/train_char_lm.py`](../../scripts/models/kraken/train_char_lm.py) in `3.37 seconds` strictly on **`split == 'train', line_type == 'main'`** (`19,739` lines, `1,187,789` NFC characters, `37`-character vocabulary, `175,766` unique 5-grams — zero validation or test data touched).
- **What is a 5-gram Kneser–Ney LM?** Given the previous `4` Arabic characters `(c_{t-4}, c_{t-3}, c_{t-2}, c_{t-1})`, it estimates the log-probability of the next character `c_t`, using **Modified Kneser–Ney smoothing** (back-off smoothing to 4-gram, 3-gram, 2-gram, and 1-gram frequencies when a rare 5-character spelling hasn't been seen before). Perplexity (uncertainty per character) reached `4.512` on train and **`7.4007`** on held-out validation (`N = 1,486`).

### Step B — Cache Raw Frame Logits & Run Prefix Beam Search (`tier1a_ctc_decode.py`)
- In [`scripts/models/kraken/tier1a_ctc_decode.py`](../../scripts/models/kraken/tier1a_ctc_decode.py), we ran Kraken's neural backbone (`best_0.8957.safetensors`) **once** per line on CPU with exact grayscale dewarping and saved the raw `(T, 174)` float16 log-probability matrices (`val`: `1,486` lines in `219.1s`; `test`: `6,744` lines in `702.0s` — cached as `.pt` files in `kraken/ctc_cache/`).
- **RTL Order Fix:** Kraken's raw image frames scan right-to-left geometrically; reversing the time axis (`lp[::-1]`) aligns frames with **logical right-to-left Arabic reading order** so our forward Arabic 5-gram LM scores real prefixes accurately!
- **CTC Prefix Beam Search (`beam_width = 25`):** Instead of picking the single best character per frame (greedy), beam search tracks the top `25` distinct merged character prefixes simultaneously and scores each candidate using three tunable calibration knobs:
  1. **`bp` (`blank_penalty`):** Subtracted from the CTC `blank` log-probability at every frame to prevent Kraken from deleting faint strokes.
  2. **`lb` (`length_bonus`):** Added every time a new non-blank character is appended, directly offsetting per-character LM length shrinkage.
  3. **`lm` (`lm_weight`):** Multiplier on the 5-gram Kneser–Ney Arabic spelling score $\ln P_{\text{LM}}(c \mid \text{history})$ added when emitting non-blank characters.

### Step C — Leak-Free Validation Grid Selection (`ValOpt`)
We swept `36` configurations in Stage 1 (`runs/tier1a_kraken_val`) and `24` extended configurations in Stage 2 (`runs/tier1a_kraken_val_ext`) purely on **`val` (`N=1,486`)**, locking in the single **Validation-Optimal config (`ValOpt`)** before opening test predictions: **`bw25_bp0.20_lb0.30_lm0.45`**.

---

## 3. Results & Visual Evidence

### Validation U-Curve Peak (`N = 1,486` — Leak-Free Selection)

![Kraken Validation U-Curve](assets/chart_03_kraken_val_ucurve.png)

On validation (`book_06 + book_11`):
- Greedy baseline starts at **`10.44%` Strict CER** (`7.58%` Ortho CER, **`38.66%` Corpus WER**).
- Pure beam search (`lm=0.00`) nudges CER slightly (`10.32%`).
- Adding the 5-gram LM creates a textbook **U-curve**: Strict CER falls cleanly through `lm=0.15` (`9.69%`), `lm=0.25` (`9.49%`), `lm=0.35` (`9.35%`), bottoming at **`lm=0.45`** with **`ValOpt` (`bw25_bp0.20_lb0.30_lm0.45`) = `9.30%` Strict CER (`-1.14 pp`)**, **`6.56%` Ortho CER**, and **`33.03%` Corpus WER (`-5.63 pp`)**.
- Pushing further to **`lm=0.55` regresses CER to `9.39%`** (catastrophic lines increase `12 → 13`), proving `lm=0.45` is a genuine non-overfit optimum.

### Official Test Generalization & Paradox Reversal (`N = 6,744` & Clean `N = 6,673`)

![Paradox Reversal Chart](assets/chart_04_paradox_reversal.png)

Evaluating the locked **`ValOpt`** setting on the unseen test set (verified in [`paper1_step2_tier1a_tables.md`](../../EDA/reports/paper1_step2_tier1a_tables.md)):

1. **Pair A — Kraken Gain (Greedy → `ValOpt Tier 1A`, Bootstrap `n=10,000, seed=42`):**

| Split / Script | Strict CER (`Greedy → T1A`) | $\Delta$ Strict CER `[95% CI]` | Ortho CER (`Greedy → T1A`) | Corp WER (`Greedy → T1A`) | $\Delta$ Corp WER `[95% CI]` | Holm Adj $p$ (`WER`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` (Maghrebi) | `30.85% → 27.23%` | **`-3.62 pp`** `[-3.76, -3.47]` | `28.76% → 25.34%` | `75.61% → 62.51%` | **`-13.09 pp`** `[-13.48, -12.69]` | `0.00e+00` (✓) |
| `book_05` (Ruq'ah) | `11.25% → 10.43%` | **`-0.83 pp`** `[-0.95, -0.70]` | `9.19% → 8.66%` | `38.28% → 31.98%` | **`-6.30 pp`** `[-6.70, -5.91]` | `2.21e-139` (✓) |
| `book_09` (Naskh) | `22.11% → 19.84%` | **`-2.27 pp`** `[-2.47, -2.08]` | `19.76% → 17.80%` | `58.30% → 48.69%` | **`-9.61 pp`** `[-10.16, -9.07]` | `6.21e-121` (✓) |
| **OVERALL (`N=6,744`)** | **`22.22% → 19.86%`** | **`-2.37 pp`** `[-2.46, -2.28]` | **`20.10% → 17.98%`** | **`59.31% → 49.25%`** | **`-10.06 pp`** `[-10.32, -9.80]` | **`0.00e+00`** (✓) |
| **Hygiene (`N=6,673`)** | **`21.88% → 19.52%`** | **`-2.36 pp`** | **`19.75% → 17.64%`** | **`59.01% → 48.94%`** | **`-10.07 pp`** | — |

   - **Overall Per-Line Win/Tie/Loss vs. Greedy:** **`4,467 / 1,229 / 1,048`** on CER; **`4,550 / 1,825 / 369`** on True WER; overall catastrophic failures drop **`272 → 242` (`-30` lines)**, and **Mean WER** drops `-10.52 pp` (`63.06% → 52.54%`). *(Reference `TestBest bp=0.40, lb=0.30, lm=0.45` achieves `19.84%` CER / `19.50%` Clean — matching `ValOpt` within `0.02 pp`.)*

2. **Pair B — Complete Paradox Reversal (`Kraken Tier 1A ValOpt` vs. `HATFormer 01_plus_blank_gate`):**

| Split / Script | Kraken T1A vs. HAT `01-BG` (Strict CER) | $\Delta$ Strict CER `[95% CI]` | $\Delta$ Ortho CER `[95% CI]` | Kraken T1A vs. HAT `01-BG` (Corp WER) | $\Delta$ Corp WER `[95% CI]` | Holm Adj $p$ (`CER` / `WER`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` (Maghrebi) | `27.23%` vs. `36.39%` | **`-9.16 pp`** `[-9.53, -8.80]` | **`-8.47 pp`** `[-8.82, -8.13]` | `62.51%` vs. `73.38%` | **`-10.86 pp`** `[-11.52, -10.19]` | `0.00e+00` / `4.31e-180` |
| `book_05` (Ruq'ah) | `10.43%` vs. `11.75%` | **`-1.33 pp`** `[-1.81, -0.83]` | **`-0.65 pp`** `[-1.12, -0.17]` | `31.98%` vs. `36.56%` | **`-4.58 pp`** `[-5.47, -3.69]` | `2.51e-57` / `6.36e-37` |
| `book_09` (Naskh) | `19.84%` vs. `23.73%` | **`-3.89 pp`** `[-4.50, -3.29]` | **`-3.50 pp`** `[-4.11, -2.91]` | `48.69%` vs. `53.18%` | **`-4.49 pp`** `[-5.57, -3.43]` | `9.52e-48` / `6.71e-19` |
| **OVERALL (`N=6,744`)** | **`19.86%`** vs. `25.22%` | **`-5.36 pp`** `[-5.64, -5.08]` | **`-4.74 pp`** `[-5.01, -4.46]` | **`49.25%`** vs. `56.67%` | **`-7.43 pp`** `[-7.91, -6.93]` | **`0.00e+00`** / **`8.68e-223`** |

   - Single-model Kraken Tier 1A (`19.86%` CER / `49.25%` WER) **surpasses even the two-model Baseline Oracle (`19.89%` CER / `50.53%` WER) from Chapter 03**!

### Qualitative Recovery Examples (`ValOpt` vs. Greedy Kraken)
- **Maghrebi (`book_03_line_003538` — CER `29.1% → 10.9%`, WER `72.7% → 36.4%`):**  
  GT: `لكم إلى قوله من الفجر حجة لفقهاء الأمصار ومن ذهب مذهبهم`  
  Greedy: `لثع إلى قوله فن العجر فجنة اعفهاه الأوصار ومن ذكعب قدبهم`  
  Tier 1A: `لكم إلى قوله من الفجر فجة الفقهاء الأوصار ومن ذهب قلبهم` *(fixes broken function words and prefixes cleanly)*
- **Ruq'ah Exact Recovery (`book_05_line_004221` — CER `9.2% → 0.0%`, WER `40.0% → 0.0%`):**  
  GT: `والفتوى والقضاء وإماطة الأذى عن الطريق والنصيحة والإعانة على البر`  
  Greedy: `والفتوى والقضاء وأماطه الأذى عن الطريقة والنصيحة والإاعاتة على البرة`  
  Tier 1A: `والفتوى والقضاء وإماطة الأذى عن الطريق والنصيحة والإعانة على البر`

### Honest Diagnostic Nuance — Why Ruq'ah & Naskh Catastrophic Counts Rose Slightly
Looking carefully at per-book counts reveals an important trade-off documented in [`report_5`](../../EDA/reports/report_5_kraken_ctc_tier1a_calibration.md):
- In **Maghrebi**, catastrophic lines dropped dramatically (`199 → 147`, rescuing `70` lines).
- In **Ruq'ah (`42 → 56`)** and **Naskh (`31 → 39`)**, catastrophic counts ticked up slightly (`+15` newly catastrophic lines in Ruq'ah, `+8` in Naskh).
- **Why?** Because every character step adds an average uncentered 5-gram LM term ($\mathbb{E}[\ln P_{\text{LM}}] \approx \ln(1/7.40) \approx -2.00\text{ nats/char}$ on `val`). Even with `lb=0.30` and `bp=0.20` recovering `863` deletions compared to uncompensated LM decoding (`8,147 → 7,284` deletions in Ruq'ah at `bp=0.40`), dense weak-ink Ruq'ah lines still suffer slightly higher raw deletions than greedy (`6,538 → 7,337`), pushing `15` borderline ~`48–49%` Ruq'ah lines across the `50%` threshold (for instance `book_05_line_003897` going `49.4% → 60.5%`, which we visualize in Chapter 06).
- **Crucial Cross-Model Rescue Check:** Verifying against paired test predictions shows HATFormer (`01_plus_blank_gate`) scores `<= 50% CER` (and strictly beats Kraken Tier 1A) on **`15 / 15` (`100%`)** of those newly catastrophic Ruq'ah lines and **`8 / 8` (`100%`)** of newly catastrophic Naskh lines!

---

## 4. Conclusion (What This Proves)

1. The Baseline CER-vs-WER Paradox was **mostly Kraken's missing Arabic language model**, NOT a permanent ceiling of CTC recognition: adding a leak-free 5-gram LM + prefix beam search cuts overall CER `-2.37 pp` and Corpus WER **`-10.06 pp`**, winning on both metrics across every script.
2. The remaining trade-off (`15` newly truncated Ruq'ah lines under `lm=0.45`) is small, structured, and **`100%` recoverable** by pairing with HATFormer.

---

## 5. How This Affects the Next Step

Kraken passed its zero-training upgrade test with flying colors. The next natural question in **Step 2 (Track A)** was: **Can zero-training decoding guards (`Blank-Gate`, `Width-Cap`, `Repetition Penalty`, `Length Penalty`, `Loop Fallback`) deliver a similar breakthrough for HATFormer—especially on its `528` catastrophic Maghrebi lines?** That investigation is documented in [Chapter 05](05_step2_hatformer_saturation_and_visual_gap.md).

---

[← Previous: Chapter 03](03_architectural_diagnosis_and_stats.md) | [Master Index](00_INDEX.md) | [Next: Chapter 05 — HATFormer Saturation →](05_step2_hatformer_saturation_and_visual_gap.md)
