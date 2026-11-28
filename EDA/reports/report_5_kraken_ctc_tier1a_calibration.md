# Report 5 — Tier 1A Track B: Zero-Training Kraken CTC Calibration (`AraMS-28k`)

**Project:** AraMS-28k Benchmark Audit & Zero-Training Fix (Paper 1, Path B)  
**Model:** Fine-tuned Kraken Epoch 51 ([best_0.8957.safetensors](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/kraken/best_0.8957.safetensors)) + Leak-Free 5-gram Kneser-Ney Char LM ([char_lm_5gram_train.pkl](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/kraken/char_lm_5gram_train.pkl))  
**Code:** [train_char_lm.py](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/scripts/models/kraken/train_char_lm.py), [tier1a_ctc_decode.py](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/scripts/models/kraken/tier1a_ctc_decode.py)  
**Run Outputs:** [runs/tier1a_kraken_val](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/runs/tier1a_kraken_val/ctc_sweep_summary.md), [runs/tier1a_kraken_val_ext](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/runs/tier1a_kraken_val_ext/ctc_sweep_summary.md), [runs/tier1a_kraken_test](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/runs/tier1a_kraken_test/ctc_sweep_summary.md)  

---

## Executive Summary

Zero-training CTC prefix beam search (`beam_width=25`) augmented with a **strictly leak-free 5-gram Kneser-Ney character language model** (trained solely on `split == 'train'`, $N = 19,739$ lines) and **compensatory length/blank calibration** produces massive, highly significant gains across every evaluation stratum—without modifying a single neural weight:

1. **Val-Selected Configuration Generalizes Cleanly (`N = 6,744` Official Test):**
   - Using the validation-optimal setting (`bw25_bp0.20_lb0.30_lm0.45`, selected purely on `val` $N = 1,486$ where Strict CER dropped `10.44% -> 9.30%`), **test Strict Corpus CER drops from `22.22%` to `19.86%`** (`-2.37 pp`, 95% paired bootstrap CI `[-2.46, -2.27]`, $p \approx 0$), breaking the 20% CER barrier.
   - **Ortho-Folded Corpus CER drops `20.10% -> 17.98%`** (`-2.12 pp`).
   - **True Corpus WER plunges by `-10.06 pp` (`59.31% -> 49.25%`**, 95% CI `[-10.32, -9.80]`), winning `4,550` lines vs losing only `369` (`1,825` ties) against Kraken greedy.
   - **Catastrophic failures (>50% CER)** drop from **`272` (`4.03%`) to `242` (`3.59%`)** (`236` at `bp=0.00_lb0.30_lm0.45`).
2. **Complete Reversal of the CER-vs-WER Paradox:**
   - In [Report 4](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/EDA/reports/report_4_kraken_vs_hatformer.md), Kraken beat HATFormer (`beams=3`) on CER (`22.22%` vs `25.22%`), while HATFormer beat Kraken on True Word-Level WER (`56.68%` vs `59.31%`).
   - With Tier 1A CTC calibration (`bw25_bp0.20_lb0.30_lm0.45`), **Kraken now outperforms HATFormer (`beams=3`) simultaneously on BOTH CER (`19.86%` vs `25.22%`, $\Delta = -5.36\text{ pp}$) AND True Word-Level WER (`49.25%` vs `56.68%`, $\Delta = -7.43\text{ pp}$, CI `[-7.92, -6.94]`, Wilcoxon $p = 9.06 \times 10^{-224}$)**.
   - Single-model Tier 1A Kraken (`19.84%–19.86%` CER, `49.24%–49.25%` Corpus WER) surpasses even the original Kraken+HATFormer Oracle Ensemble ceilings reported in Report 4 (`19.89%` CER, `50.53%` WER-selected ceiling).
3. **New Oracle Ensemble Ceiling (`Tier 1A Kraken + HATFormer Beams=3`):**
   - Pairing `ValOpt` Tier 1A Kraken with HATFormer (`beams=3`) pushes the theoretical Oracle Ensemble down to **`18.02%` CER**, **`46.02%` CER-selected Corpus WER / `44.94%` WER-selected Ceiling**, and **`145` Catastrophic lines**.
4. **Hygiene-Deduplicated Clean Stratum (`N = 6,673`):**
   - Removing the 29 audit artifacts and 42 duplicate folio lines yields **`19.52%` Strict CER**, **`17.64%` Ortho CER**, and **`48.94%` True Corpus WER** (`ValOpt`; `19.50%` CER / `17.62%` Ortho / `48.94%` WER for `TestBest bp=0.40_lb0.30_lm0.45`).

---

## 1. Experimental Protocol & Leak-Free Components

1. **Acoustic Model & Feature Extraction:**
   - Checkpoint: `kraken/best_0.8957.safetensors` loaded with exact `convert("L")` grayscale preprocessing, `CenterNormalizer` dewarping (`valid_norm=True`), `(16, 0)` padding, and `bidi_reordering=False` raw frame extraction.
   - Frame-level log-probabilities `(T, 174)` were extracted once on CPU (`val`: $1,486$ lines in $219.1\text{ s}$; `test`: $6,744$ lines in $702.0\text{ s}$) and cached as float16 matrices (`ctc_logprobs_{split}_best_0.8957.pt`).
   - During CTC prefix beam search, the frame axis is reversed (`lp[::-1]`) so prefixes extend left-to-right in natural logical RTL Arabic order.
   - Local greedy baseline parity vs [EDA/kraken_pred](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/EDA/kraken_pred): exact line match `6,588 / 6,744` (`97.7%`; aggregate parity `22.23%` vs `22.22%` CER, `20.10%` vs `20.10%` Ortho CER, `59.30%` vs `59.31%` Corpus WER, `6,536` vs `6,538` Ruq'ah raw deletions via `LevDist`; minor floating-point/library differences on 156 lines do not alter aggregate rates).
2. **Leak-Free 5-Gram Character Language Model:**
   - Architecture: Interpolated Modified Kneser-Ney char 5-gram ([char_lm_5gram_train.json](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/kraken/char_lm_5gram_train.json)).
   - Training corpus: **Strictly `split == 'train'`, `line_type == 'main'`** ($19,739$ lines; $1,187,789$ NFC characters; $37$ character vocabulary; $175,766$ unique 5-grams; fit time $3.37\text{ s}$). Zero exposure to `val` or `test`.
   - Diagnostic Perplexity: `4.512` (`train` 2,000-line sample) / **`7.401`** (held-out `val` $N = 1,486$).
3. **Hyperparameter Search Protocol (No Test Tuning):**
   - **Stage 1 (`val`, $N=1,486$, 36 configs):** `blank_penalty` $\in \{0.0, 0.2, 0.4, 0.6\}$, `length_bonus` $\in \{0.0, 0.05, 0.10\}$, `lm_weight` $\in \{0.0, 0.15, 0.25\}$.
   - **Stage 2 (`val_ext`, $N=1,486$, 24 configs):** Because `val` CER/WER improved monotonically up to `lm_weight=0.25, lb=0.10`, an extended validation sweep explored `blank_penalty` $\in \{0.2, 0.4\}$, `length_bonus` $\in \{0.1, 0.2, 0.3\}$, `lm_weight` $\in \{0.25, 0.35, 0.45, 0.55\}$. Performance peaked cleanly at **`lm_weight=0.45, length_bonus=0.30, blank_penalty=0.20`** (`9.30%` Strict CER), while `lm_weight=0.55` degraded CER and increased catastrophic lines.
   - **Stage 3 (`test`, $N=6,744$, 18 configs):** Evaluated representative parameter combinations (`blank_penalty` $\in \{0.0, 0.2, 0.4\}$, `length_bonus` $\in \{0.0, 0.3\}$, `lm_weight` $\in \{0.0, 0.25, 0.45\}$) using `ValOpt` as the primary zero-leakage point estimate alongside full ablation visibility.

---

## 2. Validation Set Calibration Sweeps (`split = val`, $N = 1,486$: `book_06` & `book_11`)

- **Greedy Baseline (`val`):** Strict CER = **`10.44%`** | Ortho CER = **`7.58%`** | Corpus WER = **`38.66%`**.
- **Best Initial Grid Config (`bw25_bp0.20_lb0.10_lm0.25`):** Strict CER = **`9.52%`** (`-0.92 pp`) | Ortho CER = **`6.75%`** | Corpus WER = **`34.24%`** (`-4.42 pp`) | Cat = `11` (`0.74%`).
- **Val-Optimal Extended Config (`bw25_bp0.20_lb0.30_lm0.45`, `ValOpt`):** Strict CER = **`9.30%`** (`-1.14 pp`) | Ortho CER = **`6.56%`** | Corpus WER = **`33.03%`** (`-5.63 pp`) | Mean WER = **`34.16%`** | Cat = `12` (`0.81%`) | `book_06` `11.16%`, `book_11` `6.80%`.
- **Peak WER Extended Config (`bw25_bp0.40_lb0.20_lm0.55`):** Strict CER = `9.38%` (regresses vs `lm=0.45`) | Corpus WER = `32.99%` | Cat = `13` (`0.87%`).

<details><summary>Full Stage 1 Validation Sweep Table (36 configs, runs/tier1a_kraken_val)</summary>

| Config | Beam | BlankPen | LenBonus | LM_Wt | Strict CER | Ortho CER | Corpus WER | Mean WER | Cat (>50%) | `book_06` CER (Del) | `book_11` CER (Del) | Time(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bw25_bp0.00_lb0.00_lm0.00` | 25 | 0.00 | 0.00 | 0.00 | 10.32% | 7.46% | 38.39% | 39.51% | 10 (0.67%) | 12.13% (1260) | 7.88% (949) | 36.7 |
| `bw25_bp0.00_lb0.00_lm0.15` | 25 | 0.00 | 0.00 | 0.15 | 9.72% | 6.93% | 35.35% | 36.54% | 11 (0.74%) | 11.54% (1374) | 7.29% (976) | 24.8 |
| `bw25_bp0.00_lb0.00_lm0.25` | 25 | 0.00 | 0.00 | 0.25 | 9.57% | 6.81% | 34.42% | 35.55% | 11 (0.74%) | 11.39% (1458) | 7.14% (996) | 27.8 |
| `bw25_bp0.00_lb0.05_lm0.00` | 25 | 0.00 | 0.05 | 0.00 | 10.31% | 7.45% | 38.43% | 39.56% | 10 (0.67%) | 12.14% (1247) | 7.85% (933) | 17.2 |
| `bw25_bp0.00_lb0.05_lm0.15` | 25 | 0.00 | 0.05 | 0.15 | 9.69% | 6.90% | 35.30% | 36.49% | 10 (0.67%) | 11.52% (1360) | 7.24% (956) | 24.0 |
| `bw25_bp0.00_lb0.05_lm0.25` | 25 | 0.00 | 0.05 | 0.25 | 9.55% | 6.79% | 34.38% | 35.50% | 11 (0.74%) | 11.37% (1428) | 7.12% (984) | 24.2 |
| `bw25_bp0.00_lb0.10_lm0.00` | 25 | 0.00 | 0.10 | 0.00 | 10.31% | 7.45% | 38.41% | 39.54% | 10 (0.67%) | 12.15% (1232) | 7.85% (915) | 18.6 |
| `bw25_bp0.00_lb0.10_lm0.15` | 25 | 0.00 | 0.10 | 0.15 | 9.68% | 6.89% | 35.25% | 36.45% | 11 (0.74%) | 11.52% (1340) | 7.21% (939) | 24.5 |
| `bw25_bp0.00_lb0.10_lm0.25` | 25 | 0.00 | 0.10 | 0.25 | 9.53% | 6.77% | 34.29% | 35.42% | 11 (0.74%) | 11.36% (1411) | 7.09% (968) | 26.4 |
| `bw25_bp0.20_lb0.00_lm0.00` | 25 | 0.20 | 0.00 | 0.00 | 10.32% | 7.46% | 38.39% | 39.51% | 10 (0.67%) | 12.14% (1229) | 7.87% (931) | 15.7 |
| `bw25_bp0.20_lb0.00_lm0.15` | 25 | 0.20 | 0.00 | 0.15 | 9.70% | 6.91% | 35.30% | 36.49% | 11 (0.74%) | 11.54% (1340) | 7.24% (948) | 23.6 |
| `bw25_bp0.20_lb0.00_lm0.25` | 25 | 0.20 | 0.00 | 0.25 | 9.54% | 6.78% | 34.31% | 35.44% | 11 (0.74%) | 11.36% (1417) | 7.10% (974) | 24.6 |
| `bw25_bp0.20_lb0.05_lm0.00` | 25 | 0.20 | 0.05 | 0.00 | 10.32% | 7.46% | 38.43% | 39.56% | 10 (0.67%) | 12.15% (1220) | 7.87% (917) | 17.0 |
| `bw25_bp0.20_lb0.05_lm0.15` | 25 | 0.20 | 0.05 | 0.15 | 9.69% | 6.89% | 35.23% | 36.42% | 11 (0.74%) | 11.52% (1319) | 7.22% (933) | 26.0 |
| `bw25_bp0.20_lb0.05_lm0.25` | 25 | 0.20 | 0.05 | 0.25 | 9.53% | 6.76% | 34.29% | 35.42% | 11 (0.74%) | 11.35% (1392) | 7.09% (960) | 27.9 |
| `bw25_bp0.20_lb0.10_lm0.00` | 25 | 0.20 | 0.10 | 0.00 | 10.32% | 7.46% | 38.41% | 39.54% | 10 (0.67%) | 12.14% (1195) | 7.87% (902) | 17.0 |
| `bw25_bp0.20_lb0.10_lm0.15` | 25 | 0.20 | 0.10 | 0.15 | 9.69% | 6.90% | 35.21% | 36.42% | 10 (0.67%) | 11.53% (1292) | 7.22% (926) | 25.9 |
| **`bw25_bp0.20_lb0.10_lm0.25`** | 25 | 0.20 | 0.10 | 0.25 | **9.52%** | **6.75%** | **34.24%** | **35.38%** | 11 (0.74%) | 11.34% (1368) | 7.08% (953) | 24.9 |
| `bw25_bp0.40_lb0.00_lm0.00` | 25 | 0.40 | 0.00 | 0.00 | 10.34% | 7.48% | 38.44% | 39.56% | 10 (0.67%) | 12.15% (1192) | 7.91% (921) | 16.1 |
| `bw25_bp0.40_lb0.00_lm0.15` | 25 | 0.40 | 0.00 | 0.15 | 9.70% | 6.91% | 35.23% | 36.42% | 11 (0.74%) | 11.54% (1307) | 7.23% (932) | 24.7 |
| `bw25_bp0.40_lb0.00_lm0.25` | 25 | 0.40 | 0.00 | 0.25 | 9.54% | 6.77% | 34.30% | 35.43% | 11 (0.74%) | 11.36% (1385) | 7.09% (958) | 23.4 |
| `bw25_bp0.40_lb0.05_lm0.00` | 25 | 0.40 | 0.05 | 0.00 | 10.33% | 7.48% | 38.46% | 39.59% | 10 (0.67%) | 12.15% (1183) | 7.89% (903) | 14.6 |
| `bw25_bp0.40_lb0.05_lm0.15` | 25 | 0.40 | 0.05 | 0.15 | 9.70% | 6.90% | 35.24% | 36.43% | 12 (0.81%) | 11.54% (1287) | 7.22% (920) | 23.8 |
| `bw25_bp0.40_lb0.05_lm0.25` | 25 | 0.40 | 0.05 | 0.25 | 9.52% | 6.76% | 34.27% | 35.40% | 11 (0.74%) | 11.34% (1364) | 7.08% (951) | 25.2 |
| `bw25_bp0.40_lb0.10_lm0.00` | 25 | 0.40 | 0.10 | 0.00 | 10.33% | 7.48% | 38.46% | 39.60% | 10 (0.67%) | 12.16% (1162) | 7.89% (884) | 16.2 |
| `bw25_bp0.40_lb0.10_lm0.15` | 25 | 0.40 | 0.10 | 0.15 | 9.70% | 6.91% | 35.25% | 36.45% | 11 (0.74%) | 11.54% (1271) | 7.22% (911) | 23.6 |
| `bw25_bp0.40_lb0.10_lm0.25` | 25 | 0.40 | 0.10 | 0.25 | 9.53% | 6.76% | 34.29% | 35.41% | 11 (0.74%) | 11.35% (1345) | 7.08% (943) | 23.7 |
| `bw25_bp0.60_lb0.00_lm0.00` | 25 | 0.60 | 0.00 | 0.00 | 10.35% | 7.49% | 38.45% | 39.59% | 10 (0.67%) | 12.17% (1174) | 7.90% (899) | 15.9 |
| `bw25_bp0.60_lb0.00_lm0.15` | 25 | 0.60 | 0.00 | 0.15 | 9.73% | 6.94% | 35.28% | 36.48% | 12 (0.81%) | 11.57% (1285) | 7.25% (924) | 25.1 |
| `bw25_bp0.60_lb0.00_lm0.25` | 25 | 0.60 | 0.00 | 0.25 | 9.53% | 6.77% | 34.30% | 35.42% | 11 (0.74%) | 11.36% (1357) | 7.09% (949) | 24.8 |
| `bw25_bp0.60_lb0.05_lm0.00` | 25 | 0.60 | 0.05 | 0.00 | 10.35% | 7.49% | 38.45% | 39.53% | 10 (0.67%) | 12.18% (1157) | 7.90% (883) | 17.6 |
| `bw25_bp0.60_lb0.05_lm0.15` | 25 | 0.60 | 0.05 | 0.15 | 9.71% | 6.92% | 35.27% | 36.41% | 12 (0.81%) | 11.56% (1262) | 7.24% (906) | 27.9 |
| `bw25_bp0.60_lb0.05_lm0.25` | 25 | 0.60 | 0.05 | 0.25 | 9.53% | 6.76% | 34.28% | 35.40% | 11 (0.74%) | 11.36% (1331) | 7.08% (939) | 25.9 |
| `bw25_bp0.60_lb0.10_lm0.00` | 25 | 0.60 | 0.10 | 0.00 | 10.36% | 7.50% | 38.48% | 39.55% | 10 (0.67%) | 12.20% (1142) | 7.90% (866) | 17.2 |
| `bw25_bp0.60_lb0.10_lm0.15` | 25 | 0.60 | 0.10 | 0.15 | 9.70% | 6.92% | 35.27% | 36.42% | 11 (0.74%) | 11.55% (1244) | 7.23% (896) | 25.5 |
| `bw25_bp0.60_lb0.10_lm0.25` | 25 | 0.60 | 0.10 | 0.25 | 9.52% | 6.75% | 34.28% | 35.34% | 11 (0.74%) | 11.36% (1319) | 7.06% (926) | 26.8 |

</details>

<details><summary>Full Stage 2 Extended Validation Sweep Table (24 configs, runs/tier1a_kraken_val_ext)</summary>

| Config | Beam | BlankPen | LenBonus | LM_Wt | Strict CER | Ortho CER | Corpus WER | Mean WER | Cat (>50%) | `book_06` CER (Del) | `book_11` CER (Del) | Time(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bw25_bp0.20_lb0.10_lm0.25` | 25 | 0.20 | 0.10 | 0.25 | 9.52% | 6.75% | 34.24% | 35.38% | 11 (0.74%) | 11.34% (1368) | 7.08% (953) | 44.2 |
| `bw25_bp0.20_lb0.10_lm0.35` | 25 | 0.20 | 0.10 | 0.35 | 9.41% | 6.66% | 33.65% | 34.79% | 11 (0.74%) | 11.23% (1461) | 6.96% (995) | 26.3 |
| `bw25_bp0.20_lb0.10_lm0.45` | 25 | 0.20 | 0.10 | 0.45 | 9.35% | 6.62% | 33.14% | 34.25% | 11 (0.74%) | 11.20% (1547) | 6.87% (1010) | 22.3 |
| `bw25_bp0.20_lb0.10_lm0.55` | 25 | 0.20 | 0.10 | 0.55 | 9.46% | 6.74% | 33.11% | 34.22% | 14 (0.94%) | 11.37% (1699) | 6.89% (1053) | 21.8 |
| `bw25_bp0.20_lb0.20_lm0.25` | 25 | 0.20 | 0.20 | 0.25 | 9.52% | 6.74% | 34.27% | 35.40% | 11 (0.74%) | 11.35% (1345) | 7.06% (923) | 22.4 |
| `bw25_bp0.20_lb0.20_lm0.35` | 25 | 0.20 | 0.20 | 0.35 | 9.36% | 6.61% | 33.56% | 34.70% | 11 (0.74%) | 11.19% (1410) | 6.91% (966) | 23.4 |
| `bw25_bp0.20_lb0.20_lm0.45` | 25 | 0.20 | 0.20 | 0.45 | 9.33% | 6.60% | 33.09% | 34.21% | 11 (0.74%) | 11.17% (1496) | 6.87% (996) | 22.0 |
| `bw25_bp0.20_lb0.20_lm0.55` | 25 | 0.20 | 0.20 | 0.55 | 9.43% | 6.70% | 33.05% | 34.16% | 13 (0.87%) | 11.34% (1645) | 6.86% (1029) | 22.2 |
| `bw25_bp0.20_lb0.30_lm0.25` | 25 | 0.20 | 0.30 | 0.25 | 9.49% | 6.71% | 34.26% | 35.38% | 10 (0.67%) | 11.33% (1292) | 7.02% (896) | 22.6 |
| `bw25_bp0.20_lb0.30_lm0.35` | 25 | 0.20 | 0.30 | 0.35 | 9.35% | 6.60% | 33.50% | 34.64% | 11 (0.74%) | 11.18% (1368) | 6.89% (948) | 22.4 |
| **`bw25_bp0.20_lb0.30_lm0.45`** | 25 | 0.20 | 0.30 | 0.45 | **9.30%** | **6.56%** | 33.03% | 34.16% | 12 (0.81%) | 11.16% (1458) | **6.80%** (962) | 24.0 |
| `bw25_bp0.20_lb0.30_lm0.55` | 25 | 0.20 | 0.30 | 0.55 | 9.39% | 6.66% | 33.05% | 34.16% | 13 (0.87%) | 11.29% (1584) | 6.85% (1002) | 23.8 |
| `bw25_bp0.40_lb0.10_lm0.25` | 25 | 0.40 | 0.10 | 0.25 | 9.53% | 6.76% | 34.29% | 35.41% | 11 (0.74%) | 11.35% (1345) | 7.08% (943) | 23.2 |
| `bw25_bp0.40_lb0.10_lm0.35` | 25 | 0.40 | 0.10 | 0.35 | 9.36% | 6.61% | 33.54% | 34.68% | 11 (0.74%) | 11.18% (1406) | 6.92% (972) | 23.3 |
| `bw25_bp0.40_lb0.10_lm0.45` | 25 | 0.40 | 0.10 | 0.45 | 9.34% | 6.61% | 33.11% | 34.23% | 11 (0.74%) | 11.19% (1502) | 6.86% (992) | 22.8 |
| `bw25_bp0.40_lb0.10_lm0.55` | 25 | 0.40 | 0.10 | 0.55 | 9.44% | 6.72% | 33.09% | 34.20% | 14 (0.94%) | 11.35% (1654) | 6.89% (1037) | 22.5 |
| `bw25_bp0.40_lb0.20_lm0.25` | 25 | 0.40 | 0.20 | 0.25 | 9.51% | 6.74% | 34.28% | 35.41% | 11 (0.74%) | 11.34% (1317) | 7.05% (911) | 24.9 |
| `bw25_bp0.40_lb0.20_lm0.35` | 25 | 0.40 | 0.20 | 0.35 | 9.36% | 6.61% | 33.54% | 34.67% | 11 (0.74%) | 11.19% (1377) | 6.91% (957) | 23.8 |
| `bw25_bp0.40_lb0.20_lm0.45` | 25 | 0.40 | 0.20 | 0.45 | 9.33% | 6.58% | 33.12% | 34.24% | 11 (0.74%) | 11.17% (1456) | 6.85% (969) | 23.7 |
| `bw25_bp0.40_lb0.20_lm0.55` | 25 | 0.40 | 0.20 | 0.55 | 9.38% | 6.66% | **32.99%** | **34.10%** | 13 (0.87%) | 11.28% (1586) | 6.83% (1008) | 22.1 |
| `bw25_bp0.40_lb0.30_lm0.25` | 25 | 0.40 | 0.30 | 0.25 | 9.48% | 6.71% | 34.24% | 35.31% | 10 (0.67%) | 11.34% (1271) | 6.99% (884) | 23.9 |
| `bw25_bp0.40_lb0.30_lm0.35` | 25 | 0.40 | 0.30 | 0.35 | 9.35% | 6.59% | 33.47% | 34.54% | 11 (0.74%) | 11.19% (1348) | 6.87% (935) | 25.3 |
| `bw25_bp0.40_lb0.30_lm0.45` | 25 | 0.40 | 0.30 | 0.45 | 9.31% | 6.57% | 33.11% | 34.23% | 12 (0.81%) | **11.15%** (1419) | 6.84% (952) | 25.7 |
| `bw25_bp0.40_lb0.30_lm0.55` | 25 | 0.40 | 0.30 | 0.55 | 9.36% | 6.62% | 33.00% | 34.11% | 13 (0.87%) | 11.25% (1529) | 6.82% (977) | 23.5 |

</details>

---

## 3. Official Test Set Results (`split = test`, $N = 6,744$)

### 3A. Overall Sweep Summary (18 Test Configurations)

| Config | Beam | BlankPen | LenBonus | LM_Wt | Strict CER | Ortho CER | Corpus WER | Mean WER | Cat (>50%) | Ruq'ah Del | Time(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Kraken Greedy Baseline (`EDA`)* | 1 | — | — | — | 22.22% | 20.10% | 59.31% | 63.06% | 272 (4.03%) | 6538 | — |
| *HATFormer (`beams=3`, `EDA`)* | 3 | — | — | — | 25.22% | 22.72% | 56.68% | 60.83% | 572 (8.48%) | 4210 | — |
| `bw25_bp0.00_lb0.00_lm0.00` (Beam-25 Pure) | 25 | 0.00 | 0.00 | 0.00 | 22.00% | 19.87% | 58.98% | 62.71% | 256 (3.80%) | 6621 | 94.9 |
| `bw25_bp0.00_lb0.00_lm0.25` | 25 | 0.00 | 0.00 | 0.25 | 20.47% | 18.52% | 52.20% | 55.66% | 247 (3.66%) | 7386 | 106.9 |
| `bw25_bp0.00_lb0.00_lm0.45` | 25 | 0.00 | 0.00 | 0.45 | 20.06% | 18.21% | 49.41% | 52.63% | 245 (3.63%) | 8147 | 104.2 |
| `bw25_bp0.00_lb0.30_lm0.00` | 25 | 0.00 | 0.30 | 0.00 | 22.06% | 19.91% | 59.13% | 62.95% | 261 (3.87%) | 5929 | 74.0 |
| `bw25_bp0.00_lb0.30_lm0.25` | 25 | 0.00 | 0.30 | 0.25 | 20.37% | 18.41% | 52.10% | 55.61% | 242 (3.59%) | 6720 | 109.1 |
| `bw25_bp0.00_lb0.30_lm0.45` | 25 | 0.00 | 0.30 | 0.45 | 19.86% | 17.99% | **49.24%** | **52.50%** | **236 (3.50%)** | 7404 | 107.8 |
| `bw25_bp0.20_lb0.00_lm0.00` | 25 | 0.20 | 0.00 | 0.00 | 22.03% | 19.89% | 59.05% | 62.81% | 262 (3.88%) | 6556 | 73.0 |
| `bw25_bp0.20_lb0.00_lm0.25` | 25 | 0.20 | 0.00 | 0.25 | 20.46% | 18.52% | 52.22% | 55.69% | 253 (3.75%) | 7335 | 153.2 |
| `bw25_bp0.20_lb0.00_lm0.45` | 25 | 0.20 | 0.00 | 0.45 | 20.03% | 18.17% | 49.39% | 52.61% | 239 (3.54%) | 8076 | 123.8 |
| `bw25_bp0.20_lb0.30_lm0.00` | 25 | 0.20 | 0.30 | 0.00 | 22.12% | 19.97% | 59.26% | 63.09% | 271 (4.02%) | 5880 | 74.9 |
| `bw25_bp0.20_lb0.30_lm0.25` | 25 | 0.20 | 0.30 | 0.25 | 20.40% | 18.44% | 52.15% | 55.67% | 244 (3.62%) | 6675 | 109.6 |
| **`bw25_bp0.20_lb0.30_lm0.45` (`ValOpt`)** | 25 | 0.20 | 0.30 | 0.45 | **19.86%** | **17.98%** | **49.25%** | **52.54%** | **242 (3.59%)** | 7337 | 104.1 |
| `bw25_bp0.40_lb0.00_lm0.00` | 25 | 0.40 | 0.00 | 0.00 | 22.07% | 19.93% | 59.16% | 62.93% | 263 (3.90%) | 6514 | 71.9 |
| `bw25_bp0.40_lb0.00_lm0.25` | 25 | 0.40 | 0.00 | 0.25 | 20.49% | 18.53% | 52.26% | 55.76% | 251 (3.72%) | 7295 | 107.9 |
| `bw25_bp0.40_lb0.00_lm0.45` | 25 | 0.40 | 0.00 | 0.45 | 20.01% | 18.14% | 49.43% | 52.68% | 249 (3.69%) | 7997 | 105.4 |
| `bw25_bp0.40_lb0.30_lm0.00` (`DelMin`) | 25 | 0.40 | 0.30 | 0.00 | 22.18% | 20.03% | 59.38% | 63.23% | 279 (4.14%) | **5834** | 71.7 |
| `bw25_bp0.40_lb0.30_lm0.25` (`RuqahOpt`) | 25 | 0.40 | 0.30 | 0.25 | 20.42% | 18.45% | 52.21% | 55.75% | 247 (3.66%) | 6623 | 335.6 |
| **`bw25_bp0.40_lb0.30_lm0.45` (`TestBest`)** | 25 | 0.40 | 0.30 | 0.45 | **19.84%** | **17.97%** | **49.25%** | 52.57% | 245 (3.63%) | 7284 | 101.6 |

### 3B. Per-Script Breakdown & Edit-Type Decomposition (Key Operating Points)

| Regime | Book / Script | $N$ | Strict CER | Ortho CER | Corp WER | Mean WER | Cat (>50%) | Ins | Del | Sub |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Kraken Base (EDA, `LevDist`)** | `book_03` (Maghrebi) | 3,661 | 30.85% | 28.76% | 75.61% | 77.42% | 199 | 11,637 | 9,487 | 36,190 |
| | `book_05` (Ruq'ah) | 2,027 | 11.25% | 9.19% | 38.28% | 39.01% | 42 | 1,734 | 6,538 | 8,056 |
| | `book_09` (Naskh) | 1,056 | 22.11% | 19.76% | 58.30% | 59.41% | 31 | 2,190 | 5,914 | 9,894 |
| **Pure Beam-25 (`lm=0`)** | `book_03` (Maghrebi) | 3,661 | 30.52% | 28.43% | 75.26% | 77.12% | 182 | 11,006 | 9,642 | 36,055 |
| (`bp0.00_lb0.00_lm0.00`) | `book_05` (Ruq'ah) | 2,027 | 11.15% | 9.08% | 37.98% | 38.59% | 43 | 1,555 | 6,621 | 7,999 |
| | `book_09` (Naskh) | 1,056 | 21.90% | 19.56% | 57.96% | 59.09% | 31 | 1,965 | 5,988 | 9,871 |
| **Del-Min Acoustic (`lm=0`)** | `book_03` (Maghrebi) | 3,661 | 31.03% | 28.92% | 76.14% | 78.05% | 216 | 13,004 | **8,235** | 36,419 |
| (`bp0.40_lb0.30_lm0.00`) | `book_05` (Ruq'ah) | 2,027 | 11.00% | 8.92% | 37.90% | 38.53% | **33** | 1,878 | **5,834** | 8,250 |
| | `book_09` (Naskh) | 1,056 | 21.89% | 19.53% | 58.07% | 59.24% | 30 | 2,405 | **5,249** | 10,159 |
| **Ruq'ah CER-Opt (`lm=0.25`)** | `book_03` (Maghrebi) | 3,661 | 28.38% | 26.41% | 66.93% | 68.81% | 166 | 10,046 | 9,800 | 32,877 |
| (`bp0.40_lb0.30_lm0.25`) | `book_05` (Ruq'ah) | 2,027 | **10.33%** | **8.48%** | 33.30% | 33.95% | 48 | 1,354 | 6,623 | 7,020 |
| | `book_09` (Naskh) | 1,056 | 20.23% | 18.06% | 51.19% | 52.34% | 33 | 1,700 | 5,962 | 8,800 |
| **`ValOpt` (`lm=0.45`)** | `book_03` (Maghrebi) | 3,661 | **27.23%** | **25.34%** | **62.51%** | **64.34%** | **147** | 8,420 | 11,568 | 30,608 |
| (`bp0.20_lb0.30_lm0.45`) | `book_05` (Ruq'ah) | 2,027 | 10.43% | 8.66% | **31.98%** | **32.60%** | 56 | 1,196 | 7,337 | 6,599 |
| | `book_09` (Naskh) | 1,056 | **19.84%** | **17.80%** | **48.69%** | **49.87%** | 39 | 1,457 | 6,590 | 8,095 |
| **`TestBest` (`bp=0.40`)** | `book_03` (Maghrebi) | 3,661 | **27.23%** | **25.34%** | 62.57% | 64.42% | 150 | 8,684 | 11,246 | 30,672 |
| (`bp0.40_lb0.30_lm0.45`) | `book_05` (Ruq'ah) | 2,027 | 10.42% | 8.65% | **31.97%** | 32.61% | 56 | 1,226 | 7,284 | 6,612 |
| | `book_09` (Naskh) | 1,056 | **19.78%** | **17.74%** | **48.61%** | **49.79%** | 39 | 1,492 | 6,487 | 8,115 |

*(Note: `bw25_bp0.00_lb0.30_lm0.45` also achieves `27.19%` CER / `141` Cat on `book_03` and `236` Cat / `49.24%` WER overall.)*

---

## 4. Paired Statistical Significance, Benchmark Strata, & Oracle Ceilings

### 4A. Paired Bootstrap (5,000 resamples, `seed=42`) & Two-Sided Wilcoxon Signed-Rank Tests ($N = 6,744$)

| Comparison ($\Delta = \text{Tier1A} - \text{Baseline}$) | $\Delta$ Strict CER (95% CI) | CER W / T / L | Wilcoxon CER $p$ | $\Delta$ Corpus WER (95% CI) | WER W / T / L | Wilcoxon WER $p$ |
|---|---|---|---:|---|---:|---:|
| **`ValOpt` vs. Kraken Greedy (`EDA`)** | **`-2.37 pp`** `[-2.46, -2.27]` | `4470 / 1226 / 1048` | `0.00e+00` | **`-10.06 pp`** `[-10.32, -9.80]` | `4550 / 1825 / 369` | `0.00e+00` |
| **`ValOpt` vs. HATFormer (`beams=3`)** | **`-5.36 pp`** `[-5.64, -5.08]` | `4822 / 491 / 1431` | `0.00e+00` | **`-7.43 pp`** `[-7.92, -6.94]` | `3923 / 1187 / 1634` | `9.06e-224` |
| **`TestBest` vs. Kraken Greedy (`EDA`)** | **`-2.38 pp`** `[-2.47, -2.29]` | `4483 / 1221 / 1040` | `0.00e+00` | **`-10.05 pp`** `[-10.31, -9.79]` | `4554 / 1811 / 379` | `0.00e+00` |
| **`TestBest` vs. HATFormer (`beams=3`)** | **`-5.37 pp`** `[-5.66, -5.09]` | `4827 / 487 / 1430` | `0.00e+00` | **`-7.42 pp`** `[-7.91, -6.93]` | `3928 / 1176 / 1640` | `1.53e-221` |

### 4B. Clean & Hygiene-Deduplicated Benchmark Strata Comparison

| Stratum | Model | Strict CER | Ortho CER | Corpus WER | Mean WER | Cat (>50%) |
|---|---|---:|---:|---:|---:|---:|
| **Artifact Clean ($N = 6,721$)** | Kraken Baseline (`EDA`) | 21.97% | 19.84% | 59.11% | 62.76% | 249 |
| *(excl. `page_034` + 3 blank crops)* | HATFormer (`beams=3`) | 24.98% | 22.47% | 56.46% | 60.46% | 549 |
| | **Kraken Tier 1A (`ValOpt`)** | **19.60%** | **17.72%** | **49.01%** | **52.21%** | **219** |
| | **Kraken Tier 1A (`TestBest`)** | **19.58%** | **17.71%** | **49.02%** | **52.24%** | **222** |
| **Hygiene-Deduplicated ($N = 6,673$)** | Kraken Baseline (`EDA`) | 21.88% | 19.75% | 59.01% | 62.65% | 241 |
| *(excl. 29 audit rows + 42 dup folio lines)* | HATFormer (`beams=3`) | 24.88% | 22.37% | 56.35% | 60.34% | 537 |
| | **Kraken Tier 1A (`ValOpt`)** | **19.52%** | **17.64%** | **48.94%** | **52.13%** | **213** |
| | **Kraken Tier 1A (`TestBest`)** | **19.50%** | **17.62%** | **48.94%** | **52.16%** | **216** |

### 4C. Updated Oracle Ensemble Upper Bound (`Tier 1A Kraken + HATFormer Beams=3`)

- **With `ValOpt` (`bw25_bp0.20_lb0.30_lm0.45`):** Oracle Corpus CER = **`18.02%`** (down from `19.89%`), CER-selected Oracle Corpus WER = **`46.02%`** (down from `52.99%`), WER-selected Oracle Ceiling = **`44.94%`** (down from `50.53%`), Oracle Catastrophic lines = **`145`** (down from `178`).
- **With `TestBest` (`bw25_bp0.40_lb0.30_lm0.45`):** Oracle Corpus CER = **`18.01%`**, CER-selected Oracle Corpus WER = **`46.03%`**, WER-selected Oracle Ceiling = **`44.94%`**, Oracle Catastrophic lines = **`145`**.

---

## 5. Mechanistic Insights & Deletion/Length Trade-Off

1. **Beam Search + 5-Gram LM Resolves Ambiguous Cursive Strokes Without Retraining:**
   - The largest absolute gains occur in dense **Maghrebi (`book_03`)**: Strict CER drops `-3.62 pp` (`30.85% -> 27.23%`) and Corpus WER drops `-13.10 pp` (`75.61% -> 62.51%`). Under paired `LevDist.opcodes`, substitutions fall `36,190 -> 30,608` (`-15.4%`) and insertions fall `11,637 -> 8,420` (`-27.6%`).
   - **Naskh (`book_09`)** similarly drops `-2.27 pp` Strict CER (`22.11% -> 19.84%`; `19.78%` at `TestBest`) and `-9.61 pp` Corpus WER (`58.30% -> 48.69%`).
2. **Why `lm_weight` Interacts Inversely with Ruq'ah Raw Deletions (and Why `length_bonus` + `blank_penalty` Compensate):**
   - At `lm_weight = 0.0`, enabling `blank_penalty=0.40 + length_bonus=0.30` directly counteracts CTC blank collapse: Ruq'ah (`book_05`) raw deletions drop **`6,538 -> 5,834` (`-10.8%`)** and Ruq'ah catastrophic lines drop **`42 -> 33`** (`CER 11.25% -> 11.00%`).
   - Conversely, every character transition under the LM incurs a negative log-probability term ($\mathbb{E}[\ln P_{\text{LM}}] \approx \ln(1/7.401) \approx -2.00\text{ nats/char}$ on `val`). Uncompensated `lm_weight=0.45` (`lb=0.0, bp=0.0`) penalizes sequence length and increases Ruq'ah deletions to **`8,147`**.
   - Adding `length_bonus=0.30` and `blank_penalty=0.40` recovers `863` deletions (`8,147 -> 7,284`) while achieving peak overall accuracy (`19.84%` CER). Operating at `bp=0.40, lb=0.30, lm=0.25` captures **Ruq'ah's best Strict CER (`10.33%`, Ortho `8.48%`, WER `33.30%`)** while keeping Ruq'ah deletions near baseline (`6,623`).
   - **Next Calibration Opportunity:** Centering LM log-probs (`ln P_LM(c|h) - mean_entropy`) or using `length_bonus >= 0.45` / script-specific `length_bonus` can offset the `-lm_weight * 2.00` per-char tax without sacrificing `lm=0.45`'s WER breakthroughs—potentially reducing both Maghrebi substitutions and Ruq'ah deletions simultaneously.

---

## 6. Qualitative Line Recovery Examples (`ValOpt` vs. Baseline Kraken)

1. **`book_03_line_003538` (Maghrebi) — CER `29.1% -> 10.9%`, WER `72.7% -> 36.4%`:**
   - **GT:** `لكم إلى قوله من الفجر حجة لفقهاء الأمصار ومن ذهب مذهبهم`
   - **Baseline:** `لثع إلى قوله فن العجر فجنة اعفهاه الأوصار ومن ذكعب قدبهم`
   - **Tier 1A:** `لكم إلى قوله من الفجر فجة الفقهاء الأوصار ومن ذهب قلبهم`
2. **`book_03_line_002599` (Maghrebi) — CER `32.7% -> 14.5%`, WER `75.0% -> 41.7%`:**
   - **GT:** `أبو حنيفة و الشافعي لا يقتل به في ذلك وقاله عطاء وإسحاق`
   - **Baseline:** `أبو ضنيعة والشر فعي لد تفتك به فبين ذالط وقاله عطا والسماف`
   - **Tier 1A:** `أبو حنيفة والشافعي لا تقتل به في ذالط وقاله عطاء والسماف`
3. **`book_05_line_004221` (Ruq'ah) — CER `9.2% -> 0.0%`, WER `40.0% -> 0.0%` (Exact Recovery):**
   - **GT:** `والفتوى والقضاء وإماطة الأذى عن الطريق والنصيحة والإعانة على البر`
   - **Baseline:** `والفتوى والقضاء وأماطه الأذى عن الطريقة والنصيحة والإاعاتة على البرة`
   - **Tier 1A:** `والفتوى والقضاء وإماطة الأذى عن الطريق والنصيحة والإعانة على البر`
4. **`book_05_line_004029` (Ruq'ah) — CER `11.7% -> 2.6%`, WER `46.7% -> 13.3%`:**
   - **GT:** `وستين سنة على الصحيح وغسله ابنه الزاهد ابو عبد الرحمن عبد الله وكفنه في ثوبين`
   - **Baseline:** `وستهر سنه على الصحيح وعمسله ابنه الزاجد أبو عبد الرحمن عبد الله وقفنه في ثوين`
   - **Tier 1A:** `وستين سنة على الصحيح ومسله ابنه الزاهد أبو عبد الرحمن عبد الله وكفنه في ثوبين`
5. **`book_09_line_006033` (Naskh) — CER `18.8% -> 4.3%`, WER `57.1% -> 21.4%`:**
   - **GT:** `هذين الموطأين على سائر الموطآت نحو من مائة حديث زائدة وهذا يدل على أن`
   - **Baseline:** `هذين الموطاين على سنابد الموطات نحيو من بائة حديث زابيدة وهمد بد على أن`
   - **Tier 1A:** `هذين الموطين على سائر الموطات نحو من مائة حديث زائدة وهما يدل على أن`
6. **`book_09_line_006361` (Naskh) — CER `28.0% -> 14.7%`, WER `93.3% -> 60.0%`:**
   - **GT:** `انكر عليه احاديث وكثرة روايته عن عمه وكلما انكروه عليه فمحتمل ان لم يروه عن`
   - **Baseline:** `أنكر عليه أحاديت ذكثره رواتيه حن عمة وكل ما أكروه علية فحتمك وأز لم يزو عمن`
   - **Tier 1A:** `أنكر عليه أحاديت وكثرة رواية عن عمه وكل ما أنكروه عليه فحتمل وإن لم يروه من`
