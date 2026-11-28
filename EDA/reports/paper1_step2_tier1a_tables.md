# Paper 1 Step 2 — Tier 1A Zero-Training Decoding Calibration Audit Tables

**Project:** AraMS-28k Benchmark Audit & Zero-Training Fix (Paper 1, Path B)  
**Official Test Split:** $N = 6,744$ (`book_03` Maghrebi $n=3,661$, `book_05` Ruq'ah $n=2,027$, `book_09` Naskh $n=1,056$)  
**Full Hygiene-Deduplicated Stratum:** $N = 6,673$ (excludes all 29 rows of `arams28k_benchmark_audit_artifact.csv` + 42 duplicated folio lines on `book_03_page_103`/`104`)  
**Statistical Protocol:** Corpus-weighted paired bootstrap ($n=10,000$, `seed=42`), two-sided paired Wilcoxon signed-rank tests with Holm–Bonferroni ($m=8$ per comparison pair).


---

## 1. Dual Stratum Comparison Table (Official $N=6,744$ vs. Full Hygiene-Deduplicated $N=6,673$)

> Clean comparison across all baseline and Tier 1A models on both the official split and the leak-/artifact-/duplicate-free hygiene stratum.

| Stratum | Split | Model | $N$ | Strict CER | Ortho CER | Corp WER | Mean WER | Cat (>50%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Official Test | **OVERALL** | Kraken Greedy (Baseline) | 6,744 | 22.22% | 20.10% | 59.31% | 63.06% | 272 (4.03%) |
| Official Test | **OVERALL** | Kraken Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`) | 6,744 | 19.86% | 17.98% | 49.25% | 52.54% | 242 (3.59%) |
| Official Test | **OVERALL** | Kraken Tier 1A TestBest (`bw25_bp0.40_lb0.30_lm0.45`) | 6,744 | 19.84% | 17.97% | 49.25% | 52.57% | 245 (3.63%) |
| Official Test | **OVERALL** | HATFormer Baseline (`beams=3`) | 6,744 | 25.22% | 22.72% | 56.68% | 60.83% | 572 (8.48%) |
| Official Test | **OVERALL** | HATFormer `01_plus_blank_gate` | 6,744 | 25.22% | 22.72% | 56.67% | 60.78% | 572 (8.48%) |
| Official Test | **OVERALL** | HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`) | 6,744 | 25.24% | 22.72% | 56.94% | 61.26% | 566 (8.39%) |
| Hygiene-Deduplicated | **OVERALL** | Kraken Greedy (Baseline) | 6,673 | 21.88% | 19.75% | 59.01% | 62.65% | 241 (3.61%) |
| Hygiene-Deduplicated | **OVERALL** | Kraken Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`) | 6,673 | 19.52% | 17.64% | 48.94% | 52.13% | 213 (3.19%) |
| Hygiene-Deduplicated | **OVERALL** | Kraken Tier 1A TestBest (`bw25_bp0.40_lb0.30_lm0.45`) | 6,673 | 19.50% | 17.62% | 48.94% | 52.16% | 216 (3.24%) |
| Hygiene-Deduplicated | **OVERALL** | HATFormer Baseline (`beams=3`) | 6,673 | 24.88% | 22.37% | 56.35% | 60.34% | 537 (8.05%) |
| Hygiene-Deduplicated | **OVERALL** | HATFormer `01_plus_blank_gate` | 6,673 | 24.88% | 22.37% | 56.35% | 60.34% | 537 (8.05%) |
| Hygiene-Deduplicated | **OVERALL** | HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`) | 6,673 | 24.89% | 22.37% | 56.62% | 60.82% | 531 (7.96%) |
| Official Test | `book_03` (Maghrebi) | Kraken Greedy (Baseline) | 3,661 | 30.85% | 28.76% | 75.61% | 77.42% | 199 (5.44%) |
| Official Test | `book_03` (Maghrebi) | Kraken Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`) | 3,661 | 27.23% | 25.34% | 62.51% | 64.34% | 147 (4.02%) |
| Official Test | `book_03` (Maghrebi) | Kraken Tier 1A TestBest (`bw25_bp0.40_lb0.30_lm0.45`) | 3,661 | 27.23% | 25.34% | 62.57% | 64.42% | 150 (4.10%) |
| Official Test | `book_03` (Maghrebi) | HATFormer Baseline (`beams=3`) | 3,661 | 36.39% | 33.81% | 73.38% | 75.54% | 528 (14.42%) |
| Official Test | `book_03` (Maghrebi) | HATFormer `01_plus_blank_gate` | 3,661 | 36.39% | 33.81% | 73.38% | 75.54% | 528 (14.42%) |
| Official Test | `book_03` (Maghrebi) | HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`) | 3,661 | 36.54% | 33.94% | 73.94% | 76.40% | 527 (14.39%) |
| Official Test | `book_05` (Ruq'ah) | Kraken Greedy (Baseline) | 2,027 | 11.25% | 9.19% | 38.28% | 39.01% | 42 (2.07%) |
| Official Test | `book_05` (Ruq'ah) | Kraken Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`) | 2,027 | 10.43% | 8.66% | 31.98% | 32.60% | 56 (2.76%) |
| Official Test | `book_05` (Ruq'ah) | Kraken Tier 1A TestBest (`bw25_bp0.40_lb0.30_lm0.45`) | 2,027 | 10.42% | 8.65% | 31.97% | 32.61% | 56 (2.76%) |
| Official Test | `book_05` (Ruq'ah) | HATFormer Baseline (`beams=3`) | 2,027 | 11.75% | 9.31% | 36.57% | 37.50% | 14 (0.69%) |
| Official Test | `book_05` (Ruq'ah) | HATFormer `01_plus_blank_gate` | 2,027 | 11.75% | 9.31% | 36.56% | 37.35% | 14 (0.69%) |
| Official Test | `book_05` (Ruq'ah) | HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`) | 2,027 | 11.72% | 9.26% | 36.64% | 37.40% | 12 (0.59%) |
| Official Test | `book_09` (Naskh) | Kraken Greedy (Baseline) | 1,056 | 22.11% | 19.76% | 58.30% | 59.41% | 31 (2.94%) |
| Official Test | `book_09` (Naskh) | Kraken Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`) | 1,056 | 19.84% | 17.80% | 48.69% | 49.87% | 39 (3.69%) |
| Official Test | `book_09` (Naskh) | Kraken Tier 1A TestBest (`bw25_bp0.40_lb0.30_lm0.45`) | 1,056 | 19.78% | 17.74% | 48.61% | 49.79% | 39 (3.69%) |
| Official Test | `book_09` (Naskh) | HATFormer Baseline (`beams=3`) | 1,056 | 23.72% | 21.30% | 53.18% | 54.57% | 30 (2.84%) |
| Official Test | `book_09` (Naskh) | HATFormer `01_plus_blank_gate` | 1,056 | 23.73% | 21.30% | 53.18% | 54.57% | 30 (2.84%) |
| Official Test | `book_09` (Naskh) | HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`) | 1,056 | 23.53% | 21.10% | 53.09% | 54.57% | 27 (2.56%) |
| Hygiene-Deduplicated | `book_03` (Maghrebi) | Kraken Greedy (Baseline) | 3,614 | 30.77% | 28.69% | 75.60% | 77.42% | 192 (5.31%) |
| Hygiene-Deduplicated | `book_03` (Maghrebi) | Kraken Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`) | 3,614 | 27.17% | 25.28% | 62.53% | 64.37% | 142 (3.93%) |
| Hygiene-Deduplicated | `book_03` (Maghrebi) | Kraken Tier 1A TestBest (`bw25_bp0.40_lb0.30_lm0.45`) | 3,614 | 27.16% | 25.27% | 62.58% | 64.44% | 145 (4.01%) |
| Hygiene-Deduplicated | `book_03` (Maghrebi) | HATFormer Baseline (`beams=3`) | 3,614 | 36.33% | 33.75% | 73.37% | 75.54% | 517 (14.31%) |
| Hygiene-Deduplicated | `book_03` (Maghrebi) | HATFormer `01_plus_blank_gate` | 3,614 | 36.33% | 33.75% | 73.37% | 75.54% | 517 (14.31%) |
| Hygiene-Deduplicated | `book_03` (Maghrebi) | HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`) | 3,614 | 36.47% | 33.87% | 73.93% | 76.40% | 516 (14.28%) |
| Hygiene-Deduplicated | `book_05` (Ruq'ah) | Kraken Greedy (Baseline) | 2,024 | 11.17% | 9.11% | 38.22% | 38.92% | 39 (1.93%) |
| Hygiene-Deduplicated | `book_05` (Ruq'ah) | Kraken Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`) | 2,024 | 10.35% | 8.58% | 31.91% | 32.50% | 53 (2.62%) |
| Hygiene-Deduplicated | `book_05` (Ruq'ah) | Kraken Tier 1A TestBest (`bw25_bp0.40_lb0.30_lm0.45`) | 2,024 | 10.34% | 8.57% | 31.91% | 32.51% | 53 (2.62%) |
| Hygiene-Deduplicated | `book_05` (Ruq'ah) | HATFormer Baseline (`beams=3`) | 2,024 | 11.67% | 9.23% | 36.50% | 37.26% | 11 (0.54%) |
| Hygiene-Deduplicated | `book_05` (Ruq'ah) | HATFormer `01_plus_blank_gate` | 2,024 | 11.67% | 9.23% | 36.50% | 37.26% | 11 (0.54%) |
| Hygiene-Deduplicated | `book_05` (Ruq'ah) | HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`) | 2,024 | 11.65% | 9.18% | 36.58% | 37.31% | 9 (0.44%) |
| Hygiene-Deduplicated | `book_09` (Naskh) | Kraken Greedy (Baseline) | 1,035 | 20.87% | 18.51% | 57.32% | 57.51% | 10 (0.97%) |
| Hygiene-Deduplicated | `book_09` (Naskh) | Kraken Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`) | 1,035 | 18.58% | 16.54% | 47.54% | 47.78% | 18 (1.74%) |
| Hygiene-Deduplicated | `book_09` (Naskh) | Kraken Tier 1A TestBest (`bw25_bp0.40_lb0.30_lm0.45`) | 1,035 | 18.52% | 16.47% | 47.45% | 47.70% | 18 (1.74%) |
| Hygiene-Deduplicated | `book_09` (Naskh) | HATFormer Baseline (`beams=3`) | 1,035 | 22.52% | 20.08% | 52.06% | 52.40% | 9 (0.87%) |
| Hygiene-Deduplicated | `book_09` (Naskh) | HATFormer `01_plus_blank_gate` | 1,035 | 22.52% | 20.08% | 52.06% | 52.40% | 9 (0.87%) |
| Hygiene-Deduplicated | `book_09` (Naskh) | HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`) | 1,035 | 22.32% | 19.87% | 51.96% | 52.40% | 6 (0.58%) |

---

## 2. Pair A — Kraken Gain: Greedy → Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`)

> Leak-free CTC prefix beam search (`beam=25, blank_pen=0.20, len_bonus=0.30, lm_wt=0.45`) with 5-gram KN LM trained solely on `split=='train'`. Negative Δ = Tier 1A reduces error.


### Strict Corpus CER (%)

| Book | Script | Kraken Greedy | Kraken Greedy 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (T1A − Greedy) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 30.85% | [30.50%, 31.20%] | 27.23% | [26.87%, 27.60%] | **-3.62** | [-3.76, -3.47] |
| `book_05` | Ruq'ah | 11.25% | [10.80%, 11.73%] | 10.43% | [9.92%, 10.96%] | **-0.83** | [-0.95, -0.70] |
| `book_09` | Naskh | 22.11% | [21.36%, 22.90%] | 19.84% | [19.04%, 20.67%] | **-2.27** | [-2.47, -2.08] |
| **OVERALL** | Combined | 22.22% | [21.89%, 22.58%] | 19.86% | [19.52%, 20.21%] | **-2.37** | [-2.46, -2.28] |

### Ortho-Folded Corpus CER (%)

| Book | Script | Kraken Greedy | Kraken Greedy 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (T1A − Greedy) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 28.76% | [28.44%, 29.10%] | 25.34% | [25.00%, 25.69%] | **-3.42** | [-3.56, -3.28] |
| `book_05` | Ruq'ah | 9.19% | [8.74%, 9.65%] | 8.66% | [8.16%, 9.18%] | **-0.53** | [-0.65, -0.41] |
| `book_09` | Naskh | 19.76% | [19.02%, 20.55%] | 17.80% | [17.02%, 18.64%] | **-1.96** | [-2.15, -1.77] |
| **OVERALL** | Combined | 20.10% | [19.76%, 20.45%] | 17.98% | [17.65%, 18.33%] | **-2.11** | [-2.20, -2.02] |

### True Corpus WER (%)

| Book | Script | Kraken Greedy | Kraken Greedy 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (T1A − Greedy) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 75.61% | [74.98%, 76.24%] | 62.51% | [61.84%, 63.20%] | **-13.09** | [-13.48, -12.69] |
| `book_05` | Ruq'ah | 38.28% | [37.38%, 39.19%] | 31.98% | [31.08%, 32.88%] | **-6.30** | [-6.70, -5.91] |
| `book_09` | Naskh | 58.30% | [57.16%, 59.47%] | 48.69% | [47.47%, 49.91%] | **-9.61** | [-10.16, -9.07] |
| **OVERALL** | Combined | 59.31% | [58.70%, 59.94%] | 49.25% | [48.66%, 49.85%] | **-10.06** | [-10.32, -9.80] |

### True Mean WER (%)

| Book | Script | Kraken Greedy | Kraken Greedy 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (T1A − Greedy) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 77.42% | [76.56%, 78.40%] | 64.34% | [63.46%, 65.32%] | **-13.08** | [-13.49, -12.65] |
| `book_05` | Ruq'ah | 39.01% | [38.03%, 40.00%] | 32.60% | [31.66%, 33.55%] | **-6.41** | [-6.88, -5.97] |
| `book_09` | Naskh | 59.41% | [57.71%, 61.47%] | 49.87% | [48.06%, 51.98%] | **-9.55** | [-10.12, -9.00] |
| **OVERALL** | Combined | 63.06% | [62.33%, 63.84%] | 52.54% | [51.82%, 53.29%] | **-10.52** | [-10.81, -10.23] |

### Holm–Bonferroni Adjusted P-Values ($m = 8$: 4 splits $\times$ 2 line metrics)

| Book | Script | Metric | Wilcoxon $W$ | Median Δ (T1A − Greedy) | Kraken T1A W/T/L | Raw $p$ | Holm–Bonferroni Adj $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | CER | 467,076 | -3.64 pp | `2641/548/472` | `0.00e+00` | **`0.00e+00`** (✓) |
| `book_03` | Maghrebi | True WER | 101,222 | -11.11 pp | `2627/906/128` | `0.00e+00` | **`0.00e+00`** (✓) |
| `book_05` | Ruq'ah | CER | 282,913 | -1.27 pp | `1071/543/413` | `2.944e-59` | **`2.944e-59`** (✓) |
| `book_05` | Ruq'ah | True WER | 86,603 | -6.25 pp | `1136/709/182` | `5.521e-140` | **`2.209e-139`** (✓) |
| `book_09` | Naskh | CER | 55,210 | -2.41 pp | `755/138/163` | `1.230e-83` | **`2.460e-83`** (✓) |
| `book_09` | Naskh | True WER | 12,584 | -7.69 pp | `787/210/59` | `2.071e-121` | **`6.213e-121`** (✓) |
| **OVERALL** | Combined | CER | 2,109,166 | -2.04 pp | `4467/1229/1048` | `0.00e+00` | **`0.00e+00`** (✓) |
| **OVERALL** | Combined | True WER | 481,924 | -9.09 pp | `4550/1825/369` | `0.00e+00` | **`0.00e+00`** (✓) |

---

## 3A. Pair B1 — Paradox Reversal: Kraken Tier 1A vs. HATFormer `01_plus_blank_gate` (LP=0.0+BG)

> Tests whether Kraken Tier 1A beats clean HATFormer (`LP=0.0 + Blank-Gate`) simultaneously on Strict CER, Ortho CER, Corpus WER, and Mean WER across all 3 scripts. Negative Δ = Kraken Tier 1A has lower error.


### Strict Corpus CER (%)

| Book | Script | HAT 01-BG | HAT 01-BG 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (K_T1A − H_BG) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 36.39% | [35.97%, 36.82%] | 27.23% | [26.87%, 27.60%] | **-9.16** | [-9.53, -8.80] |
| `book_05` | Ruq'ah | 11.75% | [11.43%, 12.09%] | 10.43% | [9.92%, 10.96%] | **-1.33** | [-1.81, -0.83] |
| `book_09` | Naskh | 23.73% | [22.94%, 24.54%] | 19.84% | [19.04%, 20.67%] | **-3.89** | [-4.50, -3.29] |
| **OVERALL** | Combined | 25.22% | [24.85%, 25.60%] | 19.86% | [19.52%, 20.21%] | **-5.36** | [-5.64, -5.08] |

### Ortho-Folded Corpus CER (%)

| Book | Script | HAT 01-BG | HAT 01-BG 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (K_T1A − H_BG) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 33.81% | [33.40%, 34.22%] | 25.34% | [25.00%, 25.69%] | **-8.47** | [-8.82, -8.13] |
| `book_05` | Ruq'ah | 9.31% | [8.99%, 9.63%] | 8.66% | [8.16%, 9.18%] | **-0.65** | [-1.12, -0.17] |
| `book_09` | Naskh | 21.30% | [20.52%, 22.13%] | 17.80% | [17.02%, 18.64%] | **-3.50** | [-4.11, -2.91] |
| **OVERALL** | Combined | 22.72% | [22.36%, 23.10%] | 17.98% | [17.65%, 18.33%] | **-4.74** | [-5.01, -4.46] |

### True Corpus WER (%)

| Book | Script | HAT 01-BG | HAT 01-BG 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (K_T1A − H_BG) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 73.38% | [72.66%, 74.08%] | 62.51% | [61.84%, 63.20%] | **-10.86** | [-11.52, -10.19] |
| `book_05` | Ruq'ah | 36.56% | [35.83%, 37.28%] | 31.98% | [31.08%, 32.88%] | **-4.58** | [-5.47, -3.69] |
| `book_09` | Naskh | 53.18% | [51.98%, 54.37%] | 48.69% | [47.47%, 49.91%] | **-4.49** | [-5.57, -3.43] |
| **OVERALL** | Combined | 56.67% | [56.07%, 57.29%] | 49.25% | [48.66%, 49.85%] | **-7.43** | [-7.91, -6.93] |

### True Mean WER (%)

| Book | Script | HAT 01-BG | HAT 01-BG 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (K_T1A − H_BG) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 75.54% | [74.51%, 76.72%] | 64.34% | [63.46%, 65.32%] | **-11.20** | [-11.97, -10.44] |
| `book_05` | Ruq'ah | 37.35% | [36.47%, 38.26%] | 32.60% | [31.66%, 33.55%] | **-4.75** | [-5.72, -3.78] |
| `book_09` | Naskh | 54.57% | [52.70%, 56.83%] | 49.87% | [48.06%, 51.98%] | **-4.70** | [-5.80, -3.62] |
| **OVERALL** | Combined | 60.78% | [59.98%, 61.64%] | 52.54% | [51.82%, 53.29%] | **-8.25** | [-8.78, -7.70] |

### Holm–Bonferroni Adjusted P-Values ($m = 8$: 4 splits $\times$ 2 line metrics)

| Book | Script | Metric | Wilcoxon $W$ | Median Δ (K_T1A − H_BG) | Kraken T1A W/T/L | Raw $p$ | Holm–Bonferroni Adj $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | CER | 667,750 | -8.93 pp | `2833/209/619` | `0.00e+00` | **`0.00e+00`** (✓) |
| `book_03` | Maghrebi | True WER | 877,498 | -10.00 pp | `2226/677/758` | `8.625e-181` | **`4.313e-180`** (✓) |
| `book_05` | Ruq'ah | CER | 472,488 | -2.63 pp | `1296/201/530` | `6.286e-58` | **`2.514e-57`** (✓) |
| `book_05` | Ruq'ah | True WER | 459,916 | -6.67 pp | `1144/335/548` | `3.181e-37` | **`6.363e-37`** (✓) |
| `book_09` | Naskh | CER | 108,692 | -3.95 pp | `692/84/280` | `3.172e-48` | **`9.516e-48`** (✓) |
| `book_09` | Naskh | True WER | 126,854 | -5.88 pp | `552/176/328` | `6.706e-19` | **`6.706e-19`** (✓) |
| **OVERALL** | Combined | CER | 3,467,612 | -5.26 pp | `4821/494/1429` | `0.00e+00` | **`0.00e+00`** (✓) |
| **OVERALL** | Combined | True WER | 3,902,352 | -8.33 pp | `3922/1188/1634` | `1.446e-223` | **`8.675e-223`** (✓) |

---

## 3B. Pair B2 — Paradox Reversal: Kraken Tier 1A vs. HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`)

> Tests whether Kraken Tier 1A beats Val-selected HATFormer Tier 1A simultaneously on CER and WER across all 3 scripts. Negative Δ = Kraken Tier 1A has lower error.


### Strict Corpus CER (%)

| Book | Script | HAT ValOpt | HAT ValOpt 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (K_T1A − H_VO) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 36.54% | [36.12%, 36.97%] | 27.23% | [26.87%, 27.60%] | **-9.31** | [-9.68, -8.96] |
| `book_05` | Ruq'ah | 11.72% | [11.41%, 12.05%] | 10.43% | [9.92%, 10.96%] | **-1.30** | [-1.78, -0.81] |
| `book_09` | Naskh | 23.53% | [22.79%, 24.32%] | 19.84% | [19.04%, 20.67%] | **-3.70** | [-4.27, -3.12] |
| **OVERALL** | Combined | 25.24% | [24.88%, 25.62%] | 19.86% | [19.52%, 20.21%] | **-5.38** | [-5.66, -5.10] |

### Ortho-Folded Corpus CER (%)

| Book | Script | HAT ValOpt | HAT ValOpt 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (K_T1A − H_VO) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 33.94% | [33.53%, 34.36%] | 25.34% | [25.00%, 25.69%] | **-8.60** | [-8.95, -8.26] |
| `book_05` | Ruq'ah | 9.26% | [8.95%, 9.57%] | 8.66% | [8.16%, 9.18%] | **-0.60** | [-1.07, -0.13] |
| `book_09` | Naskh | 21.10% | [20.37%, 21.89%] | 17.80% | [17.02%, 18.64%] | **-3.30** | [-3.86, -2.73] |
| **OVERALL** | Combined | 22.72% | [22.36%, 23.10%] | 17.98% | [17.65%, 18.33%] | **-4.74** | [-5.00, -4.46] |

### True Corpus WER (%)

| Book | Script | HAT ValOpt | HAT ValOpt 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (K_T1A − H_VO) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 73.94% | [73.21%, 74.65%] | 62.51% | [61.84%, 63.20%] | **-11.43** | [-12.09, -10.77] |
| `book_05` | Ruq'ah | 36.64% | [35.91%, 37.37%] | 31.98% | [31.08%, 32.88%] | **-4.67** | [-5.56, -3.78] |
| `book_09` | Naskh | 53.09% | [51.92%, 54.24%] | 48.69% | [47.47%, 49.91%] | **-4.39** | [-5.41, -3.38] |
| **OVERALL** | Combined | 56.94% | [56.34%, 57.55%] | 49.25% | [48.66%, 49.85%] | **-7.70** | [-8.17, -7.21] |

### True Mean WER (%)

| Book | Script | HAT ValOpt | HAT ValOpt 95% CI | Kraken T1A | Kraken T1A 95% CI | Δ (K_T1A − H_VO) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 76.40% | [75.22%, 77.73%] | 64.34% | [63.46%, 65.32%] | **-12.06** | [-13.02, -11.19] |
| `book_05` | Ruq'ah | 37.40% | [36.54%, 38.26%] | 32.60% | [31.66%, 33.55%] | **-4.80** | [-5.75, -3.86] |
| `book_09` | Naskh | 54.57% | [52.71%, 56.84%] | 49.87% | [48.06%, 51.98%] | **-4.71** | [-5.78, -3.66] |
| **OVERALL** | Combined | 61.26% | [60.41%, 62.18%] | 52.54% | [51.82%, 53.29%] | **-8.73** | [-9.34, -8.12] |

### Holm–Bonferroni Adjusted P-Values ($m = 8$: 4 splits $\times$ 2 line metrics)

| Book | Script | Metric | Wilcoxon $W$ | Median Δ (K_T1A − H_VO) | Kraken T1A W/T/L | Raw $p$ | Holm–Bonferroni Adj $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | CER | 637,488 | -9.09 pp | `2844/217/600` | `0.00e+00` | **`0.00e+00`** (✓) |
| `book_03` | Maghrebi | True WER | 828,642 | -10.00 pp | `2266/662/733` | `3.218e-197` | **`1.609e-196`** (✓) |
| `book_05` | Ruq'ah | CER | 471,608 | -2.60 pp | `1297/203/527` | `8.086e-58` | **`3.234e-57`** (✓) |
| `book_05` | Ruq'ah | True WER | 454,248 | -6.67 pp | `1140/341/546` | `9.002e-38` | **`1.800e-37`** (✓) |
| `book_09` | Naskh | CER | 111,712 | -3.95 pp | `698/78/280` | `2.612e-47` | **`7.836e-47`** (✓) |
| `book_09` | Naskh | True WER | 126,616 | -5.88 pp | `555/175/326` | `3.418e-19` | **`3.418e-19`** (✓) |
| **OVERALL** | Combined | CER | 3,418,606 | -5.33 pp | `4839/498/1407` | `0.00e+00` | **`0.00e+00`** (✓) |
| **OVERALL** | Combined | True WER | 3,781,582 | -8.33 pp | `3961/1178/1605` | `7.369e-240` | **`4.421e-239`** (✓) |

---

## 4A. Pair C — HATFormer Tier 1A Diagnostic: Baseline → Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`)

> Evaluates why zero-training decoding saturates on HATFormer: `LP=1.0 + RP=1.15 + LoopFB` improves Naskh Corpus CER (`-0.19 pp`, 95% bootstrap CI excludes 0 due to tail-loop rescues, though symmetric 73/71 per-line ranks make Wilcoxon CER non-significant) while significantly degrading Maghrebi CER (`+0.15 pp`) and WER (`+0.57 pp`). Negative Δ = Val-Selected reduces error.


### Strict Corpus CER (%)

| Book | Script | HAT Base | HAT Base 95% CI | HAT ValOpt | HAT ValOpt 95% CI | Δ (VO − Base) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 36.39% | [35.97%, 36.82%] | 36.54% | [36.12%, 36.97%] | **+0.15** | [+0.05, +0.25] |
| `book_05` | Ruq'ah | 11.75% | [11.43%, 12.09%] | 11.72% | [11.41%, 12.05%] | **-0.03** | [-0.12, +0.04] |
| `book_09` | Naskh | 23.72% | [22.93%, 24.53%] | 23.53% | [22.79%, 24.32%] | **-0.19** | [-0.43, -0.01] |
| **OVERALL** | Combined | 25.22% | [24.85%, 25.60%] | 25.24% | [24.88%, 25.62%] | **+0.02** | [-0.05, +0.09] |

### Ortho-Folded Corpus CER (%)

| Book | Script | HAT Base | HAT Base 95% CI | HAT ValOpt | HAT ValOpt 95% CI | Δ (VO − Base) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 33.81% | [33.40%, 34.22%] | 33.94% | [33.53%, 34.36%] | **+0.13** | [+0.03, +0.23] |
| `book_05` | Ruq'ah | 9.31% | [9.00%, 9.63%] | 9.26% | [8.95%, 9.57%] | **-0.05** | [-0.14, +0.02] |
| `book_09` | Naskh | 21.30% | [20.52%, 22.11%] | 21.10% | [20.37%, 21.89%] | **-0.20** | [-0.44, -0.02] |
| **OVERALL** | Combined | 22.72% | [22.36%, 23.09%] | 22.72% | [22.36%, 23.10%] | **+0.00** | [-0.07, +0.07] |

### True Corpus WER (%)

| Book | Script | HAT Base | HAT Base 95% CI | HAT ValOpt | HAT ValOpt 95% CI | Δ (VO − Base) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 73.38% | [72.66%, 74.08%] | 73.94% | [73.21%, 74.65%] | **+0.57** | [+0.43, +0.70] |
| `book_05` | Ruq'ah | 36.57% | [35.84%, 37.29%] | 36.64% | [35.91%, 37.37%] | **+0.07** | [-0.03, +0.17] |
| `book_09` | Naskh | 53.18% | [51.98%, 54.37%] | 53.09% | [51.92%, 54.24%] | **-0.09** | [-0.42, +0.14] |
| **OVERALL** | Combined | 56.68% | [56.08%, 57.29%] | 56.94% | [56.34%, 57.55%] | **+0.27** | [+0.17, +0.35] |

### True Mean WER (%)

| Book | Script | HAT Base | HAT Base 95% CI | HAT ValOpt | HAT ValOpt 95% CI | Δ (VO − Base) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 75.54% | [74.51%, 76.72%] | 76.40% | [75.22%, 77.73%] | **+0.86** | [+0.46, +1.47] |
| `book_05` | Ruq'ah | 37.50% | [36.56%, 38.48%] | 37.40% | [36.54%, 38.26%] | **-0.10** | [-0.48, +0.15] |
| `book_09` | Naskh | 54.57% | [52.70%, 56.83%] | 54.57% | [52.71%, 56.84%] | **+0.01** | [-0.33, +0.31] |
| **OVERALL** | Combined | 60.83% | [60.02%, 61.69%] | 61.26% | [60.41%, 62.18%] | **+0.44** | [+0.18, +0.78] |

### Holm–Bonferroni Adjusted P-Values ($m = 8$: 4 splits $\times$ 2 line metrics)

| Book | Script | Metric | Wilcoxon $W$ | Median Δ (VO − Base) | HAT ValOpt W/T/L | Raw $p$ | Holm–Bonferroni Adj $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | CER | 73,891 | +0.00 pp | `248/3034/379` | `6.304e-08` | **`3.783e-07`** (✓) |
| `book_03` | Maghrebi | True WER | 15,959 | +0.00 pp | `90/3310/261` | `3.992e-15` | **`2.794e-14`** (✓) |
| `book_05` | Ruq'ah | CER | 6,176 | +0.00 pp | `74/1867/86` | `0.6529` | **`1.0000`** (✗) |
| `book_05` | Ruq'ah | True WER | 1,984 | +0.00 pp | `38/1923/66` | `0.0156` | **`0.0623`** (✗) |
| `book_09` | Naskh | CER | 4,763 | +0.00 pp | `73/912/71` | `0.3621` | **`1.0000`** (✗) |
| `book_09` | Naskh | True WER | 1,334 | +0.00 pp | `32/982/42` | `0.7710` | **`1.0000`** (✗) |
| **OVERALL** | Combined | CER | 179,255 | +0.00 pp | `395/5813/536` | `4.437e-06` | **`2.219e-05`** (✓) |
| **OVERALL** | Combined | True WER | 41,316 | +0.00 pp | `160/6215/369` | `2.740e-16` | **`2.192e-15`** (✓) |

> **Blank-Gate Surgical Effect (`01_plus_blank_gate` vs. Baseline):** Identical on all 6,741 legible lines ($p=1.0000$); eliminates hallucinated Arabic boilerplate (`صلى الله عليه وسلم`, `ونحو ذلك`) on the 3 verified blank margin crops (`book_05_line_004216`, `004741`, `book_09_line_006514`) with 0 false positives across val+test.


---

## 5A. Post-Tier-1A Oracle Ceiling, 2×2 Matrix & Disagreement Scoping (Kraken ValOpt + HATFormer 01-BG)

> Paired complementarity between **`Kraken Tier 1A ValOpt`** and **`HATFormer 01_plus_blank_gate`** (`CER-selected` chooses lower line CER; `WER-selected` chooses lower line WER).


### Oracle Ceilings, Paired Win/Tie/Loss & 2×2 Catastrophic Transition Matrix

| Book | Script | N | K CER | H CER | **Oracle CER** | **Oracle Ortho CER** | K WER | H WER | **CER-Sel Oracle WER** | **WER-Sel Ceiling** | K-Cat | H-Cat | **Oracle Cat** | K-Only Cat | H-Only Cat | Both Cat | Neither | CER W/T/L (K/Tie/H) | WER W/T/L (K/Tie/H) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 3,661 | 27.23% | 36.39% | **26.05%** | **24.25%** | 62.51% | 73.38% | **60.43%** | **59.04%** | 147 | 528 | **111** | 36 | 417 | 111 | 3097 | `2833/209/619` | `2226/677/758` |
| `book_05` | Ruq'ah | 2,027 | 10.43% | 11.75% | **7.79%** | **6.09%** | 31.98% | 36.56% | **27.35%** | **26.67%** | 56 | 14 | **10** | 46 | 4 | 10 | 1967 | `1296/201/530` | `1144/335/548` |
| `book_09` | Naskh | 1,056 | 19.84% | 23.73% | **17.97%** | **15.95%** | 48.69% | 53.18% | **45.28%** | **44.22%** | 39 | 30 | **24** | 15 | 6 | 24 | 1011 | `692/84/280` | `552/176/328` |
| **OVERALL** | Combined | 6,744 | 19.86% | 25.22% | **18.03%** | **16.22%** | 49.25% | 56.67% | **46.02%** | **44.94%** | 242 | 572 | **145** | 97 | 427 | 145 | 6075 | `4821/494/1429` | `3922/1188/1634` |

### Inter-Model Disagreement Thresholds & Step 3 Router Scoping (`inter_model_cer` = `LevDist(K_T1A, H_T1A) / max_len`)

> Total Oracle character-edit reduction over Kraken Tier 1A ValOpt = **7,543** chars (word-edit reduction = **3,515** words; lines where `HATFormer 01_plus_blank_gate` strictly beats Kraken CER = **1,429 / 6,744**).


#### Cumulative Disagreement Thresholds (> {0.15, 0.25, 0.35})

| Stratum | Lines ($N$) | % Test | K-T1A CER | H CER | Oracle CER | Within-Stratum Gain (pp) | Char Edits Saved | **% Total Oracle Edit Gain** | **% Total Word Gain** | **% of Total H-Wins** | Stratum H-Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Low Agreement <= 0.15 | 1,763 | 26.1% | 8.01% | 10.18% | 7.20% | +0.81 pp | 1,013 | **13.4%** | **21.0%** | 28.8% (412/1429) | 23.4% (412/1763) |
| Disagree > 0.15 | 4,981 | 73.9% | 25.04% | 31.79% | 22.76% | +2.28 pp | 6,530 | **86.6%** | **79.0%** | 71.2% (1017/1429) | 20.4% (1017/4981) |
| Disagree > 0.25 | 3,344 | 49.6% | 29.83% | 37.57% | 26.88% | +2.95 pp | 5,347 | **70.9%** | **57.0%** | 47.3% (676/1429) | 20.2% (676/3344) |
| Disagree > 0.35 | 1,676 | 24.9% | 34.73% | 42.93% | 30.43% | +4.30 pp | 3,805 | **50.4%** | **33.9%** | 24.1% (344/1429) | 20.5% (344/1676) |

#### Disjoint Disagreement Buckets

| Stratum | Lines ($N$) | % Test | K-T1A CER | H CER | Oracle CER | Within-Stratum Gain (pp) | Char Edits Saved | **% Total Oracle Edit Gain** | **% Total Word Gain** | **% of Total H-Wins** | Stratum H-Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [0.00, 0.15] Low | 1,763 | 26.1% | 8.01% | 10.18% | 7.20% | +0.81 pp | 1,013 | **13.4%** | **21.0%** | 28.8% (412/1429) | 23.4% (412/1763) |
| (0.15, 0.25] Moderate | 1,637 | 24.3% | 16.80% | 21.86% | 15.68% | +1.12 pp | 1,183 | **15.7%** | **22.0%** | 23.9% (341/1429) | 20.8% (341/1637) |
| (0.25, 0.35] High | 1,668 | 24.7% | 25.17% | 32.47% | 23.51% | +1.66 pp | 1,542 | **20.4%** | **23.0%** | 23.2% (332/1429) | 19.9% (332/1668) |
| > 0.35 Very High | 1,676 | 24.9% | 34.73% | 42.93% | 30.43% | +4.30 pp | 3,805 | **50.4%** | **33.9%** | 24.1% (344/1429) | 20.5% (344/1676) |

---

## 5B. Post-Tier-1A Oracle Ceiling, 2×2 Matrix & Disagreement Scoping (Kraken ValOpt + HATFormer ValOpt)

> Paired complementarity between **`Kraken Tier 1A ValOpt`** and **`HATFormer 05_loopfb`** (`CER-selected` chooses lower line CER; `WER-selected` chooses lower line WER).


### Oracle Ceilings, Paired Win/Tie/Loss & 2×2 Catastrophic Transition Matrix

| Book | Script | N | K CER | H CER | **Oracle CER** | **Oracle Ortho CER** | K WER | H WER | **CER-Sel Oracle WER** | **WER-Sel Ceiling** | K-Cat | H-Cat | **Oracle Cat** | K-Only Cat | H-Only Cat | Both Cat | Neither | CER W/T/L (K/Tie/H) | WER W/T/L (K/Tie/H) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 3,661 | 27.23% | 36.54% | **26.08%** | **24.29%** | 62.51% | 73.94% | **60.57%** | **59.21%** | 147 | 527 | **109** | 38 | 418 | 109 | 3096 | `2844/217/600` | `2266/662/733` |
| `book_05` | Ruq'ah | 2,027 | 10.43% | 11.72% | **7.80%** | **6.10%** | 31.98% | 36.64% | **27.40%** | **26.71%** | 56 | 12 | **9** | 47 | 3 | 9 | 1968 | `1297/203/527` | `1140/341/546` |
| `book_09` | Naskh | 1,056 | 19.84% | 23.53% | **17.92%** | **15.90%** | 48.69% | 53.09% | **45.24%** | **44.22%** | 39 | 27 | **23** | 16 | 4 | 23 | 1013 | `698/78/280` | `555/175/326` |
| **OVERALL** | Combined | 6,744 | 19.86% | 25.24% | **18.04%** | **16.23%** | 49.25% | 56.94% | **46.09%** | **45.03%** | 242 | 566 | **141** | 101 | 425 | 141 | 6077 | `4839/498/1407` | `3961/1178/1605` |

### Inter-Model Disagreement Thresholds & Step 3 Router Scoping (`inter_model_cer` = `LevDist(K_T1A, H_T1A) / max_len`)

> Total Oracle character-edit reduction over Kraken Tier 1A ValOpt = **7,497** chars (word-edit reduction = **3,441** words; lines where `HATFormer 05_loopfb` strictly beats Kraken CER = **1,407 / 6,744**).


#### Cumulative Disagreement Thresholds (> {0.15, 0.25, 0.35})

| Stratum | Lines ($N$) | % Test | K-T1A CER | H CER | Oracle CER | Within-Stratum Gain (pp) | Char Edits Saved | **% Total Oracle Edit Gain** | **% Total Word Gain** | **% of Total H-Wins** | Stratum H-Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Low Agreement <= 0.15 | 1,773 | 26.3% | 8.04% | 10.19% | 7.23% | +0.81 pp | 1,023 | **13.6%** | **21.6%** | 29.5% (415/1407) | 23.4% (415/1773) |
| Disagree > 0.15 | 4,971 | 73.7% | 25.06% | 31.87% | 22.80% | +2.26 pp | 6,474 | **86.4%** | **78.4%** | 70.5% (992/1407) | 20.0% (992/4971) |
| Disagree > 0.25 | 3,334 | 49.4% | 29.84% | 37.62% | 26.91% | +2.94 pp | 5,310 | **70.8%** | **56.3%** | 46.8% (658/1407) | 19.7% (658/3334) |
| Disagree > 0.35 | 1,661 | 24.6% | 34.80% | 42.81% | 30.43% | +4.37 pp | 3,825 | **51.0%** | **34.1%** | 24.4% (343/1407) | 20.7% (343/1661) |

#### Disjoint Disagreement Buckets

| Stratum | Lines ($N$) | % Test | K-T1A CER | H CER | Oracle CER | Within-Stratum Gain (pp) | Char Edits Saved | **% Total Oracle Edit Gain** | **% Total Word Gain** | **% of Total H-Wins** | Stratum H-Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [0.00, 0.15] Low | 1,773 | 26.3% | 8.04% | 10.19% | 7.23% | +0.81 pp | 1,023 | **13.6%** | **21.6%** | 29.5% (415/1407) | 23.4% (415/1773) |
| (0.15, 0.25] Moderate | 1,637 | 24.3% | 16.86% | 21.99% | 15.76% | +1.11 pp | 1,164 | **15.5%** | **22.2%** | 23.7% (334/1407) | 20.4% (334/1637) |
| (0.25, 0.35] High | 1,673 | 24.8% | 25.20% | 32.77% | 23.61% | +1.59 pp | 1,485 | **19.8%** | **22.1%** | 22.4% (315/1407) | 18.8% (315/1673) |
| > 0.35 Very High | 1,661 | 24.6% | 34.80% | 42.81% | 30.43% | +4.37 pp | 3,825 | **51.0%** | **34.1%** | 24.4% (343/1407) | 20.7% (343/1661) |

---

## 6. Reproducible Diagnostic Table — Why Zero-Training Decoding Saturates on HATFormer Maghrebi (`book_03`)

> Classification follows `EDA/comparative_eda.classify_catastrophic_line` (`short_line`: GT `< 20` chars; `severe_truncation`: len ratio `< 0.40`; `severe_hallucination`: len ratio `> 1.60` or insertion share `> 60%`; `random_noise_or_garbled`: garbled score `> 0.40`; remainder: `plausible_but_wrong_arabic`). Checked under both `comparative_eda.py` loop regex and `tier1a_decode.py` 4-rule loop detector.


### HATFormer `01_plus_blank_gate` (= Baseline `B=3`, 528 Cat lines)

- **Catastrophic Maghrebi lines (`CER > 50%`):** **528 / 3661** (`14.42%`)

- **Corpus-wide Maghrebi loop incidence (`tier1a_decode` 4-rule):** **7 / 3661** (`0.19%`), accounting for **91 / 18,745** (**0.49% < 1%**) of Maghrebi raw character insertions.

- **Corpus-wide Maghrebi loop incidence (`comparative_eda` regex):** **8 / 3661** (`0.22%`), accounting for **88 / 18,745** (**0.47%**) of Maghrebi raw character insertions.


| Failure Category | Count (4-Rule Loop) | % of Cat Lines | Count (`EDA` Regex) | % of Cat (`EDA`) | % of All `book_03` (3,661) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `plausible_but_wrong_arabic` | 451 | 85.42% | 451 | 85.42% | 12.32% |
| `short_line` | 41 | 7.77% | 41 | 7.77% | 1.12% |
| `severe_hallucination` | 26 | 4.92% | 27 | 5.11% | 0.71% |
| `severe_truncation` | 6 | 1.14% | 6 | 1.14% | 0.16% |
| `repetition_loop` | 4 | 0.76% | 3 | 0.57% | 0.11% |

> **Saturation Finding:** **`451/528` (85.42%)** of catastrophic Maghrebi outputs are **`plausible_but_wrong_arabic`** (grammatically fluent Arabic words of normal length without repetition loops), while total non-loop failures represent `524/528` (99.24%). Because beam decoding cannot override a miscalibrated encoder-decoder conditional visual likelihood on Maghrebi ligatures, decoding hyperparameters (`LP`, `RP`, `LoopFB`) saturate at `~36.4%–36.5%` CER and motivate cross-model Kraken↔HATFormer arbitration in Step 3.


### HATFormer `05_bg_wc_lp1.0_rp1.15_loopfb` (Post-Tier 1A, 527 Cat lines)

- **Catastrophic Maghrebi lines (`CER > 50%`):** **527 / 3661** (`14.39%`)

- **Corpus-wide Maghrebi loop incidence (`tier1a_decode` 4-rule):** **5 / 3661** (`0.14%`), accounting for **52 / 19,496** (**0.27% < 1%**) of Maghrebi raw character insertions.

- **Corpus-wide Maghrebi loop incidence (`comparative_eda` regex):** **6 / 3661** (`0.16%`), accounting for **54 / 19,496** (**0.28%**) of Maghrebi raw character insertions.


| Failure Category | Count (4-Rule Loop) | % of Cat Lines | Count (`EDA` Regex) | % of Cat (`EDA`) | % of All `book_03` (3,661) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `plausible_but_wrong_arabic` | 455 | 86.34% | 455 | 86.34% | 12.43% |
| `short_line` | 41 | 7.78% | 41 | 7.78% | 1.12% |
| `severe_hallucination` | 27 | 5.12% | 27 | 5.12% | 0.74% |
| `severe_truncation` | 2 | 0.38% | 2 | 0.38% | 0.05% |
| `repetition_loop` | 2 | 0.38% | 2 | 0.38% | 0.05% |

> **Saturation Finding:** **`455/527` (86.34%)** of catastrophic Maghrebi outputs are **`plausible_but_wrong_arabic`** (grammatically fluent Arabic words of normal length without repetition loops), while total non-loop failures represent `525/527` (99.62%). Because beam decoding cannot override a miscalibrated encoder-decoder conditional visual likelihood on Maghrebi ligatures, decoding hyperparameters (`LP`, `RP`, `LoopFB`) saturate at `~36.4%–36.5%` CER and motivate cross-model Kraken↔HATFormer arbitration in Step 3.

