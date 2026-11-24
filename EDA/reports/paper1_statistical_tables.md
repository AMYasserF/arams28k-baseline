# Paper 1 — Statistical Audit & Benchmark Hygiene Tables

**Project:** AraMS-28k Benchmark Audit & Zero-Training Fix (Paper 1, Path B)  
**Official Test Split:** $N = 6,744$ (`book_03` $n=3,661$, `book_05` $n=2,027$, `book_09` $n=1,056$)  
**Artifact-Adjusted Clean Stratum:** $N = 6,721$ (excludes 20 lines on `book_09_page_034` + 3 verified blank crops)  
**Models:** Kraken (1D-CNN + BiLSTM + CTC) vs. HATFormer (`BEiT-base (ViT-patch16) + BlockProcessor + RoBERTa-base Arabic BBPE Decoder`, `beams=3`)

---

## 1. Paired Bootstrap 95% Confidence Intervals ($n = 10,000$, `seed=42`)

> **Definition:** $\Delta = \text{HATFormer} - \text{Kraken}$ in percentage points (negative $\Delta$ means HATFormer has lower error). Corpus metrics preserve character/word weighting inside each bootstrap resample.


### Strict Corpus CER (%)

| Book | Script | Kraken | Kraken 95% CI | HATFormer | HATFormer 95% CI | $\Delta$ (HAT−K) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 30.85% | [30.50%, 31.20%] | 36.39% | [35.97%, 36.82%] | **+5.54** | [+5.20, +5.89] |
| `book_05` | Ruq'ah | 11.25% | [10.80%, 11.73%] | 11.75% | [11.43%, 12.09%] | **+0.50** | [+0.06, +0.93] |
| `book_09` | Naskh | 22.11% | [21.36%, 22.90%] | 23.72% | [22.93%, 24.53%] | **+1.62** | [+1.05, +2.20] |
| **OVERALL** | Combined | 22.22% | [21.89%, 22.58%] | 25.22% | [24.85%, 25.60%] | **+2.99** | [+2.73, +3.25] |

### Ortho-Folded Corpus CER (%)

| Book | Script | Kraken | Kraken 95% CI | HATFormer | HATFormer 95% CI | $\Delta$ (HAT−K) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 28.76% | [28.44%, 29.10%] | 33.81% | [33.40%, 34.22%] | **+5.05** | [+4.72, +5.38] |
| `book_05` | Ruq'ah | 9.19% | [8.74%, 9.65%] | 9.31% | [9.00%, 9.63%] | **+0.12** | [-0.31, +0.54] |
| `book_09` | Naskh | 19.76% | [19.02%, 20.55%] | 21.30% | [20.52%, 22.11%] | **+1.54** | [+0.98, +2.10] |
| **OVERALL** | Combined | 20.10% | [19.76%, 20.45%] | 22.72% | [22.36%, 23.09%] | **+2.62** | [+2.37, +2.87] |

### True Corpus WER (%)

| Book | Script | Kraken | Kraken 95% CI | HATFormer | HATFormer 95% CI | $\Delta$ (HAT−K) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 75.61% | [74.98%, 76.24%] | 73.38% | [72.66%, 74.08%] | **-2.23** | [-2.89, -1.58] |
| `book_05` | Ruq'ah | 38.28% | [37.38%, 39.19%] | 36.57% | [35.84%, 37.29%] | **-1.71** | [-2.61, -0.82] |
| `book_09` | Naskh | 58.30% | [57.16%, 59.47%] | 53.18% | [51.98%, 54.37%] | **-5.12** | [-6.19, -4.06] |
| **OVERALL** | Combined | 59.31% | [58.70%, 59.94%] | 56.68% | [56.08%, 57.29%] | **-2.63** | [-3.12, -2.16] |

### True Mean WER (%)

| Book | Script | Kraken | Kraken 95% CI | HATFormer | HATFormer 95% CI | $\Delta$ (HAT−K) | $\Delta$ 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | 77.42% | [76.56%, 78.40%] | 75.54% | [74.51%, 76.72%] | **-1.88** | [-2.64, -1.12] |
| `book_05` | Ruq'ah | 39.01% | [38.03%, 40.00%] | 37.50% | [36.56%, 38.48%] | **-1.51** | [-2.52, -0.48] |
| `book_09` | Naskh | 59.41% | [57.71%, 61.47%] | 54.57% | [52.70%, 56.83%] | **-4.85** | [-5.94, -3.75] |
| **OVERALL** | Combined | 63.06% | [62.33%, 63.84%] | 60.83% | [60.02%, 61.69%] | **-2.23** | [-2.76, -1.70] |

---

## 2. Holm-Bonferroni Adjusted P-Values (Two-Sided Paired Wilcoxon)

> Family-wise error rate control across all $m = 8$ paired comparisons (4 splits $\times$ 2 metrics: line CER and True Word-Level WER).

| Book | Script | Metric | Wilcoxon $W$ | Median $\Delta$ (HF−K) | Raw $p$-value | Holm-Bonferroni Adj $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `book_03` | Maghrebi | CER | 1,317,914 | +5.36 pp | `7.509e-171` | **`5.256e-170`** |
| `book_03` | Maghrebi | True WER | 1,889,024 | +0.00 pp | `8.110e-11` | **`1.622e-10`** |
| `book_05` | Ruq'ah | CER | 598,238 | +1.37 pp | `5.903e-21` | **`2.952e-20`** |
| `book_05` | Ruq'ah | True WER | 661,984 | +0.00 pp | `1.617e-01` | **`1.617e-01`** |
| `book_09` | Naskh | CER | 183,231 | +1.37 pp | `1.378e-11` | **`4.135e-11`** |
| `book_09` | Naskh | True WER | 125,027 | -5.88 pp | `1.096e-19` | **`4.385e-19`** |
| **OVERALL** | Combined | CER | 5,557,800 | +2.78 pp | `2.755e-182` | **`2.204e-181`** |
| **OVERALL** | Combined | True WER | 6,452,854 | +0.00 pp | `4.771e-21` | **`2.863e-20`** |

---

## 3. De-Confounded `AraMS-Restore` Physical Degradation Analysis (Naskh)

> Excludes all 20 shifted lines of `book_09_page_034` ($1,056 - 20 = 1,036$: **$n_{\text{dam}} = 177$** official `AraMS-Restore` lines vs. **$n_{\text{clean}} = 859$** undamaged Naskh lines). Note: `book_09_line_006514` (blank margin crop) belongs to the clean subset.


### Error Distributions (De-Confounded Naskh)

| Subset | $N$ | Kraken Mean CER | Kraken Med CER | HAT Mean CER | HAT Med CER | Kraken Mean WER | Kraken Corp WER | HAT Mean WER | HAT Corp WER | Kraken Cat (>50%) | HAT Cat (>50%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Damaged (n=177) | 177 | 23.13% | 21.92% | 25.95% | 25.35% | 60.07% | 59.64% | 56.84% | 56.45% | 2 (1.13%) | 2 (1.13%) |
| Clean (n=859) | 859 | 20.51% | 19.28% | 21.94% | 20.83% | 57.03% | 56.93% | 51.55% | 51.27% | 9 (1.05%) | 8 (0.93%) |

### Non-Parametric Tests & Binary Catastrophic Power ($n_{\text{dam}}=177$, $n_{\text{clean}}=859$)

| Test | Model | Statistic | Two-Sided $p$-value | One-Sided $p$ (`dam > clean`) | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Mann-Whitney U (Line CER) | Kraken | $U = 89,800.0$ | `1.4406e-04` | `7.2030e-05` | Systematic upward CER shift |
| Mann-Whitney U (Line CER) | HATFormer | $U = 93,750.0$ | `1.0042e-06` | `5.0210e-07` | Systematic upward CER shift |
| Mann-Whitney U (True Line WER) | Kraken | $U = 84,380.5$ | `0.0211` | — | True WER distribution shift |
| Mann-Whitney U (True Line WER) | HATFormer | $U = 88,427.5$ | `0.0006` | — | Significant True WER upward shift |
| Fisher's Exact Test (CER > 50%) | Kraken | $\text{OR} = 1.079$ | `1.0000` | — | 2/177 (1.13%) vs 9/859 (1.05%) — not enriched |
| Fisher's Exact Test (CER > 50%) | HATFormer | $\text{OR} = 1.216$ | `0.6831` | — | 2/177 (1.13%) vs 8/859 (0.93%) — not enriched |

> **Analytic Power Analysis ($\alpha = 0.05$, two-sided, $n_{\text{dam}}=177$, $n_{\text{clean}}=859$):**
> - **Observed Post-Hoc Power:** Kraken = **5.69%** ($p_{\text{dam}}=1.13\%, p_{\text{clean}}=1.05\%$), HATFormer = **7.29%** ($p_{\text{dam}}=1.13\%, p_{\text{clean}}=0.93\%$).
> - **Minimum Detectable Odds Ratio at 80% Power:** Kraken $\text{OR} \ge 4.37$ ($p_{\text{dam}} \ge 4.42\%$), HATFormer $\text{OR} \ge 4.64$ ($p_{\text{dam}} \ge 4.18\%$). Because baseline catastrophic collapse on undamaged Naskh is rare (~1.05% once `book_09_page_034` is removed), binary Fisher testing has low power for small effects ($\text{OR} \le 2.0$), but >80% power to detect catastrophic surges $\ge 4.4\%$.


> **Sensitivity Check ($n_{\text{clean}} = 858$, also excluding blank crop `book_09_line_006514`):** Clean Mean/Med CER drops slightly to Kraken `20.42% / 19.28%` (True Mean/Corp WER `56.98% / 56.86%`, Cat `8/858 = 0.93%`) and HATFormer `21.86% / 20.83%` (True Mean/Corp WER `51.49% / 51.20%`, Cat `7/858 = 0.82%`); two-sided Mann-Whitney CER stays highly significant (Kraken $U=89,800.0, p=1.2836e-04$; HATFormer $U=93,750.0, p=8.6339e-07$) and Fisher catastrophic enrichment stays non-significant (Kraken $\text{OR}=1.214, p=0.6833$; HATFormer $\text{OR}=1.389, p=0.6560$).

| Target True OR | Kraken $p_{\text{dam}}$ | Kraken Power | HATFormer $p_{\text{dam}}$ | HATFormer Power |
| :---: | :---: | :---: | :---: | :---: |
| 1.5x | 1.56% | 12.7% | 1.39% | 12.2% |
| 2.0x | 2.07% | 25.8% | 1.85% | 24.4% |
| 3.0x | 3.08% | 54.1% | 2.74% | 50.9% |
| 4.0x | 4.06% | 74.6% | 3.62% | 71.1% |
| 5.0x | 5.03% | 86.8% | 4.49% | 83.8% |

---

## 4. Dual Test-Set Comparison: Official vs. Clean Strata

> - **Artifact-Adjusted Clean Stratum ($N=6,721$):** excludes 20 `book_09_page_034` lines + 3 verified blank crops.
> - **Full Hygiene-Deduplicated Stratum ($N=6,673$):** excludes all 29 rows in `arams28k_benchmark_audit_artifact.csv` + all 42 duplicated lines on `book_03_page_103` and `book_03_page_104`.

| Split | $N$ | K Corp CER | H Corp CER | K Ortho CER | H Ortho CER | K Corp WER | H Corp WER | K Mean WER | H Mean WER | K Cat | H Cat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Official Overall (N=6,744) | 6,744 | 22.22% | 25.22% | 20.10% | 22.72% | 59.31% | 56.68% | 63.06% | 60.83% | 272 | 572 |
| Clean Stratum Overall (N=6,721) | 6,721 | 21.97% | 24.98% | 19.84% | 22.47% | 59.11% | 56.46% | 62.76% | 60.46% | 249 | 549 |
| Official Normal >=20ch (N=6,649) | 6,649 | 22.11% | 25.08% | 19.99% | 22.58% | 59.18% | 56.53% | 62.11% | 59.71% | 218 | 520 |
| Official Short <20ch (N=95) | 95 | 106.43% | 126.65% | 103.31% | 124.45% | 140.80% | 154.40% | 129.04% | 138.60% | 54 | 52 |
| Clean Normal >=20ch (N=6,629) | 6,629 | 21.89% | 24.88% | 19.76% | 22.37% | 59.01% | 56.34% | 61.99% | 59.58% | 198 | 500 |
| Clean Short <20ch (N=92) | 92 | 89.37% | 106.69% | 86.02% | 104.33% | 125.00% | 134.17% | 118.57% | 123.55% | 51 | 49 |
| Official — Maghrebi (book_03) | 3,661 | 30.85% | 36.39% | 28.76% | 33.81% | 75.61% | 73.38% | 77.42% | 75.54% | 199 | 528 |
| Clean Stratum — Maghrebi (book_03) | 3,661 | 30.85% | 36.39% | 28.76% | 33.81% | 75.61% | 73.38% | 77.42% | 75.54% | 199 | 528 |
| Official — Ruq'ah (book_05) | 2,027 | 11.25% | 11.75% | 9.19% | 9.31% | 38.28% | 36.57% | 39.01% | 37.50% | 42 | 14 |
| Clean Stratum — Ruq'ah (book_05) | 2,025 | 11.21% | 11.71% | 9.14% | 9.26% | 38.25% | 36.53% | 38.95% | 37.29% | 40 | 12 |
| Official — Naskh (book_09) | 1,056 | 22.11% | 23.72% | 19.76% | 21.30% | 58.30% | 53.18% | 59.41% | 54.57% | 31 | 30 |
| Clean Stratum — Naskh (book_09) | 1,035 | 20.87% | 22.52% | 18.51% | 20.08% | 57.32% | 52.06% | 57.51% | 52.40% | 10 | 9 |
| Hygiene-Deduplicated Overall (N=6,673) | 6,673 | 21.88% | 24.88% | 19.75% | 22.37% | 59.01% | 56.35% | 62.65% | 60.34% | 241 | 537 |
| Hygiene Normal >=20ch (N=6,581) | 6,581 | 21.80% | 24.78% | 19.67% | 22.27% | 58.91% | 56.24% | 61.87% | 59.46% | 190 | 488 |
| Hygiene Short <20ch (N=92) | 92 | 89.37% | 106.69% | 86.02% | 104.33% | 125.00% | 134.17% | 118.57% | 123.55% | 51 | 49 |
| Hygiene Clean — Maghrebi (book_03) | 3,614 | 30.77% | 36.33% | 28.69% | 33.75% | 75.60% | 73.37% | 77.42% | 75.54% | 192 | 517 |
| Hygiene Clean — Ruq'ah (book_05) | 2,024 | 11.17% | 11.67% | 9.11% | 9.23% | 38.22% | 36.50% | 38.92% | 37.26% | 39 | 11 |
| Hygiene Clean — Naskh (book_09) | 1,035 | 20.87% | 22.52% | 18.51% | 20.08% | 57.32% | 52.06% | 57.51% | 52.40% | 10 | 9 |

---

## 5. Metadata, Short-Line Taxonomy & Image Hygiene Audit

### 5A. Short Lines ($<20$ chars, $n = 95$) Metadata & Crop Geometry Breakdown

- **`metadata.csv` Schema Limitation:** All 6,744 test rows carry `line_type = 'main'` (and polygon vertex coordinates are not stored in `metadata.csv`). However, combining `line_idx` position on page, `crop_src`, and measured pixel width cleanly separates marginal catchwords from body-line artifacts.

- **Top-of-page Headers (`line_idx == 0`):** **0 / 95** short lines occur at line index 0.

- **Likely Lower-Left Catchwords (`line_idx == page_max_idx`, `crop_src == 'bbox'`, width $\le 250$ px):** **87 / 95 (91.6%)** — 52 in Maghrebi (`book_03`), 31 in Ruq'ah (`book_05`), 4 in Naskh (`book_09`), plus 1 corrupt $4\times 4$ px blank `bbox` crop (`book_05_line_004216`).


| Short-Line Category | Book | Count | Median Width (px) | Width Range (px) | Kraken Mean CER | HAT Mean CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Blank / Corrupt Crop | `book_05` | 1 | 4 | 4–4 | 100.0% | 266.7% |
| Likely Lower-Left Catchword (bbox, final line) | `book_03` | 52 | 107 | 57–203 | 68.9% | 86.8% |
| Likely Lower-Left Catchword (bbox, final line) | `book_05` | 31 | 72 | 39–104 | 34.9% | 37.3% |
| Likely Lower-Left Catchword (bbox, final line) | `book_09` | 4 | 76 | 63–87 | 50.6% | 45.8% |
| Page-Shifted Full Body Line | `book_09` | 2 | 874 | 869–878 | 396.7% | 436.7% |
| Truncated GT / Full Body Tail | `book_03` | 5 | 635 | 629–648 | 497.9% | 533.1% |

### 5B. 64-Bit dHash Near-Duplicate Scan (`Hamming <= 6`)

> Standalone full-page scans are not stored in `data/AraMS-28k-HTR/images`; hashing was executed across all 6,744 test-split line image crops.

- **Images hashed:** 6,744
- **Exact dHash collisions (Hamming = 0):** 0
- **Near-duplicate pairs (Hamming $\le 6$):** 56 (**19 verified duplicate scans** with GT similarity $\ge 70\%$, 37 incidental low-res stroke collisions across unrelated text)

#### Verified Duplicate Bifolio Scan Discovery (`book_03_page_101/102` $\leftrightarrow$ `book_03_page_103/104`)

Cross-checking the dHash hits against ground-truth text revealed **two duplicate manuscript page spreads (84 test lines total; 42 duplicated lines)** in `book_03` (`test` split), where the same physical manuscript spread appears twice with slightly shifted polygon boundaries and minor annotator transcription differences:

| Folio Pair | Lines | Identical Line GTs | Mean Inter-Page GT Diff | Aligned dHash Hits ($\le 6$) |
| :--- | :---: | :---: | :---: | :---: |
| `book_03_page_101` $\leftrightarrow$ `book_03_page_103` | 21 vs 21 | 17/21 | 1.86% | 8 |
| `book_03_page_102` $\leftrightarrow$ `book_03_page_104` | 21 vs 21 | 14/21 | 5.42% | 11 |

<details><summary>All 19 Verified Duplicate-Folio dHash Line Hits (Hamming &le; 6)</summary>

| Image A | Image B | Hamming | GT Similarity |
| :--- | :--- | :---: | :---: |
| `book_03__book_03_page_101_line005.png` | `book_03__book_03_page_103_line005.png` | 4 | 100.0% |
| `book_03__book_03_page_101_line007.png` | `book_03__book_03_page_103_line007.png` | 5 | 100.0% |
| `book_03__book_03_page_101_line011.png` | `book_03__book_03_page_103_line011.png` | 6 | 100.0% |
| `book_03__book_03_page_101_line014.png` | `book_03__book_03_page_103_line014.png` | 6 | 90.0% |
| `book_03__book_03_page_101_line015.png` | `book_03__book_03_page_103_line015.png` | 4 | 98.0% |
| `book_03__book_03_page_101_line016.png` | `book_03__book_03_page_103_line016.png` | 5 | 93.5% |
| `book_03__book_03_page_101_line017.png` | `book_03__book_03_page_103_line017.png` | 5 | 100.0% |
| `book_03__book_03_page_101_line018.png` | `book_03__book_03_page_103_line018.png` | 4 | 100.0% |
| `book_03__book_03_page_102_line000.png` | `book_03__book_03_page_104_line000.png` | 2 | 82.5% |
| `book_03__book_03_page_102_line001.png` | `book_03__book_03_page_104_line001.png` | 4 | 100.0% |
| `book_03__book_03_page_102_line004.png` | `book_03__book_03_page_104_line004.png` | 5 | 100.0% |
| `book_03__book_03_page_102_line005.png` | `book_03__book_03_page_104_line005.png` | 6 | 100.0% |
| `book_03__book_03_page_102_line007.png` | `book_03__book_03_page_104_line007.png` | 5 | 100.0% |
| `book_03__book_03_page_102_line008.png` | `book_03__book_03_page_104_line008.png` | 6 | 75.0% |
| `book_03__book_03_page_102_line011.png` | `book_03__book_03_page_104_line011.png` | 5 | 100.0% |
| `book_03__book_03_page_102_line015.png` | `book_03__book_03_page_104_line015.png` | 2 | 88.4% |
| `book_03__book_03_page_102_line018.png` | `book_03__book_03_page_104_line018.png` | 5 | 100.0% |
| `book_03__book_03_page_102_line019.png` | `book_03__book_03_page_104_line019.png` | 3 | 100.0% |
| `book_03__book_03_page_102_line020.png` | `book_03__book_03_page_104_line020.png` | 2 | 93.0% |

</details>


### 5C. Upgraded Benchmark Audit Artifact (`EDA/arams28k_benchmark_audit_artifact.csv`)

- **Total Audited Artifact Rows:** 29

| `noise_category` | Count | Scope |
| :--- | :---: | :--- |
| `page_shift` | 23 | All 20 lines of `book_09_page_034` (5-line shift) + displaced folio lines (`book_03_page_152`, `book_05_page_025`) |
| `partial_gt_truncation` | 3 | GT omits visible continuation read consistently by both Kraken & HATFormer (`book_03`) |
| `blank_crop` | 3 | Empty/corrupt paper-margin crops (`book_09_line_006514`, `book_05_line_004741`, `book_05_line_004216`) |