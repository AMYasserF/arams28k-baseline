[← Previous: Chapter 01](01_starting_point_and_baselines.md) | [Master Index](00_INDEX.md) | [Next: Chapter 03 — Architectural Diagnosis →](03_architectural_diagnosis_and_stats.md)

---

# Chapter 02 — Benchmark Hygiene Audit (Step 1A)

> **Chapter Takeaway:** Auditing the official `6,744`-line AraMS-28k test set before trusting any error table uncovered **`29` defective label/crop rows** (a systematic 5-line ground-truth shift on `book_09_page_034` explaining `83.3%` of Naskh shared failures, `3` empty paper margin crops, `3` single-line shifts, and `3` truncated GT references) plus **`42` duplicated Maghrebi test lines** (`book_03_page_101/102` vs. `103/104`) and a **`100%`-Naskh validation split**. Removing these `71` corrupted/duplicate entries establishes our leak-free **Full Hygiene-Deduplicated Stratum (`N = 6,673`)**.

---

## 1. Why We Did This (Reason & Question)

In Chapter 01, both completely unrelated architectures (**Kraken** and **HATFormer**) failed catastrophically (`CER > 50%`) on the exact same Naskh pages—most suspiciously `book_09_page_034`. When two independent models agree closely with *each other* (`inter_model_cer < 35%` — Character Error Rate between Kraken and HATFormer) yet both fail catastrophically against the ground-truth text (`CER > 65%`), the ground truth is likely wrong rather than the models. We also wanted to test whether the `6,744` test images contained duplicate scans or mislabeled non-body crops that could distort model rankings and bias statistical tests.

---

## 2. What We Did & How (Step-by-Step + Code & File References)

We implemented a repeatable, multi-layer benchmark hygiene pipeline in [`EDA/comparative_eda.py`](../../EDA/comparative_eda.py) (Section 11) and [`EDA/paper1_stats_audit.py`](../../EDA/paper1_stats_audit.py) (Sections 4–5):

1. **Inter-Model Agreement Noise Detector (`EDA/comparative_eda.py`):**
   - Flagged every test line where **both** Kraken (`>65%` CER) and HATFormer (`>65%` CER) failed severely against ground truth while strongly agreeing with *each other* (`inter_model_cer < 35%`).
   - Inspected every flagged line visually and against neighboring lines on the same manuscript page.
2. **Blank / Corrupt Crop Pixel Check (`EDA/paper1_stats_audit.py`):**
   - Measured actual pixel `(width × height)` geometry across all `6,744` image files on disk and inspected any crops where Kraken output `""` (empty string) or extreme short text occurred.
3. **64-bit `dHash` Duplicate Image Scan (`EDA/paper1_stats_audit.py` Section 5B):**
   - Computed a **dHash** (Difference Hash — a fast 64-bit visual fingerprint measuring horizontal brightness transitions on a `9×8` resized grayscale thumbnail) for all `6,744` test line images.
   - Flagged every image pair within **Hamming distance `<= 6` bits** (out of 64), then cross-checked their **GT** (Ground-Truth text) similarity (`>= 70%`) to separate true duplicate manuscript scans from accidental low-res shape coincidences.
4. **Metadata & Short-Line Audit (`EDA/paper1_stats_audit.py` Section 5A):**
   - Cross-checked `data/AraMS-28k-HTR/metadata.csv` fields (`line_type`, `line_idx` position on page, `crop_src` `bbox` vs. `poly`, and pixel width) across all `95` short lines (`< 20` GT chars).
5. **Dual-Stratum Protocol Creation:**
   - Exported all `29` verified label/crop artifacts to [`EDA/arams28k_benchmark_audit_artifact.csv`](../../EDA/arams28k_benchmark_audit_artifact.csv), and defined the **Full Hygiene-Deduplicated Stratum (`N = 6,673`)** excluding those `29` audit rows + the `42` duplicated folio lines (`71` excluded total).

---

## 3. Results & Visual Evidence

![Visual Evidence of Three AraMS-28k Official Test-Set Benchmark Hygiene Artifacts](assets/chart_02b_hygiene_visual_evidence.png)

> **Direct Source Crop Archive (`assets/samples/`):** Inspect the raw unannotated PNG crops behind the composite above:
> - **Shifted `book_09_page_034` lines (`line001` vs. `line006` vs. `line009`):** [`shift_line001`](assets/samples/shift_book_09_line_006264_line001.png) · [`shift_line006`](assets/samples/shift_book_09_line_006269_line006.png) · [`shift_line009`](assets/samples/shift_book_09_line_006272_line009.png)
> - **Blank paper/margin crops:** [`blank_004216`](assets/samples/blank_book_05_line_004216.png) (`4×4` px) · [`blank_004741`](assets/samples/blank_book_05_line_004741.png) (`27×66` px) · [`blank_006514`](assets/samples/blank_book_09_line_006514.png) (`20×57` px)
> - **Duplicate Maghrebi spread (`page_102` vs. `page_104`):** [`p102_line000`](assets/samples/dup_book_03_page_102_line000.png) ↔ [`p104_line000`](assets/samples/dup_book_03_page_104_line000.png) · [`p102_line015`](assets/samples/dup_book_03_page_102_line015.png) ↔ [`p104_line015`](assets/samples/dup_book_03_page_104_line015.png)

### Discovery 1 — Systematic 5-Line Ground-Truth Shift on `book_09_page_034` (`20` lines)
All `20` lines on `book_09_page_034` suffer from an annotation displacement (`20 / 24 = 83.3%` of all mutual Kraken+HATFormer Naskh failures). On lines 5–17, visual crop `line_idx = i` carries the ground truth of `i - 5`.  
Consider **Line 6 (`book_09_line_006269`, Image: `book_09__book_09_page_034_line006.png`)**:
- **Assigned Wrong GT (`89.6%` Kraken CER / `84.4%` HAT CER):** `لكثرة خصومتهم قال البخاري في جامعه توفي في ذي القعدة سنة ثمان واربعين ومائتين`
- **Actual Visual Ink in the Crop (= Line 1 GT):** `قال احمد بن صالح حدثت احمد بن حنبل بحديث زيد بن ثابت في بيع الثمار فاعجبه واشترى مني مثله`
- **Kraken Prediction:** `قال أحمد صالح حدثت أحمد برجنبل بحديثب زيد بن ثابت في سبيع الثمار فأعجبة أشبراني بي مثله`
- **HATFormer Prediction:** `قال أجدر صالح حدثت أهل زخبك بحديث زيد بن ثابت بني بيع النار فأعجيه واشترابي بن مثله`
Both models read the Arabic ink cleanly and agree closely with each other (`inter_model_cer = 27.6%`), yet the benchmark scores both as catastrophic failures! Analogous isolated page shifts occur on `book_03_page_152` (`2` lines) and `book_05_page_025` (`1` line), plus `3` truncated GT strings in `book_03`.

### Discovery 2 — Three Corrupt Blank Margin Crops (`book_05_line_004216`, `004741`, `book_09_line_006514`)
Three test "lines" contain zero Arabic text—only tiny blank paper patches (`4×4` px, `27×66` px, and `20×57` px):
- **Kraken** faithfully outputs **`""`** (empty string) on all three.
- **HATFormer** hallucinates high-frequency Islamic boilerplate (`صلى الله عليه وسلم` on both Ruq'ah blanks; `ونحو ذلك` on the Naskh blank) because its autoregressive decoder must emit tokens once cross-attention starts on non-zero noise! Because GT assigns real Arabic sentences to these blank patches, Kraken's honest `""` gets penalized `100%` CER while HATFormer's hallucination on `004741` is penalized `87.0%`.

### Discovery 3 — Duplicate Maghrebi Bifolio Spread (`42` duplicated lines)
Hashing all `6,744` images uncovered `56` near-duplicate pairs (`Hamming <= 6`), of which **`19` pairs (`>= 70%` GT similarity)** came from the **exact same two manuscript page spreads scanned twice** in the test set with slightly shifted polygon boundaries:

| Folio Pair | Lines per Folio | Identical Line GTs | Mean Inter-Page GT Diff | Aligned `dHash <= 6` Hits |
| :--- | :---: | :---: | :---: | :---: |
| `book_03_page_101` ↔ `book_03_page_103` | `21` vs. `21` | `17 / 21` | `1.86%` | `8` |
| `book_03_page_102` ↔ `book_03_page_104` | `21` vs. `21` | `14 / 21` | `5.42%` | `11` |

*(For example, `book_03_page_102_line015` and `book_03_page_104_line015` differ by only `2 / 64` dHash bits and share identical text `وقال فيه حديث حسن صحيح...` vs minor annotator omissions on `line015`, weighting Maghrebi errors twice in official counts.)*

### Discovery 4 — Structural Label & Validation Split Biases
1. **Unlabeled Marginal Catchwords (*taʿqība*):** In `metadata.csv`, all `27,969` rows carry `line_type = 'main'` even though **`87 / 95` (`91.6%`)** of short lines (`<20` chars) sit at `line_idx == page_max_idx` using tight `bbox` crops (`width <= 250` px: `52` Maghrebi, `31` Ruq'ah, `4` Naskh) — classic lower-left page-turn catchwords. Because denominator length `< 20` magnifies single-letter errors, these `95` lines average `86.83%` CER (Kraken Greedy) and `101.67%` CER (HATFormer Baseline).
2. **`100%`-Naskh Validation Split:** Checking [`split_used.json`](../../data/AraMS-28k-HTR/split_used.json) shows validation consists solely of `book_06` (`874` lines) and `book_11` (`612` lines)—zero Maghrebi or Ruq'ah validation lines exist! Any hyperparameter tuned purely on `val` risks overfitting to Naskh script conventions (as we prove later in Chapter 05).

### Summary Breakdown of `EDA/arams28k_benchmark_audit_artifact.csv` (`29` rows) & Dual Strata

| Stratum | Lines (`N`) | Excluded Scope | Kraken Greedy CER | HATFormer `B=3` CER | Kraken Corp WER | HATFormer Corp WER | K Cat | H Cat |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official Test Split** | `6,744` | Unmodified official benchmark | `22.22%` | `25.22%` | `59.31%` | `56.68%` | `272` | `572` |
| **Artifact Clean** | `6,721` | Excl. `20` `page_034` lines + `3` blanks (`paper1_stats_audit.py`) | `21.97%` | `24.98%` | `59.11%` | `56.46%` | `249` | `549` |
| **Full Hygiene-Deduplicated** | **`6,673`** | Excl. all **`29`** audit rows (`23` `page_shift`, `3` `blank_crop`, `3` `partial_gt_truncation`) + **`42`** dup folio lines | **`21.88%`** | **`24.88%`** | **`59.01%`** | **`56.35%`** | **`241`** | **`537`** |

---

## 4. Conclusion (What This Proves)

1. The official AraMS-28k test split contains `29` label/crop artifacts (`0.43%` of lines, yet `28–35` catastrophic failures) and `42` duplicated Maghrebi lines (`book_03_page_103/104`).
2. When those `71` distorted lines are removed, Naskh catastrophic collapse drops dramatically from `31 / 30` (`2.94% / 2.84%`) down to just **`10` Kraken / `9` HATFormer (`0.97% / 0.87%`)** — showing clean Naskh is actually highly reliable!
3. Every downstream experiment must report **both** the **Official `N=6,744`** split (for comparability with published AraMS-28k papers) and the **Full Hygiene-Deduplicated `N=6,673`** stratum (for clean scientific measurement).

---

## 5. How This Affects the Next Step

Having removed the confounding `book_09_page_034` label shift and identified exact ground-truth noise, we could now safely ask the central diagnostic questions of **Step 1B** in [Chapter 03](03_architectural_diagnosis_and_stats.md): **What truly drives the CER-vs-WER Paradox, how much does Arabic orthographic inconsistency inflate error, and does real physical page damage (`AraMS-Restore`) break Kraken or HATFormer?**

---

[← Previous: Chapter 01](01_starting_point_and_baselines.md) | [Master Index](00_INDEX.md) | [Next: Chapter 03 — Architectural Diagnosis →](03_architectural_diagnosis_and_stats.md)
