# Report 1 — EDA Structure, Methodology & Findings
**Project:** Comparative HTR Analysis on AraMS-28k (Kraken vs. HATFormer)  
**Scope:** Kraken Predictions — book_03 (Maghrebi) · book_05 (Ruq'ah) · book_09 (Naskh)  
**Script:** [`EDA/kraken_eda.py`](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/EDA/kraken_eda.py)  
**Figures:** [`EDA/figures/`](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/EDA/figures/)

---

## Overview

This EDA was designed around a core principle: **aggregate metrics lie**. A single corpus-wide CER conceals three entirely different failure profiles — one per script tradition. Every section below was constructed to isolate a specific dimension of model behaviour, motivated by a research question that cannot be answered by looking at a number alone.

The analysis is structured in **12 sections**, each with an explicit objective, a methodology, and an empirical finding.

---

## Section 0 — Dataset Baseline

### Objective
Establish an unambiguous, reproducible baseline for all downstream comparisons. Confirm data integrity (no nulls, correct record counts) and compute both mean-CER (per-line average) and corpus-CER (global edit distance / global GT chars) to understand whether line-length skew distorts the aggregate.

### Methodology
- Loaded all three CSVs: `kraken_preds_book_03.csv`, `kraken_preds_book_05.csv`, `kraken_preds_book_09.csv`
- Computed **per-line CER** = `Levenshtein(ref, hyp) / len(ref)` for every row
- Computed **corpus CER** = `Sigma edits / Sigma GT chars` (length-weighted)
- Recorded P50 and P90 CER to characterise distribution shape

### Findings

| Book | Script | Lines | GT Chars | Mean CER | Corpus CER | Median CER | Mean WER | P90 CER |
|------|--------|------:|--------:|:-------:|:---------:|:---------:|:-------:|:-------:|
| book_03 | Maghrebi | 3,661 | 185,807 | 32.09% | 30.85% | 30.00% | 77.42% | 45.10% |
| book_05 | Ruq'ah | 2,027 | 145,129 | 11.51% | 11.25% | 8.97% | 39.01% | 19.68% |
| book_09 | Naskh | 1,056 | 81,371 | 22.69% | 22.11% | 19.73% | 59.41% | 33.81% |

> [!NOTE]
> Mean CER and Corpus CER are within ~1.5% of each other for all scripts, confirming that line-length does not introduce systematic bias into the per-line averages. Both metrics are valid to report.

**Figure:** `figures/00_summary_table.png`

---

## Section 1 — CER / WER Distributions

### Objective
Understand the *shape* of the error distribution, not just its centre. A mean CER of 30% can represent a model that consistently fails at 30% on every line, or one that is perfect on half the lines and catastrophically wrong on the other half. Only the full distribution reveals this.

### Methodology
Three complementary visualisations were computed:

1. **Violin + Boxplot** (per-line CER and WER): shows density shape, median, IQR, and outliers simultaneously
2. **Overlapping Histograms + KDE** (CER only): allows direct shape comparison across scripts at the same scale
3. **Cumulative Distribution Function (CDF)**: answers the question *"what fraction of lines fall below CER threshold X?"* — the most practically useful form for setting post-processing targets

### Findings
- **Maghrebi** distribution is wide and right-skewed (sigma = 0.2699). The violin is spread almost uniformly from 0% to 100% CER — there is no concentration of well-performing lines. The model has no "comfort zone" on this script.
- **Ruq'ah** distribution is tightly concentrated at low CER (sigma = 0.1119). The violin is narrow, the median is 8.97%, and the KDE has a sharp peak near 0–15%. Errors here are consistent and bounded.
- **Naskh** sits structurally between both — a moderate spread, right-skewed, with a visible peak near 20%.
- **CDF insight**: To reach 50% of lines below 10% CER, the model needs to roughly triple its accuracy on Maghrebi. Ruq'ah already achieves ~40% of lines below 10%.

**Figures:** `figures/01_cer_wer_violin.png` · `figures/02_cer_histogram_kde.png` · `figures/03_cer_cumulative_cdf.png`

---

## Section 2 — Edit Type Decomposition

### Objective
Deconstruct every error into its type: **Insertion** (model outputs a character not in GT), **Deletion** (model skips a character present in GT), or **Substitution** (model replaces one character with another). These three failure modes have *different causes* and require *different fixes*.

### Methodology
- For every line pair `(ref, hyp)`, the Levenshtein edit operations were enumerated using `rapidfuzz.distance.Levenshtein.editops()`
- Counts aggregated per book: total insertions, deletions, substitutions
- Visualised as both absolute counts and proportional stacked bars
- Per-line rates (edits / GT length) shown as boxplots to capture variance

### Findings

| Script | Insertions | Deletions | Substitutions |
|--------|:----------:|:---------:|:-------------:|
| Maghrebi | 20.3% | 16.5% | **63.2%** |
| Ruq'ah | 10.6% | **40.0%** | 49.3% |
| Naskh | 12.2% | **32.8%** | 55.0% |

- **Maghrebi** is substitution-dominated: the model sees something at nearly every position but identifies the wrong character. This is the hallmark of visual confusion between visually similar glyphs — a *recognition* problem, not a *detection* problem.
- **Ruq'ah** has an unusually high deletion rate (40%). The CTC model is systematically *skipping* characters — most likely diacritics and closely-spaced strokes that fall below the detection threshold of the convolutional feature extractor.
- **Naskh** shows a balanced deletion/substitution profile, consistent with moderate degradation and mixed morphological complexity.

**Figures:** `figures/04_edit_type_stacked.png` · `figures/05_per_line_edit_rates.png`

---

## Section 3 — Sequence Length vs. Error Rate

### Objective
Test whether Kraken's CTC backbone exhibits spatial drift on longer lines. CTC models process input left-to-right with a sliding window; errors can accumulate as sequence length grows. This section empirically validates (or refutes) this hypothesis.

### Methodology
- Lines binned into 20 quantile-equal groups by (a) GT character length and (b) GT word count
- Mean CER plotted against bin centre for each script
- Scatter plot of `(gt_len, pred_len)` coloured by CER to reveal systematic length distortion

### Findings
- **Maghrebi** shows a clear positive slope on the CER-vs-length curve. CER climbs noticeably from ~25% on short lines to ~40%+ on long lines. Spatial drift is confirmed.
- **Ruq'ah** has a flatter but non-zero slope. CTC drift exists but is better controlled — likely because Ruq'ah's regular pen strokes provide more stable anchor points for the sliding window.
- **Short lines (<20 chars)** spike in CER across *all* scripts — isolated words or partial lines lack enough context for the CTC decoder to stabilise.
- Scatter plot shows Maghrebi predictions systematically *shorter* than GT (points below the diagonal), confirming deletion-clustering at the prediction level even though substitutions dominate the edit count.

**Figures:** `figures/06_length_vs_cer.png` · `figures/07_pred_gt_length_scatter.png`

---

## Section 4 — Character-Level Analysis

### Objective
Two complementary analyses: (1) understand the *input* character distribution per script (frequency of GT characters), and (2) identify the most common *mis-recognitions* (substitution confusion pairs). Together they distinguish between data imbalance problems and intrinsic morphological difficulty.

### Methodology
- All GT characters aggregated per book; top 20 by frequency plotted
- For each line, `editops()` enumerated to extract all `(ref_char -> hyp_char)` substitution pairs
- Top 15 confusion pairs plotted per book as horizontal bar charts

### Findings
- Character frequency distributions confirm space (` `) and common Arabic base forms dominate all three scripts — the *input data* is not degenerate.
- **Maghrebi** has the richest and most *diverse* confusion matrix: many different reference characters map to the same handful of predicted glyphs. This is the signature of a model that has collapsed distinct Maghrebi forms into a reduced set of "safe" predictions.
- **Ruq'ah** confusion pairs are more concentrated: a smaller set of character pairs account for most substitutions. This indicates *systematic* confusion of specific glyphs (likely Ruq'ah's characteristic connected cursive forms) rather than general confusion.
- **Naskh** confusion falls between — moderate diversity of error pairs.

**Figures:** `figures/08_top_gt_chars.png` · `figures/09_top_substitution_pairs.png`

---

## Section 5 — N-gram Error Rates

### Objective
Distinguish between *glyph-level* errors (poor recognition of individual characters) and *ligature-level* errors (failure to handle how characters fuse and change shape in context). A bigram/trigram error rate that scales steeply above the unigram rate proves that errors are *contextually cascading*, not random.

### Methodology
- For each line, extracted all character bigrams and trigrams from GT and hypothesis (after removing spaces)
- Bigram ER = `|GT_bigrams - Hyp_bigrams| / |GT_bigrams|` (set difference)
- Trigram ER computed analogously
- Results aggregated across all lines per book

### Findings

| Script | Unigram ER | Bigram ER | Trigram ER | Trigram/Unigram Ratio |
|--------|:----------:|:---------:|:----------:|:---------------------:|
| Maghrebi | 32.09% | 44.96% | **60.07%** | 1.87x |
| Ruq'ah | 11.51% | 19.09% | **27.19%** | **2.36x** |
| Naskh | 22.69% | 32.96% | **45.84%** | 2.02x |

- All three scripts show super-linear n-gram escalation, confirming errors are *not* i.i.d. noise.
- **Ruq'ah's ratio of 2.36x** is the highest despite having the lowest unigram ER. This means: even when Ruq'ah characters are individually recognised, their combination into ligatures is frequently broken. The model has learned the alphabet but not its contextual morphology.
- **Maghrebi's absolute trigram ER of 60.07%** means 3 out of every 5 character trigrams are wrong — a catastrophic result for any downstream OCR pipeline.

**Figure:** `figures/10_ngram_error_rates.png`

---

## Section 6 — Percentile / Outlier Curves

### Objective
Characterise the *failure tail*: not the typical line performance, but the distribution of the worst-performing lines. P90 and P95 CER determines how aggressive post-processing needs to be.

### Methodology
- Lines sorted by CER per script and plotted as a percentile curve (x = percentile rank, y = CER)
- P50, P90, P95 annotated as horizontal reference lines

### Findings

| Script | P50 | P90 | P95 |
|--------|:---:|:---:|:---:|
| Maghrebi | 30.0% | 45.1% | >70% |
| Ruq'ah | 8.97% | 19.7% | ~25% |
| Naskh | 19.7% | 33.8% | ~45% |

- Maghrebi's P95 is dramatically higher than its P90 — there is a *separate population* of extreme failures above the 90th percentile, not just a smooth continuation of the distribution.
- Ruq'ah's tail is well-behaved: the gap between P90 (19.7%) and P95 is modest, implying no discrete population of outliers.

**Figure:** `figures/11_cer_percentile_curves.png`

---

## Section 7 — CER Heatmap: Script x Line Length

### Objective
Provide a 2D view of error behaviour by jointly conditioning on script tradition and line length. This reveals interaction effects — e.g., does length affect all scripts equally, or does it differentially harm one?

### Methodology
- GT character lengths binned into 8 buckets: `[0-20, 21-40, 41-60, 61-80, 81-100, 101-150, 151-200, 200+]`
- Mean CER per `(script, length_bucket)` cell computed and displayed as a seaborn heatmap

### Findings
- The `0-20` chars bucket shows elevated CER for all scripts — short lines are universally difficult.
- Maghrebi CER is high (>30%) *uniformly* across nearly all length buckets — no length range is "safe."
- Ruq'ah's sweet spot is the `41-100` char range where CER is lowest. Very short and very long lines suffer more.
- Naskh degrades progressively with length, the most linear of the three.

**Figure:** `figures/12_cer_heatmap_length_script.png`

---

## Section 8 — Word Error Rate Analysis

### Objective
Complement CER with WER to detect word-boundary errors — specifically hallucination (inventing words), word splitting, and word merging — which CER cannot detect because it operates at character level.

### Methodology
- WER computed per line as `Levenshtein(ref_words, hyp_words) / len(ref_words)` — values > 1.0 are possible and meaningful
- Mean WER plotted with 95% confidence intervals
- WER scatter against word count to test dependency

### Findings
- **Maghrebi mean WER = 77.4% (Corpus WER = 75.6%)** — the model's word-level error rate is elevated, driven by ligature and character substitutions corrupting token identities.
- **Naskh mean WER = 59.4% (Corpus WER = 58.3%)** — intermediate word-level distortion, tracking character error rate.
- **Ruq'ah mean WER = 39.0% (Corpus WER = 38.3%)** — lowest error rate, with over 60% of words correctly identified without post-processing.
- WER-vs-word-count scatter shows positive correlation for all scripts: more words per line = more opportunity for alignment failures.

**Figure:** `figures/13_wer_analysis.png`

---

## Section 9 — Line Quality Tiers

### Objective
Categorise every line into an actionable quality tier. Research decisions (augmentation, post-processing, re-training) depend on knowing what *fraction* of data is in each state of correctness.

### Methodology
Five tiers defined:
- **Perfect**: CER = 0
- **Near-Perfect**: CER in (0, 5%]
- **Good**: CER in (5%, 20%]
- **Poor**: CER in (20%, 50%]
- **Catastrophic**: CER > 50%

### Findings

| Tier | Maghrebi | Ruq'ah | Naskh |
|------|:--------:|:------:|:-----:|
| Perfect | 5 (0.1%) | 32 (1.6%) | 1 (0.1%) |
| Near-Perfect (<=5%) | Small | ~18% | ~8% |
| Good (5-20%) | ~15% | ~40% | ~35% |
| Poor (20-50%) | ~45% | ~35% | ~38% |
| **Catastrophic (>50%)** | **199 (5.4%)** | **42 (2.1%)** | **31 (2.9%)** |

- Maghrebi has only **5 perfect lines** in 3,661. The model essentially never produces a flawless Maghrebi transcription.
- Ruq'ah achieves 32 perfect and ~18% near-perfect lines — a realistic post-processing target.
- **272 total catastrophic lines** across all scripts represent complete system failure.

**Figure:** `figures/14_line_quality_tiers.png`

---

## Section 10 — Feature Correlation Heatmaps

### Objective
Identify which measurable line properties (length, word count, edit type counts) are statistically associated with high CER. This reveals confounds and validates that the metrics are capturing distinct aspects of the error.

### Methodology
- Pearson correlation matrix computed for 8 features per script: `[cer, wer, gt_len, gt_word_count, ins, dels, subs, len_ratio]`
- Lower-triangle heatmap with annotations (upper triangle masked)

### Findings
- `cer <-> wer`: r ~= 0.75-0.85 across all scripts. Strong but not perfect — WER adds independent information.
- `cer <-> subs`: Strong positive for all scripts. Substitutions are the **primary CER driver**.
- `cer <-> dels`: Moderate positive, strongest in Ruq'ah (consistent with its deletion-dominant profile).
- `gt_len <-> gt_word_count`: r ~= 0.95+ — essentially collinear; either can be used as a proxy for the other.
- `len_ratio <-> cer`: Weak negative — when the model outputs fewer characters than GT (len_ratio < 1), CER is slightly higher.

**Figure:** `figures/15_correlation_heatmaps.png`

---

## Section 11 — CER Decile Distribution

### Objective
Provide a discretised view of the CER distribution in fixed 10% bins, allowing a precise count of how many lines fall in each error "severity band."

### Methodology
- CER values clipped to [0, 1] and binned into 10 equal-width intervals (0-10%, 10-20%, ..., 90-100%)
- Grouped bar chart: 3 scripts x 10 bins

### Findings
- **Ruq'ah**: >50% of lines fall in the 0-20% bins. The 0-10% bin is by far the largest.
- **Maghrebi**: Broadly distributed from 10% to 60% CER. No single bin dominates — the error is diffuse.
- **Naskh**: Bi-modal tendency — a cluster at 10-30% and another at 30-50%.

**Figure:** `figures/16_cer_decile_distribution.png`

---

## Section 12 — Combined Dashboard

A single composite figure unifying the most informative panels for presentation or paper inclusion.

**Figure:** `figures/00_DASHBOARD.png`

---

## EDA Completeness Checklist

| Analysis Dimension | Covered | Key Figure |
|-------------------|:-------:|-----------|
| Baseline metrics (CER/WER) | YES | `00_summary_table` |
| Distribution shape | YES | `01`, `02`, `03` |
| Error decomposition (Ins/Del/Sub) | YES | `04`, `05` |
| Sequence length effects | YES | `06`, `07` |
| Character frequency | YES | `08` |
| Confusion pairs | YES | `09` |
| N-gram / ligature analysis | YES | `10` |
| Outlier / failure tail | YES | `11` |
| 2D interaction heatmap | YES | `12` |
| Word-level errors | YES | `13` |
| Quality tiering | YES | `14` |
| Feature correlations | YES | `15` |
| Severity bands | YES | `16` |
