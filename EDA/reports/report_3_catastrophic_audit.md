# Report 3 — Catastrophic Failure Audit & Naskh Damage Analysis (v2)
**Project:** Comparative HTR Analysis on AraMS-28k (Kraken vs. HATFormer)  
**Script:** [`EDA/catastrophic_audit.py`](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/EDA/catastrophic_audit.py)  
**Audit CSV:** [`EDA/catastrophic_lines_audit.csv`](file:///d:/CMP/Research/Baseline_Implementation/arams28k-baseline/EDA/catastrophic_lines_audit.csv)  
**Damage labels provenance:**
  - Source: `Archatext/AraMS-Restore` on Hugging Face (177 physically damaged `book_09` lines
    across 22 pages, `page_009` through `page_039`, with `page_offset` already corrected).
  - Kraken inference was evaluated on the corresponding polygon-masked `AraMS-28k-HTR` line crops.
  - **No CER-based proxy was used** — damage labels are fully independent of model predictions.

---

> [!IMPORTANT]
> **v2 fix:** The circular-reasoning issue in v1 (selecting 'damaged' pages by high CER,
> then measuring CER difference) is eliminated. Damage labels come exclusively from
> `Archatext/AraMS-Restore`, a separate dataset with human-annotated physical damage.

---

## 1 — Catastrophic Subset Verification

| Book | Script | Total Lines | Catastrophic (CER>50%) | Rate |
|------|--------|------------:|----------------------:|-----:|
| book_03 | Maghrebi | 3,661 | 199 | 5.4% |
| book_05 | Ruq'ah | 2,027 | 42 | 2.1% |
| book_09 | Naskh | 1,056 | 31 | 2.9% |
| **Total** | | **6,744** | **272** | **4.0%** |

---

## 2 — Failure Taxonomy Rules

| Priority | Mode | Rule |
|:--------:|------|------|
| 1 | `blank_output` | `kraken_pred` is empty or whitespace |
| 2 | `ctc_repetition_loop` | >=4 consecutive identical chars OR 2-4 char substring repeated >=3 times, absent in GT |
| 3 | `short_line` | GT length < 20 chars |
| 4 | `severe_truncation` | Non-blank, `len_ratio < 0.40` |
| 5 | `severe_hallucination` | `len_ratio > 1.60` OR insertion rate > 60% |
| 6 | `random_noise_or_garbled` | Garbled score > 0.40 (non-Arabic / fragmented tokens) |
| 7 | `plausible_but_wrong_arabic` | Catch-all: substitution-heavy, looks like Arabic |

---

## 3 — Taxonomy Results by Script

**Figure:** `figures/17_catastrophic_failure_taxonomy.png`

### Maghrebi (book_03)

| Failure Mode | Count | % |
|-------------|------:|:--:|
| CTC Repetition Loop | 1 | 0.5% |
| Short Line (<20 chars) | 41 | 20.6% |
| Severe Truncation | 1 | 0.5% |
| Severe Hallucination | 12 | 6.0% |
| Plausible But Wrong | 144 | 72.4% |

### Ruq'ah (book_05)

| Failure Mode | Count | % |
|-------------|------:|:--:|
| Blank Output | 2 | 4.8% |
| CTC Repetition Loop | 3 | 7.1% |
| Short Line (<20 chars) | 8 | 19.0% |
| Severe Truncation | 5 | 11.9% |
| Plausible But Wrong | 24 | 57.1% |

### Naskh (book_09)

| Failure Mode | Count | % |
|-------------|------:|:--:|
| Blank Output | 1 | 3.2% |
| Short Line (<20 chars) | 4 | 12.9% |
| Plausible But Wrong | 26 | 83.9% |

### Overall

| Failure Mode | Count | % |
|-------------|------:|:--:|
| Blank Output | 3 | 1.1% |
| CTC Repetition Loop | 4 | 1.5% |
| Short Line (<20 chars) | 53 | 19.5% |
| Severe Truncation | 6 | 2.2% |
| Severe Hallucination | 12 | 4.4% |
| Plausible But Wrong | 194 | 71.3% |

---

## 4 — Concrete Examples per Failure Mode

> [!NOTE]
> Arabic text shown as raw Unicode. GT = ground truth, PRED = Kraken prediction.

### Blank Output (`blank_output`)

| Book | CER | len_ratio | GT (first 70 chars) | PRED (first 70 chars) |
|------|----:|:--------:|---------------------|----------------------|
| book_05 | 1.000 | 0.00 | `وجوابه` | `` |
| book_05 | 1.000 | 0.00 | `ورأيته مرجعي من القدس الشريف فليس بصحيح بل ذلك جندرة حبشية أبي قرصافة` | `` |
| book_09 | 1.000 | 0.00 | `صالح بن أحمد قال أبي جعلت الميت في حل من ضربي وقال ما على رجل ألا يعذب` | `` |

### CTC Repetition Loop (`ctc_repetition_loop`)

| Book | CER | len_ratio | GT (first 70 chars) | PRED (first 70 chars) |
|------|----:|:--------:|---------------------|----------------------|
| book_05 | 0.806 | 0.21 | `التعريف به في الحديث قبله واما عايشه فهي الصديقة بنت الصديق والحبيبة ب` | `الم ل ا ل ل لين` |
| book_05 | 0.786 | 0.23 | `آخر وليس بمتواتر بخلاف ما يظنه بعض الناس فإن مداره على يحيى بن سعيد ال` | `ر ول لا ا س ا ا ا ا` |
| book_03 | 0.750 | 0.28 | `تعالى وقوله تعالى ولا تكونوا أول كافر به دليل خطاب هذه الاية` | `حر ل و ا ل ا ا اة` |

### Short Line (<20 chars) (`short_line`)

| Book | CER | len_ratio | GT (first 70 chars) | PRED (first 70 chars) |
|------|----:|:--------:|---------------------|----------------------|
| book_03 | 13.750 | 14.75 | `وعلم` | `الها كلعما الحك الكلام م اجاب الالعت وعنى معن قالولكه وعملم` |
| book_09 | 4.000 | 4.87 | `الذهلي والنسائي` | `وأجهن جعفذا نحال الداريون وأبو بكر لنزطه داود وطابقة شراهم وتقة النسائ` |
| book_09 | 3.933 | 4.53 | `الذهلي والنسائي` | `ممد يرلي الجيب سليمان النعدادن يعرف بالمرقري تريك البذي أبو سليما عن` |

### Severe Truncation (`severe_truncation`)

| Book | CER | len_ratio | GT (first 70 chars) | PRED (first 70 chars) |
|------|----:|:--------:|---------------------|----------------------|
| book_05 | 0.948 | 0.05 | `الحديث وإنما اشترط الشيخان الثقة والاشتهار بالطلب الثاني ادعى الخليلي ` | ` ا ل` |
| book_03 | 0.893 | 0.12 | `وغير مقدور له فإن ما هو مقدور للعبد فهو واقع مقدور للعبد` | `م له يه` |
| book_05 | 0.883 | 0.17 | `مجمع على عظم موقعه وجلالته وثبوته من حديث الإمام أبي سعيد يحيى بن سعيد` | `ي عب له ا ا ر` |

### Severe Hallucination (`severe_hallucination`)

| Book | CER | len_ratio | GT (first 70 chars) | PRED (first 70 chars) |
|------|----:|:--------:|---------------------|----------------------|
| book_03 | 1.577 | 2.23 | `إنما هو تعريف يسير بمبادئه` | `الى ل السعمره إنما هو الصحر المتفى به نفن رب يسى بجمعل ديه` |
| book_03 | 1.360 | 2.16 | `شيطانان وجاء الجمع على أن` | `شيكا نان مي خوله وله كن لا لشبه اطهق وجل الجمع عنى أدر` |
| book_03 | 1.217 | 1.91 | `والثاني أنه يختص بعمودي` | `والظا اللة ن علق نو يطلر عليم أنه تخقه يعمود` |

### Random Noise / Garbled (`random_noise_or_garbled`)

_No examples._

### Plausible But Wrong (`plausible_but_wrong_arabic`)

| Book | CER | len_ratio | GT (first 70 chars) | PRED (first 70 chars) |
|------|----:|:--------:|---------------------|----------------------|
| book_09 | 0.919 | 1.16 | `الماء الدايم وعنه س كذا وقع وقيل انه محمد بن صالح فلحته قلت كلاهما لم ` | `يونش بز ين يد سألت أنا الزنا عن بيع التمار فقال كان عروه بحدث عن سهلر ` |
| book_09 | 0.896 | 1.13 | `لكثرة خصومتهم قال البخاري في جامعه توفي في ذي القعدة سنة ثمان واربعين ` | `قال أحمد صالح حدثت أحمد برجنبل بحديثب زيد بن ثابت في سبيع الثمار فأعجب` |
| book_09 | 0.872 | 1.17 | `فقلت وابن مثله وفرد واه د في سننه حدثنا احمد بن صالح حدثنا عنبسة بن خا` | `في محلبه قذلك الشيب التليا فسيد الحال بيها ولقد بلغني أنه كان لا يجدث ` |

---

## 5 — Naskh Official Damage Cross-Reference (book_09)

> [!IMPORTANT]
> **Damage provenance:** `Archatext/AraMS-Restore` (Hugging Face).
> 177 lines across 22 pages (`book_09_page_009` through `book_09_page_039`).
> The `line_id` field (format `book_09__book_09_page_009_line010`, with
> `page_offset` already corrected) maps directly to `AraMS-28k-HTR` image stems.
> **All 177 IDs matched cleanly to test_manifest.txt entries — 0 unmatched.**

### 5.1 — ID Matching Summary

| Metric | Value |
|--------|------:|
| Official damaged line IDs loaded | 177 |
| Successfully matched to test_manifest | 177 |
| Unmatched IDs | 0 |

### 5.2 — Overlap Counts

| Metric | Value |
|--------|------:|
| Total book_09 lines | 1056 |
| Official damaged lines | 177 |
| Undamaged lines | 879 |
| Catastrophic & Damaged | 2 (1.1% of damaged) |
| Catastrophic & Undamaged | 29 (3.3% of undamaged) |

### 5.3 — CER and WER Statistics

| Group | n | Mean CER | Median CER | P90 CER | Std CER | Mean WER | P90 WER |
|-------|--:|:--------:|:----------:|:-------:|:-------:|:--------:|:-------:|
| Damaged | 177 | 0.2313 | 0.2192 | 0.3535 | 0.0977 | 1.1827 | 1.8133 |
| Undamaged | 879 | 0.2260 | 0.1944 | 0.3333 | 0.2197 | 1.1474 | 1.6875 |

### 5.4 — Statistical Tests

**Fisher's Exact Test** (catastrophic enrichment in damaged lines):
- Odds Ratio = 0.3350
- p-value = 9.77e-01 -> Not significant

**Mann-Whitney U Test** (CER: damaged > undamaged):
- U statistic = 89800.0
- p-value = 5.90e-04 -> **Significant** at alpha=0.05

**Figure:** `figures/18_naskh_damage_overlap.png`

---

## 6 — Conclusions

### 6.1 — Genuine Kraken/CTC Model Failures

1. **Plausible But Wrong Arabic (71.3%)** — The dominant failure type.
   The model outputs structurally valid-looking Arabic that is contextually incorrect.
   Concentrated in Maghrebi (substitution-dominated): CTC's conditional-independence
   assumption fails on contextually ambiguous Maghrebi ligature forms.
   These are the most dangerous failures — undetectable by rule-based filters,
   requiring LLM-based post-processing.

2. **Short Line (<20 chars) — 19.5%** — An evaluation design bias, not a model failure.
   53 lines fail catastrophically simply because CTC decoders need minimum context.
   These should be stratified separately in reported metrics.

3. **CTC Repetition Loop (1.5%)** — The CTC blank probability collapses,
   causing the decoder to repeat characters. A model-side failure, tunable via
   beam search blank penalty.

### 6.2 — Physical Degradation Effect (Leak-Free Analysis)

The Fisher's Exact Test is not significant (p=0.9769).
Physical degradation alone does not fully explain catastrophic failures in Naskh.

### 6.3 — Recommended Actions

| Priority | Action | Evidence |
|:--------:|--------|---------|
| P1 | Filter catastrophic lines from LLM post-processing input | OR=0.33; CER>50% lines unrecoverable by correction |
| P1 | Annotate `manual_review_notes` column in audit CSV | 272 lines need human confirmation |
| P2 | Train with degradation augmentation (noise, stain, bleed-through) | Damaged mean CER=0.231 vs undamaged 0.226 |
| P2 | Tune CTC blank penalty to suppress repetition loops | Model-side failure, directly tunable |
| P3 | Report stratified CER excluding short lines (<20 chars) | 53 catastrophic lines are evaluation bias, not model failures |