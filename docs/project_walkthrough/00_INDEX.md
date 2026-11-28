# AraMS-28k Historical Arabic HTR — Project Walkthrough & Master Index

[Start Reading: Chapter 01 — Starting Point & Baseline Models →](01_starting_point_and_baselines.md)

---

> **60-Second Executive Summary:**  
> We evaluated two fundamentally different Arabic **HTR** (Handwritten Text Recognition — reading scanned historical manuscript images directly into digital Unicode text) systems on the **AraMS-28k** dataset across three distinct Arabic script styles (**Maghrebi**, **Ruq'ah**, and **Naskh**).  
>
> First, our **Step 1 Audit** uncovered hidden dataset flaws—most notably a 5-line shifted ground-truth page, 3 blank paper crops, and 42 duplicated lines—and revealed a striking **CER-vs-WER Paradox**: the fast column-based **Kraken** model made fewer character mistakes (**CER** — Character Error Rate), while the neural **HATFormer** model made fewer whole-word mistakes (**WER** — Word Error Rate).  
>
> Second, in **Step 2 (Zero-Training Calibration)**, adding a leak-free statistical 5-gram **LM** (Language Model — scores how likely a sequence of Arabic letters is) to Kraken's decoder without changing a single neural weight dropped Kraken's Strict Corpus CER from **`22.22%` → `19.86%`** and slashed its Corpus WER by **`-10.06 percentage points`** (**`59.31%` → `49.25%`**), reversing the paradox so Kraken now wins on **both** characters and words across every script. Meanwhile, decoding fixes barely moved HATFormer (`25.22% → 25.24%` CER) because `85.42%` of its Maghrebi failures are grammatically fluent Arabic hallucinations caused by visual ambiguity, not looping bugs. Combining both models after Step 2 leaves a **`18.03%` CER / `44.94%` WER Oracle ceiling**, pointing directly to a lightweight selective **Step 3** routing/arbitration pass for **Paper 1**, followed by **Paper 2** neural architecture upgrades.

---

## Walkthrough Chapters (Read in Order)

| Chapter | Core Question Answered | Key Headline Result | Status |
| :--- | :--- | :--- | :---: |
| [01 — Starting Point & Baseline Models](01_starting_point_and_baselines.md) | What models, dataset splits, and metrics did we start with? | Kraken (`5.03M` params) vs. HATFormer (`beams=3`), evaluated across `N=6,744` official test lines (`3` scripts) with unified strict CER & true word-level WER. | **Done** |
| [02 — Benchmark Hygiene Audit (Step 1A)](02_benchmark_hygiene_audit.md) | Can we trust every label and image in the official AraMS-28k test set? | Found `29` mislabeled/blank artifact rows (`book_09_page_034` 5-line shift + `3` blank crops) & `42` duplicate lines; created Clean Stratum **`N=6,673`**. | **Done** |
| [03 — Architectural Diagnosis & Statistics (Step 1B)](03_architectural_diagnosis_and_stats.md) | Why did Kraken win CER (`22.22%`) while HATFormer won WER (`56.68%`)? | Proved complementary failure modes (CTC Ruq'ah deletions vs. HATFormer Maghrebi hallucinations), sub-10% Ortho CER, & de-confounded physical damage impact. | **Done** |
| [04 — Step 2 Track B: Kraken Calibration](04_step2_kraken_zero_training_fix.md) | Can decoding + a leak-free character LM fix Kraken without retraining? | **Yes — complete Paradox Reversal:** `22.22% → 19.86%` CER (`-2.37 pp`), `59.31% → 49.25%` WER (`-10.06 pp`), beating HATFormer everywhere ($p \le 6.7 \times 10^{-19}$). | **Done** |
| [05 — Step 2 Track A: HATFormer Saturation](05_step2_hatformer_saturation_and_visual_gap.md) | Can zero-training decoding guards fix HATFormer's `36.39%` Maghrebi CER? | **No — decoding saturates:** `Blank-Gate` fixes `3` empty crops, `LP=1.0` hurts Maghrebi (`+0.15 pp`), & `85.42%` of Maghrebi failures are non-loop fluent guesses. | **Done** |
| [06 — Post-Tier-1A Oracle & Step 3 Plan](06_post_tier1a_oracle_and_step3_plan.md) | How much complementary gain remains and how will Step 3 capture it? | Post-Tier-1A Oracle = **`18.03%` CER / `44.94%` WER**; `50.4%` of all remaining edit savings concentrate in the top `24.9%` high-disagreement lines. | **Next** |
| [07 — Paper 1 → Paper 2 Roadmap & Decisions](07_roadmap_paper1_to_paper2_and_decisions.md) | How does this turn into two strong papers and what decisions do we need? | Clear two-paper split (**Paper 1** audit + zero-training fix; **Paper 2** neural CTC-attention hybrid HATFormer) + supervisor meeting checklist. | **Planned** |

---

## Master Comparison Snapshot (All Verified Milestones)

All metrics below use identical corpus-weighted definitions on both the **Official Test Split (`N = 6,744`)** and the **Full Hygiene-Deduplicated Stratum (`N = 6,673`)**, verified in [paper1_step2_tier1a_tables.md](../../EDA/reports/paper1_step2_tier1a_tables.md).

| Stage / System | Stratum | Strict Corpus CER | Ortho-Folded CER | True Corpus WER | True Mean WER | Catastrophic (`>50% CER`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kraken Greedy (Baseline)** | `N=6,744` | `22.22%` | `20.10%` | `59.31%` | `63.06%` | `272` (`4.03%`) |
| **HATFormer (`beams=3`, Baseline)** | `N=6,744` | `25.22%` | `22.72%` | `56.68%` | `60.83%` | `572` (`8.48%`) |
| **HATFormer `01_plus_blank_gate`** (Step 2A) | `N=6,744` | `25.22%` | `22.72%` | `56.67%` | `60.78%` | `572` (`8.48%`) |
| **HATFormer ValOpt (`05_loopfb`)** (Step 2A) | `N=6,744` | `25.24%` | `22.72%` | `56.94%` | `61.26%` | `566` (`8.39%`) |
| **Kraken Tier 1A (`ValOpt`)** *(Step 2B Winner)* | `N=6,744` | **`19.86%`** | **`17.98%`** | **`49.25%`** | **`52.54%`** | **`242` (`3.59%`)** |
| **Post-Tier-1A Oracle (`K_T1A` + `HAT 01-BG`)** | `N=6,744` | **`18.03%`** | **`16.22%`** | **`44.94%`** *(WER-sel)* / `46.02%` *(CER-sel)* | — | **`145` (`2.15%`)** |
| **Kraken Greedy (Baseline)** | `N=6,673` | `21.88%` | `19.75%` | `59.01%` | `62.65%` | `241` (`3.61%`) |
| **HATFormer `01_plus_blank_gate`** | `N=6,673` | `24.88%` | `22.37%` | `56.35%` | `60.34%` | `537` (`8.05%`) |
| **Kraken Tier 1A (`ValOpt`)** *(Clean Stratum)* | `N=6,673` | **`19.52%`** | **`17.64%`** | **`48.94%`** | **`52.13%`** | **`213` (`3.19%`)** |

---

## Plain-English Glossary

| Term / Abbreviation | Plain-English Meaning |
| :--- | :--- |
| **AraMS-28k** | A benchmark dataset of `27,969` main-text line images cropped from `14` historical Arabic manuscripts across multiple writing styles (**train**: `19,739`; **val**: `1,486`; **test**: `6,744`). |
| **AraMS-Restore** | An independent companion dataset identifying `177` physically damaged lines (ink bleed, holes, stains) within the Naskh test book (`book_09`). |
| **CER (Character Error Rate)** | Percentage of character-level edits (insertions + deletions + substitutions) needed to turn a model's prediction into the true text. **Strict Corpus CER** sums edits across all lines divided by total reference characters; **Ortho-Folded CER** ignores non-semantic Arabic spelling variants (`أ/إ/آ→ا`, `ة→ه`, `ى→ي`). |
| **WER (Word Error Rate)** | Percentage of whole-word edits (word insertions + deletions + substitutions) divided by total reference words. **Corpus WER** weights by word count globally; **Mean WER** averages line-by-line WER equally. |
| **Catastrophic Failure** | Any manuscript line where a model gets **`> 50%` Strict CER** (more than half of the line's characters are wrong). |
| **CTC (Connectionist Temporal Classification)** | A sequence alignment method used by Kraken that scans image columns strictly right-to-left and predicts one character (or a blank separator) per column, without a built-in language model. |
| **CNN + BiLSTM** | Convolutional Neural Network (small sliding visual filters that detect local ink edges) followed by a Bidirectional Long Short-Term Memory network (sequence memory looking forward and backward across the line). |
| **BEiT-base + BlockProcessor + RoBERTa BBPE** | HATFormer's true neural architecture: a `16×16` patch Vision Transformer (`BEiT-base`) wrapped around a `384×384` row-stacked image canvas (`BlockProcessor`), paired with an autoregressive Arabic subword decoder (`RoBERTa-base`, `50,265` Byte-Level BPE tokens). **Not** a Swin-Transformer. |
| **Prefix Beam Search + Kneser–Ney 5-Gram LM** | Instead of greedily picking the highest-probability letter at each column, Prefix Beam Search keeps the top `B=25` candidate hypotheses alive and scores their spelling fluency using a **5-gram LM** (a statistical model predicting the probability of the next letter given the previous 4 letters, smoothed via Kneser–Ney discounting). |
| **Decoding Knobs (`bp`, `lb`, `lm`)** | Kraken Step 2 search weights tuned strictly on validation: **Blank Penalty (`bp`)** discourages CTC blank-token skipping; **Length Bonus (`lb`)** rewards emitting characters; **LM Weight (`lm`)** controls how strongly the 5-gram Arabic language model guides decoding. |
| **HATFormer Safeguards (`BG`, `WC`, `RP`, `LP`, `LoopFB`)** | Zero-training decoding tests for HATFormer: **Blank-Gate (`BG`)** skips empty paper crops; **Width-Cap (`WC`)** limits maximum generated tokens based on crop width; **Repetition Penalty (`RP`)** discourages repeated tokens; **Length Penalty (`LP`)** biases toward longer outputs; **Loop Fallback (`LoopFB`)** re-decodes detected repeating tails. |
| **dHash (Difference Hash)** | A fast `64`-bit visual fingerprint computed from adjacent pixel brightness differences to spot near-duplicate manuscript images. |
| **Paired Bootstrap 95% CI** | Confidence Interval built by resampling the paired test lines with replacement `10,000` times (`seed=42`) to prove an observed gap is statistically reliable and not driven by a few lucky lines. |
| **Two-Sided Paired Wilcoxon Test** | A non-parametric significance test comparing per-line error differences between two models on the exact same lines without assuming a bell-curve error distribution. |
| **Holm–Bonferroni Correction** | A strict multiple-testing adjustment applied across `m=8` hypothesis tests per model pair to prevent false positives when testing multiple scripts and metrics at once. |
| **Mann–Whitney U & Fisher's Exact Tests** | Unpaired significance tests used to compare physically damaged (`n=177`) vs. undamaged (`n=859`) Naskh lines on continuous CER shifts (Mann–Whitney U) and binary `>50%` collapse enrichment (Fisher's Exact). |
| **Oracle Ceiling & 2×2 Rescue Matrix** | A theoretical best-of-both upper bound that picks whichever model has lower error on each line, paired with a 4-cell count showing how many catastrophic lines each model rescues for the other. |
| **Inter-Model CER & Step 3 Routing** | Character edit distance between Kraken and HATFormer predictions (`LevDist(K, H) / max(len(K), len(H), 1)`). **Routing** switches between the two models per line using simple rules or a selective judge without retraining either network. |
| **ROVER vs. Selective VLM Judge** | Two Step 3 arbitration options: **ROVER** (Recognizer Output Voting Error Reduction — pure local character-level alignment voting) vs. **Selective VLM** (Vision-Language Model reading only high-disagreement line crops). |

---

## Code & Data Reference Map

All paths below are relative to `docs/project_walkthrough/` and point directly to verified repository artifacts:

| Category | File / Directory | Purpose |
| :--- | :--- | :--- |
| **Step 1 Audit Scripts** | [EDA/kraken_eda.py](../../EDA/kraken_eda.py) | Single-model Kraken baseline EDA and edit decompositions |
| | [EDA/catastrophic_audit.py](../../EDA/catastrophic_audit.py) | Failure taxonomy rules & official `AraMS-Restore` damage audit |
| | [EDA/comparative_eda.py](../../EDA/comparative_eda.py) | Full Kraken vs. HATFormer Report 4 generator & label-noise audit |
| | [EDA/paper1_stats_audit.py](../../EDA/paper1_stats_audit.py) | Statistical hardening (bootstrap, Wilcoxon, Holm, `dHash`, hygiene artifact) |
| **Step 2 Calibration Code** | [scripts/models/kraken/train_char_lm.py](../../scripts/models/kraken/train_char_lm.py) | Builds leak-free 5-gram Kneser–Ney LM strictly on `split=='train'` |
| | [scripts/models/kraken/tier1a_ctc_decode.py](../../scripts/models/kraken/tier1a_ctc_decode.py) | Kraken CTC logits cache, prefix beam search + LM decoding sweep |
| | [scripts/models/hatformer/tier1a_decode.py](../../scripts/models/hatformer/tier1a_decode.py) | HATFormer Blank-Gate, Width-Cap, RP/LP, and Loop-Fallback suite |
| **Step 2 Closing Audit** | [EDA/paper1_step2_audit.py](../../EDA/paper1_step2_audit.py) | Closing Step 2 statistical & diagnostic audit (`Pair A/B/C`, Oracle, Maghrebi taxonomy) |
| **Walkthrough Asset Script** | [scripts/build_walkthrough_assets.py](../../scripts/build_walkthrough_assets.py) | Reproducibly builds all charts and sample manuscript crops in `assets/` |
| **Key Verified Data & Tables** | [EDA/paired_kraken_vs_hatformer.csv](../../EDA/paired_kraken_vs_hatformer.csv) | Baseline paired dataset (`6,744` rows) |
| | [EDA/paired_tier1a_kraken_vs_hatformer.csv](../../EDA/paired_tier1a_kraken_vs_hatformer.csv) | Unified post-Step-2 paired dataset (`6,744 × 37`, incl. `in_hygiene_stratum`) |
| | [EDA/arams28k_benchmark_audit_artifact.csv](../../EDA/arams28k_benchmark_audit_artifact.csv) | Exact `29` mislabeled / blank / truncated official test lines |
| | [EDA/reports/paper1_statistical_tables.md](../../EDA/reports/paper1_statistical_tables.md) | Step 1 statistical & benchmark hygiene tables |
| | [EDA/reports/paper1_step2_tier1a_tables.md](../../EDA/reports/paper1_step2_tier1a_tables.md) | Step 2 closing statistical, Oracle, & Maghrebi saturation tables |
| | [EDA/reports/report_4_kraken_vs_hatformer.md](../../EDA/reports/report_4_kraken_vs_hatformer.md) | Full Report 4 comparative diagnostic report |
| | [EDA/reports/report_5_kraken_ctc_tier1a_calibration.md](../../EDA/reports/report_5_kraken_ctc_tier1a_calibration.md) | Full Report 5 Kraken Tier 1A report |
| **Raw Sweep Outputs** | [runs/tier1a_kraken_val_ext/ctc_sweep_summary.md](../../runs/tier1a_kraken_val_ext/ctc_sweep_summary.md) | Kraken Stage 2 validation U-curve sweep (`24` configs) |
| | [runs/tier1a_kraken_test/ctc_sweep_summary.md](../../runs/tier1a_kraken_test/ctc_sweep_summary.md) | Kraken official test sweep (`18` configs) |
| | [runs/tier1a_hatformer_results/test_ablation/ablation_summary.md](../../runs/tier1a_hatformer_results/test_ablation/ablation_summary.md) | HATFormer Kaggle T4 decoding ablation (`19` configs) |

---

[Start Reading: Chapter 01 — Starting Point & Baseline Models →](01_starting_point_and_baselines.md)
