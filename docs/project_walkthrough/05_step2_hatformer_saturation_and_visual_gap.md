[← Previous: Chapter 04](04_step2_kraken_zero_training_fix.md) | [Master Index](00_INDEX.md) | [Next: Chapter 06 — Post-Tier-1A Oracle & Step 3 →](06_post_tier1a_oracle_and_step3_plan.md)

---

# Chapter 05 — Step 2 Track A: HATFormer Saturation & Visual Gap

> **Chapter Takeaway:** Zero-training decoding interventions hit a **hard saturation wall on HATFormer** (`25.22% → 25.24%` CER, `56.68% → 56.94%` WER). While our **Blank-Gate** cleanly suppresses `3/3` empty paper hallucinations (`0` false positives), the validation-selected config (`LP=1.0, RP=1.15, LoopFB`) hurts Maghrebi (`+0.15 pp` CER, $p = 3.78 \times 10^{-7}$; `+751` insertions)—revealing the danger of a `100%`-Naskh validation set. Our failure taxonomy proves why: **`451 / 528` (`85.42%`)** of HATFormer's catastrophic Maghrebi lines are **fluent, grammatically plausible non-loop Arabic hallucinations**, while true loops represent just `0.49%` of Maghrebi insertions. This exposes HATFormer's root limitation: **visual feature inferiority**, not decoding loops.

---

## 1. Why We Did This (Reason & Question)

In Chapter 03, HATFormer trailed Kraken on CER (`25.22%` vs. `22.22%`), driven by **`528` catastrophic lines in Maghrebi** (`36.39%` CER) and **`+61.0%` raw insertions**. Initially, a natural hypothesis in neural sequence-to-sequence models was that greedy/unconstrained beam decoding was trapping HATFormer in endless repetition loops (like `ثنا ثنا ثنا...`) or hallucinating infinitely on blank/narrow crops.

We asked: **If we add deterministic zero-training decoding safeguards to HATFormer on Kaggle GPU—without retraining weights—how much error drops?**

---

## 2. What We Did & How (Step-by-Step + Code & File References)

Built in [`scripts/models/hatformer/tier1a_decode.py`](../../scripts/models/hatformer/tier1a_decode.py) (verified locally via `--self-test` before GPU execution), run across `19` cumulative/grid configurations on Kaggle T4 via [`kaggle/tier1a_decoding_kaggle.ipynb`](../../kaggle/tier1a_decoding_kaggle.ipynb), with full tables stored in [`runs/tier1a_hatformer_results/val_ablation/ablation_summary.md`](../../runs/tier1a_hatformer_results/val_ablation/ablation_summary.md) and [`runs/tier1a_hatformer_results/test_ablation/ablation_summary.md`](../../runs/tier1a_hatformer_results/test_ablation/ablation_summary.md).

### Four Modular Zero-Training Safeguards Tested
1. **Blank-Gate (`BG`, `--blank-gate`):** Measures pixel contrast (`span < 25` & `std < 12`, `dark_ratio < 0.002`, or tiny geometry `w < 15` / `h < 15`) outside white exterior mask padding. Any empty crop skips the neural decoder and outputs `""`. (Self-test calibrated: `0 / 1,486` validation false positives, `3 / 3` recall on the blank crops.)
2. **Train-Fit Width-Cap (`WC`, `--width-cap`):** Learns a piecewise `aspect_ratio → max_new_tokens` cap strictly on `split == 'train'` (`N = 19,719` non-sliver rows) to stop short catchwords (`ar < 3.5`) from rambling to 64 tokens while leaving standard full-width lines uncapped (`max_text_length = 64`).
3. **Mild Repetition Penalty + Length Penalty Grid (`RP ∈ {1.0, 1.10, 1.15, 1.20}`, `LP ∈ {1.0, 1.1}`):** Replaces aggressive `no_repeat_ngram=3` (which breaks legal Arabic genealogy chains) with gentle token discouragement and standard beam length normalization (`LP`).
4. **Post-Decode 4-Rule Loop Fallback (`LoopFB`, `--loop-fallback`):** After decoding, flags lines containing genuine repetition degeneracies (4+ identical non-space chars, 3+ repeated 2–5 char substrings, 3+ repeated 1–4 word n-grams, or non-particle word bursts `>=4/7` words—carefully exempting authentic *isnad* chains like `عن ... عن ...` and `بن`) and surgically re-decodes **only flagged lines** with `rep_penalty=1.35` and `no_repeat_ngram=4`.

---

## 3. Results & Visual Evidence

### Ablation Results Across Validation (`N=1,486`) & Test (`N=6,744`)

On the **validation split** (`book_06 + book_11`), adding `BG + WC + LP=1.0 + RP=1.15 + LoopFB` (`05_bg_wc_lp1.0_rp1.15_loopfb`) dropped validation CER **`9.68% → 9.47%`** (`Cat 10 → 9`). However, evaluating that **exact Val-Selected config** alongside clean `01_plus_blank_gate` on **test (`N=6,744`)** exposed sharp script divergence ([`ablation_summary.md`](../../runs/tier1a_hatformer_results/test_ablation/ablation_summary.md) & [`paper1_step2_tier1a_tables.md`](../../EDA/reports/paper1_step2_tier1a_tables.md)):

| Config | Beams / LP / RP | Overall Strict CER | Overall Corp WER | Maghrebi (`book_03`) CER / Ins | Ruq'ah (`book_05`) CER | Naskh (`book_09`) CER | Overall Cat (`>50%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`00_baseline_beams3`** | `3 / 0.0 / 1.00` | `25.22%` | `56.68%` | `36.39%` (`18,745`) | `11.75%` | `23.72%` | `572` (`8.48%`) |
| **`01_plus_blank_gate`** *(Clean)* | `3 / 0.0 / 1.00` (+BG) | `25.22%` | **`56.67%`** | `36.39%` (`18,745`) | `11.75%` | `23.73%` | `572` (`8.48%`) |
| `02_plus_width_cap` | `3 / 0.0 / 1.00` (+BG+WC) | `25.22%` | `56.67%` | `36.39%` (`18,745`) | `11.75%` | `23.73%` | `572` (`8.48%`) |
| **`05_bg_wc_lp1.0_rp1.15_loopfb`** *(ValOpt)* | `3 / 1.0 / 1.15` (+All) | `25.24%` | `56.94%` | **`36.54%`** ⚠ (`19,496`) | **`11.72%`** | **`23.53%`** | `566` (`8.39%`) |

- **Blank-Gate Effect:** Surgically turns all `3` blank crops into `""` (`p = 1.0000` on overall distribution, `0` false positives on `8,227` legible lines).
- **The `LP=1.0` & Naskh-Only Validation Trap (Pair C):** Because **`val` is 100% Naskh**, tuning `LP=1.0 + RP=1.15 + LoopFB` picked settings that **improved Naskh (`-0.19 pp` CER, `[-0.43, -0.01]` bootstrap CI; `Cat 30 → 27`)**, yet **statistically significantly degraded Maghrebi**: Strict CER rose **`+0.15 pp`** (`36.39% → 36.54%`, Holm–Bonferroni `p = 3.78e-07`), Corpus WER rose **`+0.57 pp`** (`73.38% → 73.94%`, Holm `p = 2.79e-14`), and Maghrebi insertions jumped by **`+751` characters** (`18,745 → 19,496`) because `LP=1.0` rewards longer outputs even when uncertain! (`LoopFB` itself only triggered `8` times out of `6,744` lines.)

### Why Zero-Training Decoding Saturates on Maghrebi (`book_03` Taxonomy)

![Root Cause of HATFormer Saturation on Maghrebi](assets/chart_05_hatformer_maghrebi_taxonomy.png)

Reproducing [`comparative_eda.py`](../../EDA/comparative_eda.py)'s exact failure taxonomy alongside our 4-rule `detect_tail_loop` detector across all `528` catastrophic Maghrebi lines in `01_plus_blank_gate` (Baseline) proves why tuning decoding flags cannot move HATFormer ([`paper1_step2_tier1a_tables.md`](../../EDA/reports/paper1_step2_tier1a_tables.md) Section 6):

| Failure Category | Count (`4-Rule Loop`) | % of `528` Cat Lines | Count (`EDA Regex`) | % of All `book_03` (`3,661`) |
| :--- | :---: | :---: | :---: | :---: |
| **`plausible_but_wrong_arabic`** *(Fluent non-loop guesses)* | **`451`** | **`85.42%`** | `451` (`85.42%`) | `12.32%` |
| `short_line` (`GT < 20` chars) | `41` | `7.77%` | `41` (`7.77%`) | `1.12%` |
| `severe_hallucination` (Length ratio `>1.60` or `ins_rate > 60%`, no loop) | `26` | `4.92%` | `27` (`5.11%`) | `0.71%` |
| `severe_truncation` (Length ratio `< 0.40`) | `6` | `1.14%` | `6` (`1.14%`) | `0.16%` |
| `repetition_loop` | `4` | `0.76%` | `3` (`0.57%`) | `0.11%` |

- **Corpus-Wide Loop Incidence (`book_03`, `N = 3,661`):** Only **`7 / 3,661` (`0.19%`)** lines contain repetition loops, accounting for just **`91 / 18,745` (`0.49% < 1%`)** of raw Maghrebi insertions!
- **Total Non-Loop Catastrophes:** **`524 / 528` (`99.24%`)** have no loop at all (even after LoopFB in `05_loopfb`, `525 / 527` = `99.62%` remain non-loop fluent guesses).

### Real Maghrebi Manuscript Examples of Fluent Hallucination (`01_plus_blank_gate` vs. Kraken Tier 1A)

1. **`book_03_line_000210` (Kraken Tier 1A `26.3%` CER vs. HATFormer `59.6%` CER):**  
   ![Maghrebi Example 000210](assets/samples/maghrebi_hall_book_03_line_000210.png)
   - **GT:** `نقضه كالحنث في اليمين بالله لأن الأيمان والنذور من العهود`
   - **Kraken Tier 1A:** `بعضه كالحفا في اليمن بالقلة لا الحا والنور من العمود` *(stays aligned stroke-by-stroke)*
   - **HATFormer:** `تفضه كما الخفابي في العين بالعث لأن لا قال الله تعالى والغفور من العمومون` *(hallucinates realistic Islamic phrase `"قال الله تعالى والغفور"`)*

2. **`book_03_line_001788` (Kraken Tier 1A `28.6%` CER vs. HATFormer `66.7%` CER):**  
   ![Maghrebi Example 001788](assets/samples/maghrebi_hall_book_03_line_001788.png)
   - **GT:** `عائشة أن ذلك في الأنصار وذلك أنهم كانوا قد`
   - **Kraken Tier 1A:** `مل شة أن ذا الا في الأنصار وذا لم أنهم كانوا فم` *(correctly anchors `الأنصار` and word flow)*
   - **HATFormer:** `على بشيء هذه قال مالك عبدي الاقصار هذا لك أنهم كانوا فمنهم` *(invents jurist dialogue `"قال مالك عبدي..."`)*

3. **`book_03_line_002535` (Kraken Tier 1A `34.6%` CER vs. HATFormer `55.8%` CER):**  
   ![Maghrebi Example 002535](assets/samples/maghrebi_hall_book_03_line_002535.png)
   - **GT:** `تعالى أنه كتبها في التوراة على موسى بن عمران وهم أهل`
   - **Kraken Tier 1A:** `فعلى أنه طنكلاب القوة عني مولى بن عمير أن وهم أهل`
   - **HATFormer:** `فقيلي أنه كفيلها في التوراة يحلى موسى عن عمر بن عبد الله بن عبد الله` *(substitutes a classical narrator chain `"عن عمر بن عبد الله بن عبد الله"` unrelated to the ink!)*

### Summary: 4 Visual Reasons Kraken Visually Beats HATFormer
Why can Kraken stay grounded on dense Maghrebi and stained Naskh while HATFormer hallucinates? Every factor traces to **visual encoding**, proven across our data:
1. **Overlapping `3×3` CNN Filters vs. Non-Overlapping `16×16` ViT Patches:** On de-confounded physically damaged Naskh (`n=177`, Ch. 03), Kraken's baseline CER lead over HATFormer nearly doubles (`-1.43 pp` on clean `n=859` → **`-2.76 pp` Corpus / `-2.81 pp` Mean** on damaged `n=177`, widening to `-4.55 pp` with Tier 1A). Overlapping CNN convolutions read around ink blotches; coarse `16×16` ViT patches corrupt entire token embeddings.
2. **Variable-Width Geometry vs. Fixed `384×384` Canvas Wrapping:** On short lines (`< 20` chars, `n = 95`), Kraken's native aspect-ratio scan averages `86.83%` Greedy (`82.79%` Tier 1A) vs. HATFormer's `101.67%` (`99.92%` `01-BG`), because wrapping resized images onto `64px`-tall `384×384` strips injects empty black padding transitions.
3. **Monotonic Strict 1D RTL Column Alignment:** CTC's right-to-left frame transitions make skipping half a sentence or fabricating extra jurist names physically very costly compared to soft global cross-attention.
4. **Grayscale Dewarped Ink Focus vs. ImageNet RGB Patches:** Kraken operates purely on height-dewarped grayscale ink intensity without ImageNet natural-photo RGB pretraining baggage.

---

## 4. Conclusion (What This Proves)

1. Zero-training decoding cannot unlock HATFormer on AraMS-28k (`25.22% → 25.24%` CER), except for the surgical **Blank-Gate** (`01_plus_blank_gate`).
2. Trying to tune HATFormer length/repetition penalties on AraMS-28k's `100%`-Naskh validation split actively harms Maghrebi (`+0.15 pp` CER, `+0.57 pp` WER).
3. HATFormer's Maghrebi weakness (`85.42%` non-loop fluent Arabic guesses, `<1%` loops) is a **neural visual-acoustic grounding defect**—requiring architectural retraining in **Paper 2**, NOT beam tweaking.

---

## 5. How This Affects the Next Step

Comparing Track B ([Chapter 04](04_step2_kraken_zero_training_fix.md)) and Track A (this chapter) gives an unmistakable verdict for **Paper 1**: **Kraken Tier 1A (`ValOpt`) is our new primary single-model engine (`19.86%` CER / `49.25%` WER)**, while **HATFormer `01_plus_blank_gate`** remains a strong specialist helper on deleted Ruq'ah lines and weak-ink Naskh. [Chapter 06](06_post_tier1a_oracle_and_step3_plan.md) quantifies this **Post-Tier-1A Oracle ceiling** and lays out the exact **Step 3 routing/arbitration plan**.

---

[← Previous: Chapter 04](04_step2_kraken_zero_training_fix.md) | [Master Index](00_INDEX.md) | [Next: Chapter 06 — Post-Tier-1A Oracle & Step 3 →](06_post_tier1a_oracle_and_step3_plan.md)
