[← Previous: Chapter 06](06_post_tier1a_oracle_and_step3_plan.md) | [Master Index](00_INDEX.md)

---

# Chapter 07 — Paper 1 → Paper 2 Roadmap & Supervisor Decisions

> **Chapter Takeaway:** Rather than mixing a benchmark-audit/zero-training calibration story with slow neural retraining experiments in one delayed paper, splitting into **two complementary papers (`Path B → Path A` — fast zero-training paper first, deep neural retraining second)** maximizes speed, focus, and scientific impact: **Paper 1 (`Path B`)** delivers our AraMS-28k benchmark audit, Paradox Diagnosis & Reversal, and Step 2+3 zero-training pipeline (nearly finished!), while **Paper 2 (`Path A`)** attacks HATFormer's root visual-acoustic flaws via **4 targeted architectural upgrades** aiming to beat the two-model **Oracle** (`< 18.0%` **CER** Character Error Rate / `< 45%` **WER** Word Error Rate) in a single neural model.

---

## 1. Why We Did This (Reason & Question)

Coming out of Steps 1 and 2, we noticed two completely different kinds of breakthroughs in our results:
1. Immediate, training-free empirical discoveries (**benchmark label audit**, **CER-vs-WER Paradox**, **Kraken LM** Language Model reversal, **HATFormer Maghrebi saturation proof**, and **Step 3 selective routing**).
2. Deep architectural flaws in HATFormer (`384×384` row-wrapped canvas, coarse `16×16` **ViT** Vision Transformer patches losing dots/diacritics and physical stain resilience, zero monotonic alignment loss, and Naskh-only validation).

Trying to cram both a zero-training diagnostic paper and slow multi-week PyTorch architectural retraining into a single manuscript would dilute both contributions and delay our first submission unnecessarily. We needed a clean **Two-Paper Strategy**.

---

## 2. What We Did & How (The Two-Stage Strategy `Path B → Path A`)

### Paper 1 (`Path B` — Fast-Turnaround Diagnostic, Benchmark Audit & Zero-Training Paper)
- **Status:** Steps 1 & 2 (`100%` complete & statistically audited); only **Step 3** (length/confidence gate + arbitration) and manuscript drafting remain!
- **Core Thesis:**
  1. Benchmark hygiene audit (`N=6,744` vs. clean `N=6,673`) exposes `29` mislabeled/blank crops + `42` duplicate folio lines + `100%`-Naskh validation risk.
  2. Exposes and resolves the **CER-vs-WER Paradox**: adding a leak-free 5-gram Kneser–Ney LM to Kraken reverses its word deficit (`59.31% → 49.25%` WER, `22.22% → 19.86%` CER) without retraining.
  3. Proves *why* encoder-decoder beam calibration saturates on Maghrebi (`85.42%` non-loop fluent hallucinations, `<1%` loops) and why **CNN+CTC** (Convolutional Neural Network + Connectionist Temporal Classification) retains physical damage (`AraMS-Restore`) and short-line advantages.
  4. Closes the loop with a training-free **Step 3 selective router/arbitrator** capturing the remaining Kraken↔HATFormer complementarity (especially Ruq'ah truncations).

### Paper 2 (`Path A` — Deep Architectural Upgrade of HATFormer)
Directly solves every visual flaw uncovered in Chapter 05 across **4 concrete neural upgrades**:
1. **Auxiliary CTC Alignment Head (Hybrid CTC-Attention Training):** Attach a joint CTC objective atop HATFormer's visual encoder feature sequence alongside the **RoBERTa** autoregressive **cross-entropy loss** (standard next-token prediction loss) so visual representations enforce **monotonic right-to-left character grounding**—eliminating fluent hallucinations unrelated to the ink (`book_03_line_002535`).
2. **Hybrid CNN Stem / Fine-Grained Dot Resolution (Fixing Coarse `16×16` Patches & Wrapping):** Replace raw `16×16` ViT patches and height-64 `384×384` row-wrapping (`BlockProcessor`) with an overlapping convolutional stem + higher horizontal spatial resolution so small Maghrebi diacritics/dots and stained Naskh strokes (`AraMS-Restore`) stay separable without CNN weakness.
3. **Style Conditioning & Script-Balanced Validation Split:** Fix the `100%`-Naskh validation pitfall by carving out stratified **Maghrebi + Ruq'ah + Naskh** validation folds (strictly page/book-disjoint or stratified folio-disjoint within non-test sources) and conditioning/normalizing across calligraphic styles during training.
4. **Decoder Context Dropout & Cross-Architecture Distillation from Kraken Tier 1A:** Apply scheduled target/context dropout inside RoBERTa **cross-attention** (forcing the decoder to trust visual image keys over fluent n-gram guesses) and distill soft CTC/lattice posteriors (probability distributions over characters) from calibrated **Kraken Tier 1A** on Maghrebi & damaged lines.

---

## 3. Results & Concrete Target Milestones

### Paper 1 vs. Paper 2 Comparison & Quantitative Milestones

| Dimension | Paper 1 (`Path B`) | Paper 2 (`Path A`) |
| :--- | :--- | :--- |
| **Core Focus** | Benchmark Audit, Error Paradox Diagnosis & Zero-Training Pipeline | New Hybrid Vision-CTC-Attention HATFormer Architecture |
| **Neural Weights Retrained?** | **Zero** (`best_0.8957` + Kaggle HATFormer weights frozen) | **Yes** (multi-seed Kaggle/GPU training + ablations) |
| **Current / Target Strict CER (`N=6,744`)** | **`19.86%`** (Step 2) → Target **`< 19.0%`–`18.5%`** (Step 3) | **Solid Publishable:** `< 19.5%` \| **Single-Model Oracle Beater:** **`< 18.0%`** |
| **Current / Target Corpus WER (`N=6,744`)** | **`49.25%`** (Step 2) → Target **`< 47.0%`** (Step 3) | **Target:** **`< 45.0%`** (beating the two-model Oracle!) |
| **Target Maghrebi (`book_03`) Benchmark** | `27.23%` CER (Kraken Tier 1A) | Beat both HATFormer (`36.39%`) and Kraken Tier 1A (`27.23%`) |

*(Baseline progress already achieved on Official `N=6,744`: Published AraMS-28k HATFormer `26.74%` → Our baseline HATFormer `25.22%` / Kraken `22.22%` → Step 2 Kraken Tier 1A `19.86%` → Post-Tier-1A Oracle `18.03%`.)*

---

## 4. Conclusion (What This Proves)

1. The project already holds **publishable, statistically hardened, reproducible contributions** ready for Paper 1 (`22.22% → 19.86%` CER, `-10.06 pp` WER, complete benchmark hygiene audit with `29` artifact rows + `42` duplicate lines, and Maghrebi saturation proof).
2. Every future neural architecture decision in **Paper 2** now rests on verified quantitative evidence rather than intuition: **fix visual patch granularity, monotonic CTC grounding, and validation script balance—not decoder beam parameters**.

---

## 5. How This Affects the Next Step — Decisions Needed in Our Supervisor Meeting

Before starting Step 3 implementation and drafting Paper 1, we need clear alignment with **Dr. Youssef Ghattas** on **3 concrete decisions**:

- [ ] **Decision 1 — Paper 1 Target Venue & Timeline:** Confirm splitting into Paper 1 (`Path B`, audit + zero-training calibration + selective routing) first vs. Paper 2 (`Path A`, new hybrid architecture), and set the target conference/workshop submission deadline.
- [ ] **Decision 2 — Step 3 Selective Arbitration Design (Top `24.9%` Disagreement Lines, `1,676` lines):** After implementing the free local **Length/Confidence-Gated Router (Part 1)** tuned on `val`:
  - Should Paper 1 stay **100% local & lightweight** (**Option A:** character-level **ROVER** Recognizer Output Voting Error Reduction alignment voting between Kraken Tier 1A + HATFormer `01_plus_blank_gate`), OR
  - Should we also include an ablation adding **Option B: Selective VLM Judge** (**VLM** = Vision-Language Model inspecting only high-disagreement crops) exclusively on the `1,676` high-disagreement lines (`> 0.35 inter_model_cer` — character edit distance between models, capturing `50.4%` of Oracle edit savings)?
- [ ] **Decision 3 — Paper 2 Architecture Priorities:** Approve starting Paper 2 design around the **4 verified pillars** (Auxiliary CTC head on BEiT/CNN visual encoder, fine-grained CNN-ViT stem replacing `384×384` row wrap, script-balanced validation split, and Kraken Tier 1A cross-architecture distillation) with `< 18.0%` CER / `< 45.0%` WER single-model targets.

---

[← Previous: Chapter 06](06_post_tier1a_oracle_and_step3_plan.md) | [Master Index](00_INDEX.md)
