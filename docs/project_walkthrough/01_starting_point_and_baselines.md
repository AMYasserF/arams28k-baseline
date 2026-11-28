[← Master Index](00_INDEX.md) | [Next: Chapter 02 — Benchmark Hygiene Audit →](02_benchmark_hygiene_audit.md)

---

# Chapter 01 — Starting Point & Baseline Models

> **Chapter Takeaway:** We began with two fine-tuned Arabic **HTR** (Handwritten Text Recognition — converting handwritten manuscript images into digital Unicode text) baselines built on completely different philosophies: the lightweight **`5.03M`-parameter Kraken** (`1D-CNN + BiLSTM + CTC`) and the neural encoder-decoder **HATFormer** (`BEiT-base ViT-patch16 + BlockProcessor + RoBERTa-base BBPE`, `beams=3`). After fixing a buggy character-counted WER (Word Error Rate) bug, both models established strong baselines on the `6,744`-line AraMS-28k official test split (`22.22%` Kraken CER vs. `25.22%` HATFormer CER — outperforming the published AraMS-28k paper benchmark of `26.74%`).

---

## 1. Why We Did This (Reason & Question)

Before trying to improve recognition accuracy or design a new architecture, we needed a **fair, exact, bug-free starting line**:
1. Both **Kraken** and **HATFormer** had just finished fine-tuning on the AraMS-28k training split (`19,739` lines).
2. We needed to know **exactly what each model outputs on the exact same test lines**, verify that zero test lines were ever leaked into training or tuning, and establish **consistent character- and word-level evaluation formulas** across all scripts (**Maghrebi**, **Ruq'ah**, and **Naskh**).
3. Crucially, our early audit discovered a bug in the starter evaluation code where `wer_line()` accidentally computed *character-level* edit distance on space-joined words divided by word count (artificially doubling WER to `~160%`), so unifying true word-level WER before making any scientific claims was mandatory.

---

## 2. What We Did & How (Step-by-Step + Code & File References)

### Step A — Dataset Structure (`AraMS-28k-HTR`)
We organized all experiments strictly around the official AraMS-28k book-level splits recorded in [`data/AraMS-28k-HTR/dataset_stats.json`](../../data/AraMS-28k-HTR/dataset_stats.json) and [`data/AraMS-28k-HTR/split_used.json`](../../data/AraMS-28k-HTR/split_used.json), using `line_type == 'main'` (`27,969` lines across `14` books and `3,043` manuscript pages):
- **Train Split (`19,739` lines, `9` books):** Used solely for fine-tuning and (later in Step 2) fitting statistical priors. Zero overlap with validation or test.
- **Validation Split (`1,486` lines, `2` books):** `book_06` (`874` lines) and `book_11` (`612` lines). Used strictly for checkpoint selection and **HPO** (Hyperparameter Optimization — choosing decoding settings without ever touching the test set).
- **Official Test Split (`6,744` lines, `3` distinct scripts):**
  - `book_03` (**Maghrebi**): `3,661` lines (`172` pages) — dense looping calligraphic script from North Africa/Andalusia.
  - `book_05` (**Ruq'ah**): `2,027` lines (`95` pages) — compact, slanted, fast-hand everyday script.
  - `book_09` (**Naskh**): `1,056` lines (`46` pages) — clear classical book hand, containing `177` physically damaged lines labeled independently in **AraMS-Restore** (a separate companion damage benchmark).

### Step B — Two Fine-Tuned Baseline Models
1. **Model A — Kraken (`CNN + BiLSTM + CTC`, `5.03M` parameters):**
   - Trained via [`scripts/train_kraken.py`](../../scripts/train_kraken.py) from the Muharaf checkpoint (`muharaf_rec_best.mlmodel`) to Epoch 51 ([`kraken/best_0.8957.safetensors`](../../kraken/best_0.8957.safetensors)).
   - **How it works:** Takes a grayscale line image normalized vertically with Kraken's `CenterNormalizer` dewarping (keeping natural horizontal width intact), runs **1D/2D CNN** (Convolutional Neural Network — small sliding filters that detect ink edges) and a **2-layer BiLSTM** (Bidirectional Long Short-Term Memory — reading context left and right along the line), and decodes frame-by-frame using **CTC** (Connectionist Temporal Classification — an alignment layer that scans strictly column-by-column right-to-left without any built-in language model).
   - **Decoding mode tested at baseline:** Greedy CTC decoding (picking the single highest-scoring symbol or blank separator at each image frame).

2. **Model B — HATFormer (`BEiT-base ViT-patch16 + BlockProcessor + RoBERTa-base Arabic BBPE`, `333.9M` parameters):**
   - Fine-tuned on Kaggle T4 via [`scripts/models/hatformer/train_hatformer.py`](../../scripts/models/hatformer/train_hatformer.py) and [`configs/ocr_hatformer.yaml`](../../configs/ocr_hatformer.yaml) (`20,000` steps, `lr=5e-5`, `fp32` master weights + `fp16` autocast, initialized from `hatformer-muharaf`).
   - **Important Architectural Clarification:** Contrary to loose informal references calling it a "Hierarchical Swin-Transformer", inspecting the checkpoint config confirms HATFormer's true architecture is a **`BEiT-base` Vision Transformer** (`16×16` pixel patches, NOT Swin) coupled with a **`BlockProcessor`** (which resizes each flipped line crop to height `64` and wraps wide lines onto stacked rows inside a fixed **`384×384` RGB canvas**, see [`dataset.py`](../../scripts/models/hatformer/dataset.py)) and a **`RoBERTa-base` Arabic BBPE decoder** (`50,265` Byte-Level BPE subword tokens).
   - **Decoding mode compared:** Greedy (`beams=1`, [`EDA/hatformer_beams1_pred/`](../../EDA/hatformer_beams1_pred/)) vs. **Beam Search (`beams=3`, `LP=0.0`)** ([`EDA/hatformer_beams3_pred/`](../../EDA/hatformer_beams3_pred/)), selecting `beams=3` as the primary HATFormer baseline.

### Step C — Unified Evaluation Metrics & Bug Fix
Implemented identically across [`EDA/comparative_eda.py`](../../EDA/comparative_eda.py), [`EDA/paper1_stats_audit.py`](../../EDA/paper1_stats_audit.py), and [`EDA/paired_kraken_vs_hatformer.csv`](../../EDA/paired_kraken_vs_hatformer.csv):
- **Strict Corpus CER** (Character Error Rate): Total Levenshtein character edits (insertions + deletions + substitutions) across all lines divided by total reference characters.
- **Ortho-Folded Corpus CER** (Orthography-Normalized CER): Same formula after folding non-semantic Arabic spelling variations (`أ / إ / آ → ا`, `ة → ه`, `ى → ي`).
- **True Corpus WER & True Mean WER** (Word Error Rate): True word-sequence Levenshtein edits (`LevDist.distance(ref_words, hyp_words)`) divided by total words (Corpus WER) or averaged per line (Mean WER), fixing the earlier character-level WER bug.
- **Catastrophic Failure Rate (`CER > 50%`):** Share of lines where more than half the characters are wrong.

---

## 3. Results & Visual Evidence

![Architecture Overview Comparison](assets/chart_00_architecture_overview.png)

### Visual Examples of Clean Representative Lines (Baseline Predictions)

Even before Step 2 fixes, both architectures transcribe clean, clear Arabic calligraphy accurately, though small differences in character fidelity vs. fluent word guessing already appear:

1. **Maghrebi (`book_03_line_002247`, Clean Baseline Reference):**  
   ![Clean Maghrebi Line](assets/samples/clean_maghrebi_book_03_line_002247.png)
   - **Ground Truth (`GT`):** `قال كان أصحاب رسول الله صلى الله عليه وسلم يقولون إذا`
   - **Kraken Tier 1A (`0.0%` CER):** `قال كان أصحاب رسول الله صلى الله عليه وسلم يقولون إذا`
   - **HATFormer Baseline (`1.9%` CER):** `قال كان اصحاب رسول الله صلى الله عليه وسلم يقولون إذا` *(dropped hamza on `أصحاب`)*

2. **Ruq'ah (`book_05_line_004190`, Clean Baseline Reference):**  
   ![Clean Ruq'ah Line](assets/samples/clean_ruqah_book_05_line_004190.png)
   - **Ground Truth (`GT`):** `عنه يدخل في هذا الحديث ثلث العلم وقال في رواية الربيع هذا الحديث`
   - **Kraken Tier 1A (`0.0%` CER):** `عنه يدخل في هذا الحديث ثلث العلم وقال في رواية الربيع هذا الحديث`
   - **HATFormer Baseline (`4.7%` CER):** `عنه يدخل في هذا الحديث ثلاثا لعلم وقال في رواية الربيع هذا الحديث`

3. **Naskh (`book_09_line_005807`, Clean Baseline Reference):**  
   ![Clean Naskh Line](assets/samples/clean_naskh_book_09_line_005807.png)
   - **Ground Truth (`GT`):** `في الرجال أشد من شرط البخاري ومسلم محمد رسول الله صلى الله عليه وسلم`
   - **Kraken Tier 1A (`0.0%` CER):** `في الرجال أشد من شرط البخاري ومسلم محمد رسول الله صلى الله عليه وسلم`
   - **HATFormer Baseline (`7.4%` CER):** `في الرحال اشتد من شرط البخاري وشلم محمد رسول الله صلى الله عليه وسلم`

### Baseline Quantitative Scorecard (`Official Test Split, N = 6,744`)

First, verifying HATFormer `beams=3` vs. `beams=1` ([`report_4_kraken_vs_hatformer.md`](../../EDA/reports/report_4_kraken_vs_hatformer.md)): Beam search improved every script (`book_03`: `37.55% → 36.39%`, `book_05`: `12.17% → 11.75%`, `book_09`: `24.36% → 23.72%`; overall `26.01% → 25.22%`, surpassing the published AraMS-28k paper HATFormer baseline of `26.74%`).

| Book / Script | Lines | Kraken Greedy CER | HATFormer `B=3` CER | Kraken Ortho CER | HATFormer Ortho CER | Kraken Corp WER | HATFormer Corp WER | Kraken Cat (`>50%`) | HATFormer Cat (`>50%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` (Maghrebi) | `3,661` | **`30.85%`** | `36.39%` | **`28.76%`** | `33.81%` | `75.61%` | **`73.38%`** | **`199` (`5.44%`)** | `528` (`14.42%`) |
| `book_05` (Ruq'ah) | `2,027` | **`11.25%`** | `11.75%` | **`9.19%`** | `9.31%` | `38.28%` | **`36.57%`** | `42` (`2.07%`) | **`14` (`0.69%`)** |
| `book_09` (Naskh) | `1,056` | **`22.11%`** | `23.72%` | **`19.76%`** | `21.30%` | `58.30%` | **`53.18%`** | `31` (`2.94%`) | **`30` (`2.84%`)** |
| **OVERALL** | **`6,744`** | **`22.22%`** | `25.22%` | **`20.10%`** | `22.72%` | `59.31%` | **`56.68%`** | **`272` (`4.03%`)** | `572` (`8.48%`) |

*(Note: Paper Table 4 reference for HATFormer was `37.88%` Maghrebi, `13.26%` Ruq'ah, `25.37%` Naskh, `26.74%` overall; our Kaggle-trained checkpoint improved on every script.)*

---

## 4. Conclusion (What This Proves)

1. Both baselines are genuinely strong and properly trained (**HATFormer `25.22%` CER** beats the published AraMS-28k baseline of `26.74%`, and **Kraken `22.22%` CER** beats HATFormer by `2.99`–`3.00 pp` on character error).
2. Fixing the word-level WER formula immediately revealed an unexpected split: **Kraken leads on CER across every script (`22.22%` vs. `25.22%`), while HATFormer leads on WER across every script (`56.68%` vs. `59.31%`)**.
3. Both models also share `>50%` catastrophic failure lines (especially `31` in Naskh and `528` for HATFormer in Maghrebi) that demanded closer inspection before trusting aggregate numbers.

---

## 5. How This Affects the Next Step

Before trying to explain *why* Kraken and HATFormer disagree—or tuning any fixes—we had to answer a prerequisite scientific question: **are those catastrophic failures real model mistakes, or is the AraMS-28k official test set itself broken on certain pages?** That question drove our **Step 1A Benchmark Hygiene Audit** in [Chapter 02](02_benchmark_hygiene_audit.md).

---

[← Master Index](00_INDEX.md) | [Next: Chapter 02 — Benchmark Hygiene Audit →](02_benchmark_hygiene_audit.md)
