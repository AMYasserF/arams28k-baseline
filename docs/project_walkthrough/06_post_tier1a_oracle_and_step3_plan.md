[← Previous: Chapter 05](05_step2_hatformer_saturation_and_visual_gap.md) | [Master Index](00_INDEX.md) | [Next: Chapter 07 — Paper 1 to Paper 2 Roadmap →](07_roadmap_paper1_to_paper2_and_decisions.md)

---

# Chapter 06 — Post-Tier-1A Oracle Ceiling & Step 3 Execution Plan

> **Chapter Takeaway:** Pairing our calibrated **Kraken Tier 1A (`ValOpt`)** with clean **HATFormer (`01_plus_blank_gate`)** lowers the theoretical **Oracle ceiling** to **`18.03%` Strict CER** (`16.22%` Ortho CER), **`44.94%` WER-selected Ceiling** (`46.02%` CER-selected WER), and **`145` Catastrophic lines**—cracking **`7.79%` CER (`6.09%` Ortho)** on Ruq'ah! Crucially, HATFormer rescues **`97` Kraken-only catastrophic lines** (`46` in Ruq'ah, including all `15` newly truncated Tier 1A Ruq'ah lines), and **`50.4%` (`3,805 / 7,543`)** of all remaining character-edit savings sit inside the top `24.9%` high-disagreement lines (`inter_model_cer > 0.35`). This creates a sharp, selective two-part **Step 3 Plan**.

---

## 1. Why We Did This (Reason & Question)

After Step 2, Kraken Tier 1A leads HATFormer (`01_plus_blank_gate`) by **`-5.36 pp` CER (`19.86%` vs. `25.22%`)** and **`-7.43 pp` WER (`49.25%` vs. `56.67%`)**. That raised a critical decision gate before finalizing **Paper 1**:
1. Does HATFormer still offer unique value alongside calibrated Kraken Tier 1A, or is Kraken now strictly dominant everywhere?
2. If genuine complementarity remains, **which exact lines** account for the remaining Oracle upside so Step 3 doesn't waste compute or hurt clean Kraken lines?

---

## 2. What We Did & How (Step-by-Step + Code & File References)

Analyzed in [`EDA/paper1_step2_audit.py`](../../EDA/paper1_step2_audit.py) (Section 5A/5B), stored in [`EDA/paired_tier1a_kraken_vs_hatformer.csv`](../../EDA/paired_tier1a_kraken_vs_hatformer.csv) and [`EDA/reports/paper1_step2_tier1a_tables.md`](../../EDA/reports/paper1_step2_tier1a_tables.md):
1. **Paired Kraken Tier 1A (`ValOpt bw25_bp0.20_lb0.30_lm0.45`) with clean HATFormer (`01_plus_blank_gate`)** (and double-checked `05_bg_wc_lp1.0_rp1.15_loopfb` for completeness).
2. **Computed Per-Book & Overall Oracle Bounds:** Both **CER-selected Oracle** (picking whichever model has lower character error per line) and **WER-selected Oracle** (lower word error per line).
3. **Built the 2×2 Catastrophic Transition Matrix (`>50% CER`):** Counting lines catastrophic in Kraken-only (rescued by HATFormer), HATFormer-only (rescued by Kraken), Both, and Neither.
4. **Stratified All `6,744` Lines by `inter_model_cer` (`LevDist(K_T1A, H_01BG) / max_len`):** Bucketed lines across thresholds `{0.15, 0.25, 0.35}` to quantify exactly what percentage of all `7,543` saved Oracle edits fall into high-disagreement strata.

---

## 3. Results & Visual Evidence

### Post-Tier-1A Oracle & 2×2 Catastrophic Matrix (`Kraken Tier 1A ValOpt` + `HATFormer 01_plus_blank_gate`)

| Book / Script | `N` | K-T1A CER | H-BG CER | **Oracle CER** | **Oracle Ortho** | K-T1A WER | **CER-Sel WER** | **WER-Sel Ceiling** | K / H / **Oracle Cat** | K-Only Cat *(Rescued by HAT)* | H-Only Cat | Both Cat | CER W/T/L (`K/Tie/H`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `book_03` (Maghrebi) | `3,661` | `27.23%` | `36.39%` | **`26.05%`** | `24.25%` | `62.51%` | `60.43%` | **`59.04%`** | `147 / 528` / **`111`** | `36` | `417` | `111` | `2833 / 209 / 619` |
| `book_05` (Ruq'ah) | `2,027` | `10.43%` | `11.75%` | **`7.79%`** | **`6.09%`** | `31.98%` | `27.35%` | **`26.67%`** | `56 / 14` / **`10`** | **`46`** | `4` | `10` | `1296 / 201 / 530` |
| `book_09` (Naskh) | `1,056` | `19.84%` | `23.73%` | **`17.97%`** | `15.95%` | `48.69%` | `45.28%` | **`44.22%`** | `39 / 30` / **`24`** | `15` | `6` | `24` | `692 / 84 / 280` |
| **OVERALL** | **`6,744`** | `19.86%` | `25.22%` | **`18.03%`** | **`16.22%`** | `49.25%` | **`46.02%`** | **`44.94%`** | `242 / 572` / **`145`** | **`97`** | `427` | `145` | `4821 / 494 / 1429` |

*(Pairing with `05_loopfb` yields virtually identical numbers: `18.04%` CER, `45.03%` WER ceiling, `141` Cat, `101` K-only rescues.)*

### Disagreement Scoping — Where Complementarity Lives

![Oracle Gain Concentration Across Disagreement Strata](assets/chart_06_disagreement_oracle_gain.png)

Out of **`7,543` total Oracle character edits saved** over Kraken Tier 1A (`1,429 / 6,744` lines where HATFormer beats Kraken CER):

| Disjoint Disagreement Bucket (`inter_model_cer`) | Lines (`N`) | % Test | K-T1A CER | HAT 01-BG CER | **Oracle CER** | Within-Bucket Gain | Char Edits Saved | **% of All `7,543` Saved Edits** | Stratum H-Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`[0.00, 0.15]` Low** | `1,763` | `26.1%` | `8.01%` | `10.18%` | `7.20%` | `+0.81 pp` | `1,013` | `13.4%` | `23.4%` (`412/1763`) |
| **`(0.15, 0.25]` Moderate** | `1,637` | `24.3%` | `16.80%` | `21.86%` | `15.68%` | `+1.12 pp` | `1,183` | `15.7%` | `20.8%` (`341/1637`) |
| **`(0.25, 0.35]` High** | `1,668` | `24.7%` | `25.17%` | `32.47%` | `23.51%` | `+1.66 pp` | `1,542` | `20.4%` | `19.9%` (`332/1668`) |
| **`> 0.35` Very High** | **`1,676`** | **`24.9%`** | `34.73%` | `42.93%` | **`30.43%`** | **`+4.30 pp`** | **`3,805`** | **`50.4%`** | `20.5%` (`344/1676`) |
| *Cumulative `> 0.15`* | `4,981` | `73.9%` | `25.04%` | `31.79%` | `22.76%` | `+2.28 pp` | `6,530` | **`86.6%`** | `20.4%` (`1017/4981`) |
| *Cumulative `> 0.25`* | `3,344` | `49.6%` | `29.83%` | `37.57%` | `26.88%` | `+2.95 pp` | `5,347` | **`70.9%`** | `20.2%` (`676/3344`) |

### Visual Examples — Why Kraken Ruq'ah Truncations Are Trivial to Detect (`Step 3 Part 1`)

Look at the exact signature when Kraken Tier 1A truncates a faint Ruq'ah line while HATFormer reads it fluently:

1. **`book_05_line_003897` (Kraken Greedy `49.4%` → Kraken Tier 1A `60.5%` CER vs. HATFormer `13.6%` CER):**  
   ![Ruq'ah Truncated 003897](assets/samples/ruqah_trunc_book_05_line_003897.png)
   - **GT:** `ذكر له ايضا هو ابو الحسين مسلم بن الحجاج بن مسلم القشيري النيسابوري صاحب التصانيف` (`82` chars)
   - **Kraken Tier 1A (`42` chars — severe CTC deletion truncation):** `رسول اليه الحسن ما الحج ماشي النار الف`
   - **HATFormer (`80` chars — crisp lexical rescue):** `تعرض لها هو أبو الحسين مسلم بن الحجاج بن مسلم القشيري النيسابوري صاحب التعائف`

2. **`book_05_line_003719` (Kraken Greedy `48.1%` → Kraken Tier 1A `58.0%` CER vs. HATFormer `12.3%` CER):**  
   ![Ruq'ah Truncated 003719](assets/samples/ruqah_trunc_book_05_line_003719.png)
   - **GT:** `بعضهم قال القرطبي وروى عن النبي صلى الله عليه وسلم انه قال ان في النسا اربع نبيات` (`81` chars)
   - **Kraken Tier 1A (`38` chars — severe CTC deletion truncation):** `من الي وعن القول الله له ولم ان الا ا`
   - **HATFormer (`81` chars — near-exact rescue):** `بعضهم قال القرطي وروي عن التي صلى الله عليه وسلم انه قال إن في النساء أربع بنئات`

Notice the unmistakable physical clue: Kraken emitted `< 60%` of HATFormer's character length on a long line crop!

---

## 4. Conclusion (What This Proves)

Kraken Tier 1A alone hits `19.86%` **CER** (Character Error Rate), **and `97` Kraken-only catastrophic lines (especially `46` truncated Ruq'ah lines) remain cleanly rescuable by HATFormer**. Better yet, **`50.4%` of all remaining Oracle savings sit inside the top `24.9%` (`1,676` lines)** where the two models strongly disagree (`inter_model_cer > 0.35` — character edit distance between Kraken and HATFormer predictions), meaning Step 3 can stay fast, selective, and surgical.

---

## 5. How This Affects the Next Step — The Concrete Step 3 (Tier 1B) Execution Plan

Completing Step 3 wraps up all experimental work for **Paper 1**! Our findings directly prescribe a three-part execution pipeline before transitioning to [Chapter 07](07_roadmap_paper1_to_paper2_and_decisions.md):

### Part 1 — Mandatory, 100% Local & Zero-Cost: Length/Confidence-Gated Router
- **How it works:** Default every line to **Kraken Tier 1A (`ValOpt`)**. Switch to **HATFormer (`01_plus_blank_gate`)** *only* when simple unsupervised signals detect **CTC** (Connectionist Temporal Classification) blank collapse (e.g., severe length asymmetry `len(K) < τ · len(H)`, low mean CTC acoustic peak probability, and non-looping HATFormer output).
- **Verified Diagnostic Potential:** On our `EDA/paired_tier1a_kraken_vs_hatformer.csv` diagnostic check of a simple character-length gate `len(K_T1A) < 0.60 * len(H_01BG)`, switching just `46 / 2,027` Ruq'ah lines drops **Ruq'ah CER `10.43% → 9.43%` (`-1.00 pp`)**, Ruq'ah **Corpus WER** (Word Error Rate) `31.98% → 30.90%`, and **Ruq'ah catastrophic lines `56 → 20`** (`Overall Cat 242 → 192`, `19.86% → 19.47%` CER).
- **Leakage Rule:** Just like Step 2, all Step 3 router thresholds (`τ`, CTC confidence quantiles, width rules) **MUST be tuned strictly on `val` (`N=1,486`)** (and verified on `N=6,744` & `N=6,673`), never tuned on test.

### Part 2 — Supervisor Decision Gate: Arbitrating Remaining High-Disagreement Lines (`inter_model_cer > 0.35`, `24.9%` of lines)
Two options for resolving the top `1,676` high-disagreement lines holding `50.4%` of Oracle upside:
- **Option A — Pure Local `ROVER` Character-Level Lattice Alignment (Zero External API):** Run **ROVER** (Recognizer Output Voting Error Reduction — character confusion-network / minimum-Bayes-risk voting) to align Kraken Tier 1A (and top-K CTC beams + **LM** Language Model scores) with HATFormer `01_plus_blank_gate` weighted by local CTC confidence. Zero GPU cost, 100% reproducible locally.
- **Option B — Selective `VLM` (Vision-Language Model — multimodal neural model reading manuscript images alongside text candidates) Judge (`24.9%` of lines only):** Feed *only* the `1,676` high-disagreement line images + `(Kraken, HATFormer)` candidates to an Arabic-capable VLM to adjudicate or copy the optically grounded reading. Highest accuracy ceiling, requires API/GPU inference budget. *(Or combine Part 1 + Option A first as a clean ablation step before testing Option B!)*

### Part 3 — Final Statistical Hardening (`paper1_step2_audit.py` protocol)
Every Step 3 variant will be audited identically with `n_boot=10,000` paired **bootstrap CIs** (Confidence Intervals via resampling with replacement), two-sided paired **Wilcoxon** (non-parametric rank test) + **Holm–Bonferroni** (multiple-testing correction), and both `N=6,744` Official and `N=6,673` Hygiene-Deduplicated strata.

Next, [Chapter 07](07_roadmap_paper1_to_paper2_and_decisions.md) presents our full **Paper 1 → Paper 2 two-paper publication strategy** and lists the exact decisions needed from Dr. Youssef to execute smoothly.

---

[← Previous: Chapter 05](05_step2_hatformer_saturation_and_visual_gap.md) | [Master Index](00_INDEX.md) | [Next: Chapter 07 — Paper 1 to Paper 2 Roadmap →](07_roadmap_paper1_to_paper2_and_decisions.md)
