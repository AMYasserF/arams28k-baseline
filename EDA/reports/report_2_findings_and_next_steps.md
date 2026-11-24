# Report 2 — Findings Explained & Actionable Next Steps
**Project:** Comparative HTR Analysis on AraMS-28k (Kraken vs. HATFormer)  
**Based on:** Kraken EDA — 6,744 lines across Maghrebi · Ruq'ah · Naskh

---

> [!IMPORTANT]
> This report is **interpretive and prescriptive**. It takes every empirical finding from Report 1 and answers two questions: *What does this actually mean?* and *What do we do about it?* Every recommendation is directly traceable to specific EDA evidence.

---

## Finding 1 — The Ruq'ah Gap is Real and Structural

### What the data says
Ruq'ah (book_05) achieves a **mean CER of 11.5%** while Maghrebi (book_03) sits at **32.1%** — a **2.8x gap** using the same Kraken model weights. This is not explained by dataset size (Ruq'ah has *fewer* lines: 2,027 vs 3,661).

### What this means
Kraken's CNN-RNN backbone was trained on a mixture of Arabic scripts with Ruq'ah likely better represented in its upstream training data, or Ruq'ah's regular pen geometry is simply more amenable to convolutional feature extraction. Ruq'ah letters have a consistent baseline, predictable stroke width, and minimal diacritical density compared to Maghrebi. The model's sliding window "sees" a stable, consistent image signature per character. Maghrebi, by contrast, features looping descenders, stacked dots, and highly contextual ligature forms that change character identity based on surrounding strokes.

### What this means for your paper
This gap is your **central comparative result**. When you bring HATFormer into the analysis, the first question is: does HATFormer's attention mechanism close this gap — specifically on Maghrebi? If HATFormer also struggles on Maghrebi, the problem is in the *training data*, not the architecture. If HATFormer handles Maghrebi significantly better, the problem is CTC's inability to model contextual ligature disambiguation.

### Next step
> **Reproduce this table with HATFormer CER values placed alongside Kraken CER values in a side-by-side comparison.** The gap magnitude per script will be the most important table in your results section.

---

## Finding 2 — Maghrebi's Failure is Substitution-Dominated (Recognition Failure, Not Detection Failure)

### What the data says
63.2% of all Kraken errors on Maghrebi are **substitutions** — the model outputs *something* at nearly every position but identifies the wrong character. Only 16.5% are deletions (skipped characters).

### What this means
This is a *recognition* failure, not a *detection* failure. The convolutional feature extractor is finding a stroke at the right position (hence not deleting it), but the classifier head is mapping it to the wrong character class. This is the hallmark of **visual ambiguity** between character forms. In Maghrebi script, numerous letters share nearly identical visual forms — their identity is determined by subtle dot placement, loop closure, or connection angle. These features require fine-grained spatial resolution and contextual understanding of surrounding characters.

Kraken's CTC loss trains the model to maximise character-sequence likelihood in a conditionally independent manner — each output step is treated as independent given the input features at that position. This architecture structurally *cannot* model the contextual form-switching that Maghrebi requires.

### What this means for your paper
The substitution dominance, combined with the n-gram escalation data (Finding 5), gives you a **mechanistic explanation** for the error profile — not just a number. You can argue: "Kraken's CTC independence assumption is violated by Maghrebi's contextual morphology, leading to cascading substitution errors."

### Next step
> **Cross-tabulate the top-15 substitution pairs (from Section 4) with their Unicode character names.** This will reveal specific Maghrebi character confusion clusters (e.g., fa vs. qaf vs. tha) that you can cite by name in the paper. This is publishable as a "character confusion analysis."

> **Check whether the same confusion pairs appear in Naskh.** If they do, the confusion is *universal Arabic* (a data problem); if Maghrebi-specific pairs dominate, the confusion is *script-tradition-specific* (an architecture problem).

---

## Finding 3 — Ruq'ah's Dominant Failure is Deletion (Detection Failure, Not Recognition Failure)

### What the data says
40.0% of all Kraken errors on Ruq'ah are **deletions** — the model skips characters that exist in GT. Despite this, Ruq'ah has the *lowest* overall CER (11.5%).

### What this means
The model is *missing* characters in Ruq'ah, not misidentifying them. This is consistent with two known CTC failure modes:

1. **Diacritic blindness**: Ruq'ah's diacritical marks (tashkeel) are small, faint, and vertically displaced from the baseline. The CTC model's feature map may not resolve them at the input resolution, causing systematic deletion.
2. **CTC blank dominance**: CTC decoding uses blank tokens between characters. If the model assigns high probability to blank at a character's temporal position, that character is deleted from the output sequence. This is a known failure on tightly-spaced scripts.

The low CER despite high deletion rate also means the *remaining* characters (after deletion) are mostly correct. This is actually a **structurally better failure mode** than Maghrebi's — deletions can often be recovered by a language model that knows the word must have more characters, but substitutions that produce valid-looking Arabic are much harder to detect and correct.

### Next step
> **Measure diacritic-specific deletion rate.** Filter the Ruq'ah GT to lines containing Unicode diacritical marks (harakat: U+064B-U+0652). Compute CER only on those characters. If the diacritic deletion rate is significantly higher than the base character deletion rate, this confirms diacritic blindness and motivates **input resolution upscaling or diacritic-aware augmentation** as an intervention.

> **Tune CTC beam search parameters** (beam width, blank penalty) on the Ruq'ah validation set. A higher blank penalty reduces the probability of blank tokens and directly targets deletion errors.

---

## Finding 4 — Word Error Rate Reveals Word-Boundary and Token Corruption

### What the data says
Maghrebi mean WER = **77.4%** (Corpus WER = **75.6%**), Naskh mean WER = **59.4%** (Corpus WER = **58.3%**), and Ruq'ah mean WER = **39.0%** (Corpus WER = **38.3%**).
*(Note: A previous calculation artifact computed character-level distance across word sequences; true word-level Levenshtein distance confirms that over 3 out of 4 words in Maghrebi suffer from corruption, whereas over 60% of words in Ruq'ah are correctly transcribed.)*

### What this means
Kraken is not just mis-reading individual characters — it is **breaking the word structure** of Maghrebi and Naskh text. This manifests as:
- **Word splitting**: One GT word appears as two or more predicted tokens (a space hallucinated mid-word)
- **Word merging**: Two GT words appear without the space between them
- **Phantom words**: The model inserts entirely fabricated words not present in GT
- **Word loss**: A whole GT word is collapsed to nothing

This is distinct from character-level confusion and cannot be detected or fixed at the character level. It requires a **word-aware correction model** — specifically a language model that understands what valid Arabic word sequences look like.

The CER metric alone would *never* surface this problem. Severe word-level error is an emergency signal.

### What this means for your paper
This is your **strongest empirical justification for LLM post-processing**. The argument becomes: "Character-level correction is insufficient because word boundaries themselves are corrupted. We require a contextual language model to restore word-level coherence." You have the numbers to prove it.

### Next step
> **Implement the LLM post-processing pipeline on Maghrebi.** The input to the LLM is Kraken's raw output (corrupted, broken word boundaries). The prompt should be: *"The following is a corrupted Arabic text transcription from a manuscript. Correct it into fluent Arabic while preserving all content words."* Measure the **post-processing WER delta**: WER_before_LLM - WER_after_LLM. This delta is a publishable result.

> **Specifically target the 199 catastrophic lines (CER > 50%)** as an LLM post-processing test set. These are the worst-case scenarios and the most visible proof of concept.

---

## Finding 5 — N-gram Escalation Proves Ligature-Level Cascading

### What the data says
All three scripts show super-linear n-gram error escalation:

| Script | Unigram ER -> Trigram ER | Ratio |
|--------|:------------------------:|:-----:|
| Maghrebi | 32.1% -> **60.1%** | 1.87x |
| Ruq'ah | 11.5% -> **27.2%** | **2.36x** |
| Naskh | 22.7% -> **45.8%** | 2.02x |

### What this means
If errors were random and independent at the character level, bigram ER would be approximately `1 - (1-CER)^2` and trigram ER would be `1 - (1-CER)^3`. For Maghrebi (CER=0.321), this would predict: Bigram ~54%, Trigram ~69%. The *observed* bigram is 45% and trigram is 60% — both lower than the independent prediction.

This means that when errors occur, they *cluster* in specific problematic n-gram contexts (particular ligature environments) rather than being uniformly distributed. The model fails hard in certain ligature environments and succeeds in others.

For Ruq'ah, the ratio of 2.36x means errors are *more correlated* than independence would predict — a character error at position i strongly predicts an error at position i+1 (ligature context propagation).

The key result for your paper: **errors are not i.i.d. noise**. They are structurally coupled to Arabic's ligature morphology. This is a formal argument for why a character-level error metric alone is insufficient.

### Next step
> **Extend to word n-grams (word bigrams and trigrams).** Compute the word-level bigram error rate: what fraction of *consecutive word pairs* from GT appear verbatim in the hypothesis? This is a stronger metric for evaluating whether post-processing has restored coherent text flow.

> **Identify the specific trigrams with the highest error rates per script.** These are the target ligature environments. Training data augmentation should over-sample lines containing these high-error trigrams.

---

## Finding 6 — 272 Catastrophically Failing Lines Require Special Handling

### What the data says
- **Maghrebi**: 199 lines (5.4%) with CER > 50%
- **Ruq'ah**: 42 lines (2.1%) with CER > 50%
- **Naskh**: 31 lines (2.9%) with CER > 50%
- **Total**: 272 lines across the entire test set

### What this means
CER > 50% means the edit distance between the prediction and GT is more than half the GT length. At this severity, the transcription is not "wrong" — it is *unrecoverable* by a simple correction model. These lines likely correspond to:
- Extreme physical degradation (staining, bleed-through, fading)
- Atypical ligature clusters the model has never encountered
- Very short lines (isolated words) where the decoder has no context
- Lines where the model has entered a CTC repetition loop (the model repeats a character many times because the CTC blank probability is suppressed)

These 272 lines are a *distinct population* from the normally-failing lines. They should not be averaged into the main CER metric without acknowledgment.

### Next step
> **Audit all 272 catastrophic lines manually or semi-automatically.** Extract them from the CSVs using `df[df['cer'] > 0.50]`. Classify each by failure type (repetition loop, complete blank output, random noise output, plausible-but-wrong Arabic). This taxonomy will tell you whether the failure is a *model* problem or a *data quality* problem (truly degraded images that no model could read).

> **For Naskh specifically**, cross-reference the 31 catastrophic lines against the `real_damage_lines.txt` file (the 177 degraded lines referenced in your EDA plan). If the catastrophic lines disproportionately overlap with the damaged subset, you have direct empirical proof that **physical degradation is the cause** — which motivates degradation-robust training augmentation (noise injection, stain simulation).

---

## Finding 7 — Short Lines Are Universally Broken

### What the data says
Lines with fewer than 20 GT characters have elevated CER across **all three scripts** (visible in the heatmap and length-vs-CER plots).

### What this means
CTC decoders need a minimum sequence length to stabilise their output. Very short sequences (single words, partial lines, isolated headers) provide insufficient temporal signal for the LSTM's hidden state to converge on a confident character sequence. The model defaults to "uncertain" predictions in these conditions.

This is an **evaluation design issue** as much as a model issue. If your test set contains many short partial-line extractions (e.g., from page boundaries or column headers), those lines artificially inflate the reported CER for all models. This should be disclosed in your paper.

### Next step
> **Report stratified CER: "normal lines" (gt_len >= 20 chars) vs. "short lines" (gt_len < 20 chars) separately.** If short-line CER is dramatically higher, exclude them from the primary comparison metric and report them as a separate row in your results table, with the note: *"CER is computed on lines >= 20 characters to avoid short-sequence decoder instability bias."*

---

## Finding 8 — Ruq'ah Has Only 32 Perfect Lines (1.6%) — There is Room

### What the data says
Despite Ruq'ah's relatively good performance, only 32 of 2,027 lines (1.6%) are perfectly transcribed (CER = 0). Approximately 40% of lines fall in the "good" tier (5-20% CER).

### What this means
Ruq'ah has a **tractable, recoverable error profile**. The good/near-perfect tier (40%+ of lines below 20% CER) means post-processing has real material to work with. A language model correcting Ruq'ah transcriptions is operating on mostly-correct input, which is the ideal scenario for LLM-based correction.

This contrasts sharply with Maghrebi where the input to any post-processor is fundamentally corrupted — the LLM would be doing reconstruction, not correction.

### Next step
> **Ruq'ah should be the first LLM post-processing pilot.** Use the 2,027 Ruq'ah lines as your post-processing development set. The expected WER improvement will be the highest (starting from 39.0% with mostly-intact word structure) and the results will be interpretable. Maghrebi should be the *challenge* target, introduced after the pipeline is proven on Ruq'ah.

---

## Consolidated Intervention Roadmap

```
EDA Finding                            Recommended Intervention
--------------------------------------------------------------------
Maghrebi substitution dominance     -> Targeted ligature data augmentation
                                       (over-sample confused char pairs)

Ruq'ah deletion dominance           -> CTC blank penalty tuning
                                       Input resolution upscaling
                                       Diacritic-aware augmentation

High WER (word boundary & token corruption) -> LLM zero-shot post-processing
                                       (restore word structure semantically)

N-gram escalation (ligature cascade)-> Script-specific fine-tuning
                                       (separate model heads per script)

272 catastrophic lines              -> Manual audit + degradation taxonomy
                                       Cross-ref with real_damage_lines.txt

Short lines (<20 chars) failure     -> Stratified reporting
                                       Minimum-length filtering in evaluation

Ruq'ah tractable error profile      -> First LLM post-processing pilot
                                       Establish WER delta baseline
```

---

## Priority Ranking of Next Steps

| Priority | Action | Expected Impact | Effort |
|:--------:|--------|:--------------:|:------:|
| P1 (Critical) | Add HATFormer CER values to baseline table | Unlocks the core comparison | Low |
| P1 (Critical) | Audit catastrophic lines (CER > 50%) | Understand failure taxonomy | Medium |
| P2 (High) | LLM post-processing pilot on Ruq'ah | Proof of concept for paper | Medium |
| P2 (High) | Cross-ref catastrophic Naskh lines vs damaged subset | Degradation causation proof | Low |
| P3 (Medium) | Diacritic-specific deletion rate analysis | Justify resolution intervention | Low |
| P3 (Medium) | Stratified CER by line length in paper | Methodological transparency | Low |
| P4 (Nice) | Word bigram/trigram ER extension | Stronger post-processing metric | Medium |
| P4 (Nice) | Confusion pair Unicode naming table | Named character analysis in paper | Low |

---

## What These Findings Say About the Architecture

The EDA results collectively tell a coherent architectural story:

**Kraken (CTC)** is a *detection-first* model. It scans the image, finds strokes, and emits characters. It is good at finding *that something is there* (low deletion on Maghrebi) but bad at identifying *what it is* when forms are contextually ambiguous (high substitution on Maghrebi). It also fails to *find* small, faint features (high deletion on Ruq'ah diacritics). The CTC independence assumption means it cannot model the fact that knowing character i-1 and i+1 completely determines character i in many Arabic ligature environments.

**HATFormer (attention)** should, in theory, solve the second problem through cross-attention over the full image. If the HATFormer results show lower substitution rates on Maghrebi (because attention can resolve ambiguity by looking at surrounding glyphs), that is a direct empirical validation of the architectural motivation. If HATFormer instead shows higher insertion rates (hallucination), that is the known attention failure mode on long sequences — and it closes the loop on the EDA plan's architectural comparison section.

The EDA has set up the exact comparisons needed to make the architectural argument rigorous.
