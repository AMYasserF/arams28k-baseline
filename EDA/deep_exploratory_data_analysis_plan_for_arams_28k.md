# Empirical Exploratory Data Analysis (EDA) Framework
**Project:** Comparative HTR Analysis on AraMS-28k (Kraken vs. HATFormer)
**Objective:** To systematically quantify error distributions across distinct script traditions, model architectures, and physical degradation levels, providing empirical justification for targeted post-processing and training interventions.

---

## 1. Script-Specific Error Topography (Font Separation)
*The dataset comprises three distinct hand-copied scripts (Maghrebi: `book_03`, Ruq'ah: `book_05`, Naskh: `book_09`). Aggregate CER obscures morphological challenges unique to each tradition.*

*   **1.1. Script-Isolated Confusion Matrices & N-gram Failure Rates:**
    *   **Methodology:** Compute distinct character-level confusion matrices for each test book. Expand this beyond unigrams to calculate Bigram and Trigram Error Rates (BER/TER).
    *   **Objective:** Maghrebi and Ruq'ah have vastly different ligature patterns. By isolating the N-grams, we identify if the models fail on individual glyph recognition or ligature disambiguation (where characters stack or fuse).
*   **1.2. Spatial Error Distribution (Intra-line Position):**
    *   **Methodology:** Map the probability of a character error against its normalized position within the text line (Beginning [0-0.3], Middle [0.3-0.7], End [0.7-1.0]).
    *   **Objective:** Arabic characters change shape based on position. We must determine if the models struggle disproportionately with word-final flourishes (common in Maghrebi) or initial isolated characters.

## 2. Architectural Comparative Analysis (Kraken vs. HATFormer)
*Kraken utilizes a CNN-RNN backbone with CTC loss, while HATFormer employs a Vision-Encoder-Decoder (Attention) architecture. Their failure modes are mathematically distinct.*

*   **2.1. Error Categorization Profiling (Insertion/Deletion/Substitution):**
    *   **Methodology:** Deconstruct the Levenshtein distance into its core components (Insertions, Deletions, Substitutions) and plot the ratios for both models across the test sets.
    *   **Objective:** CTC models (Kraken) often suffer from deletion errors (skipping characters) or repeating characters due to alignment failures. Attention models (HATFormer) are prone to hallucination (insertions) and catastrophic repetition loops. This metric will prove we understand *how* the models fail, not just *that* they fail.
*   **2.2. Sequence Length Decay (Attention Drift):**
    *   **Methodology:** Plot Word Error Rate (WER) and CER against the pixel-width of the line image and the character-length of the target string.
    *   **Objective:** HATFormer's cross-attention mechanism may degrade on unusually long text lines, whereas Kraken's sliding window (CTC) maintains consistent spatial awareness. This analysis will empirically validate if HATFormer loses context at the end of long manuscript lines.

## 3. Degradation and Robustness Analysis (The Damaged Subset)
*The paper authors curated a specific out-of-distribution evaluation set consisting of 177 naturally degraded lines (fading, staining, bleed-through) located in `book_09` (Naskh).*

*   **3.1. Clean vs. Degraded Performance Delta:**
    *   **Methodology:** Isolate the 177 lines specified in `real_damage_lines.txt`. Calculate the baseline CER/WER for these specific lines for both Kraken and HATFormer, and compare it against the CER/WER of the remaining "clean" Naskh lines in `book_09`.
    *   **Objective:** To quantify the exact drop in accuracy caused by native manuscript degradation. Does the deep attention of HATFormer handle bleed-through better than the localized convolutions of Kraken? This is a critical research finding.
*   **3.2. Attention Map Correlation on Damaged Pixels (Optional extension):**
    *   **Methodology:** If time permits, extract HATFormer attention weights specifically on the bounding boxes of the 177 damaged lines to see if attention scatters when encountering stains versus clean ink.

## 4. Lexical and Frequency Distribution Analysis
*Understanding how the training data distribution affects testing accuracy.*

*   **4.1. Frequency-Accuracy Correlation:**
    *   **Methodology:** Calculate the occurrence frequency of all unique characters and sub-words in the training split. Plot this frequency against the specific error rate of those tokens in the test split. 
    *   **Objective:** To mathematically prove whether high error rates on specific Maghrebi or Naskh characters are a symptom of "Data Imbalance" (rare characters) or "Morphological Complexity" (characters that are frequent but consistently misread).
*   **4.2. Out-Of-Vocabulary (OOV) Penalty:**
    *   **Methodology:** Identify words in the test sets that never appeared in the training corpus. Calculate the WER specifically for OOV words vs. In-Vocabulary words for both models.
    *   **Objective:** Evaluates the true generalization of the models. Does HATFormer memorize vocabulary, or does it learn the visual alphabet as effectively as Kraken?

## 5. Synthesis: Driving the Next Phase
*This EDA is not just observational; it dictates the engineering interventions required for the final contribution.*

1.  If **Section 1 & 4** reveal errors concentrated in rare ligatures/characters, we will implement **Targeted Data Augmentation (e.g., GridMask or Cutout)** during training to force feature learning.
2.  If **Section 2 & 3** reveal that errors are structurally valid Arabic words but contextually wrong (hallucinations), or if performance tanks on damaged lines, we will implement **Zero-Shot LLM Post-Processing** to restore reading flow using contextual language semantics.