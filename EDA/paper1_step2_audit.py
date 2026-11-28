#!/usr/bin/env python3
"""
paper1_step2_audit.py
======================
Closing statistical & diagnostic audit for Step 2 (Tier 1A: Zero-Training
Decoding Calibration) — Paper 1 (Path B: Diagnostic, Benchmark Audit &
Zero-Training Fix on AraMS-28k).

Reuses exact metric & statistical definitions from EDA/paper1_stats_audit.py
and EDA/comparative_eda.py:
  - ortho_fold: أ/إ/آ -> ا, ة -> ه, ى -> ي
  - paired corpus-level bootstrap (n_boot=10,000, seed=42) over 4 metrics:
      Strict Corpus CER, Ortho-Folded Corpus CER, True Corpus WER, True Mean WER
  - two-sided paired Wilcoxon + Holm-Bonferroni (m=8 per pair: 4 splits x 2 metrics)
  - Full Hygiene-Deduplicated Stratum (N=6,673)
  - Maghrebi catastrophic taxonomy (EDA/comparative_eda.py + tier1a_decode.py rules)

Inputs
------
  EDA/paired_kraken_vs_hatformer.csv
  EDA/arams28k_benchmark_audit_artifact.csv
  runs/tier1a_kraken_test/bw25_bp0.20_lb0.30_lm0.45/      (Kraken ValOpt Tier 1A)
  runs/tier1a_kraken_test/bw25_bp0.40_lb0.30_lm0.45/      (Kraken TestBest reference)
  runs/tier1a_hatformer_results/test_ablation/00_baseline_beams3/
  runs/tier1a_hatformer_results/test_ablation/01_plus_blank_gate/
  runs/tier1a_hatformer_results/test_ablation/05_bg_wc_lp1.0_rp1.15_loopfb/

Outputs
-------
  EDA/paired_tier1a_kraken_vs_hatformer.csv
  EDA/reports/paper1_step2_tier1a_tables.md
  stdout (all key tables)
"""

import io
import re
import sys
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein as LevDist
from rapidfuzz.distance.Levenshtein import editops as lev_editops
from scipy.stats import wilcoxon

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PAIRED_CSV = ROOT / "EDA" / "paired_kraken_vs_hatformer.csv"
AUDIT_CSV = ROOT / "EDA" / "arams28k_benchmark_audit_artifact.csv"
KRAKEN_VALOPT_DIR = ROOT / "runs" / "tier1a_kraken_test" / "bw25_bp0.20_lb0.30_lm0.45"
KRAKEN_TESTBEST_DIR = ROOT / "runs" / "tier1a_kraken_test" / "bw25_bp0.40_lb0.30_lm0.45"
HAT_ABL_DIR = ROOT / "runs" / "tier1a_hatformer_results" / "test_ablation"
HAT_00_BASE_DIR = HAT_ABL_DIR / "00_baseline_beams3"
HAT_01_BG_DIR = HAT_ABL_DIR / "01_plus_blank_gate"
HAT_05_LFB_DIR = HAT_ABL_DIR / "05_bg_wc_lp1.0_rp1.15_loopfb"

OUT_CSV = ROOT / "EDA" / "paired_tier1a_kraken_vs_hatformer.csv"
OUT_MD = ROOT / "EDA" / "reports" / "paper1_step2_tier1a_tables.md"

BOOKS = {
    "book_03": "Maghrebi",
    "book_05": "Ruq'ah",
    "book_09": "Naskh",
}
DUP_FOLIO_PAGES = {"book_03_page_103", "book_03_page_104"}
ARABIC_RANGE = (0x0600, 0x06FF)
LOOP_WINDOW_EXEMPT_WORDS = frozenset({
    "بن", "عن", "في", "من", "على", "إلى", "الى", "أن", "ان", "و", "أو", "او", "ما",
})


# ─────────────────────────────────────────────────────────────────────────────
# Text, Metric & Edit Helpers (aligned with paper1_stats_audit.py & comparative_eda.py)
# ─────────────────────────────────────────────────────────────────────────────

def ortho_fold(text: str) -> str:
    """Fold non-semantic Arabic orthographic variants: أ/إ/آ->ا, ة->ه, ى->ي."""
    if not isinstance(text, str):
        return ""
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
    )


def norm_nfc_ws(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", text).split())


def decompose_edits(ref: str, hyp: str) -> tuple[int, int, int]:
    ops = lev_editops(ref, hyp)
    ins = sum(1 for op in ops if op.tag == "insert")
    dels = sum(1 for op in ops if op.tag == "delete")
    subs = sum(1 for op in ops if op.tag == "replace")
    return ins, dels, subs


def precompute_model_arrays(gt_list: list[str], pred_list: list[str], prefix: str) -> dict[str, np.ndarray]:
    """Compute exact per-line character, ortho-char, word, edit-op arrays."""
    n = len(gt_list)
    gt_len = np.empty(n, dtype=np.float64)
    gt_olen = np.empty(n, dtype=np.float64)
    gt_words = np.empty(n, dtype=np.float64)

    ced = np.empty(n, dtype=np.float64)
    oed = np.empty(n, dtype=np.float64)
    wed = np.empty(n, dtype=np.float64)
    l_cer = np.empty(n, dtype=np.float64)
    l_wer = np.empty(n, dtype=np.float64)
    ins_arr = np.empty(n, dtype=np.int64)
    dels_arr = np.empty(n, dtype=np.int64)
    subs_arr = np.empty(n, dtype=np.int64)

    for i, (r, h) in enumerate(zip(gt_list, pred_list)):
        gl = len(r)
        gt_len[i] = gl
        ce = LevDist.distance(r, h)
        ced[i] = ce
        l_cer[i] = ce / max(gl, 1)

        ro = ortho_fold(r)
        ho = ortho_fold(h)
        gt_olen[i] = len(ro)
        oed[i] = LevDist.distance(ro, ho)

        rw = r.split()
        hw = h.split()
        gw = len(rw)
        gt_words[i] = gw
        we = LevDist.distance(rw, hw)
        wed[i] = we
        l_wer[i] = we / max(gw, 1)

        ins, dels, subs = decompose_edits(r, h)
        ins_arr[i] = ins
        dels_arr[i] = dels
        subs_arr[i] = subs

    return {
        "gt_len": gt_len,
        "gt_olen": gt_olen,
        "gt_words": gt_words,
        f"{prefix}_ced": ced,
        f"{prefix}_oed": oed,
        f"{prefix}_wed": wed,
        f"{prefix}_cer": l_cer,
        f"{prefix}_wer": l_wer,
        f"{prefix}_ins": ins_arr,
        f"{prefix}_dels": dels_arr,
        f"{prefix}_subs": subs_arr,
    }


def agg_metrics(df_sub: pd.DataFrame, mkey: str) -> dict[str, float | int]:
    gt_c = df_sub["gt_len"].sum()
    gt_oc = df_sub["gt_olen"].sum()
    gt_w = df_sub["gt_words"].sum()
    ced = df_sub[f"{mkey}_ced"].sum()
    oed = df_sub[f"{mkey}_oed"].sum()
    wed = df_sub[f"{mkey}_wed"].sum()
    cat_n = int((df_sub[f"{mkey}_cer"] > 0.50).sum())
    n = len(df_sub)
    return {
        "N": n,
        "strict_cer": float(ced / gt_c * 100.0) if gt_c > 0 else 0.0,
        "ortho_cer": float(oed / gt_oc * 100.0) if gt_oc > 0 else 0.0,
        "corp_wer": float(wed / gt_w * 100.0) if gt_w > 0 else 0.0,
        "mean_wer": float(df_sub[f"{mkey}_wer"].mean() * 100.0) if n > 0 else 0.0,
        "cat_n": cat_n,
        "cat_pct": float(cat_n / n * 100.0) if n > 0 else 0.0,
        "ins": int(df_sub[f"{mkey}_ins"].sum()),
        "dels": int(df_sub[f"{mkey}_dels"].sum()),
        "subs": int(df_sub[f"{mkey}_subs"].sum()),
    }


def bootstrap_paired_metrics(
    df_sub: pd.DataFrame,
    key_a: str,
    key_b: str,
    n_boot: int = 10000,
    seed: int = 42,
    chunk_size: int = 500,
) -> dict[str, dict[str, float]]:
    """
    Vectorized paired bootstrap (n_boot=10,000, seed=42) over 4 metrics:
      Strict Corpus CER (%), Ortho-Folded Corpus CER (%),
      True Corpus WER (%), True Mean WER (%).
    Delta = Metric(B) - Metric(A) in percentage points.
    Exact pattern from paper1_stats_audit.bootstrap_paired_all_metrics.
    """
    rng = np.random.default_rng(seed)
    n = len(df_sub)

    gt_len = df_sub["gt_len"].to_numpy(dtype=np.float64)
    a_ced = df_sub[f"{key_a}_ced"].to_numpy(dtype=np.float64)
    b_ced = df_sub[f"{key_b}_ced"].to_numpy(dtype=np.float64)

    gt_olen = df_sub["gt_olen"].to_numpy(dtype=np.float64)
    a_oed = df_sub[f"{key_a}_oed"].to_numpy(dtype=np.float64)
    b_oed = df_sub[f"{key_b}_oed"].to_numpy(dtype=np.float64)

    gt_w = df_sub["gt_words"].to_numpy(dtype=np.float64)
    a_wed = df_sub[f"{key_a}_wed"].to_numpy(dtype=np.float64)
    b_wed = df_sub[f"{key_b}_wed"].to_numpy(dtype=np.float64)

    a_wer_line = df_sub[f"{key_a}_wer"].to_numpy(dtype=np.float64) * 100.0
    b_wer_line = df_sub[f"{key_b}_wer"].to_numpy(dtype=np.float64) * 100.0

    points = {
        "Strict Corpus CER (%)": (a_ced.sum() / gt_len.sum() * 100.0, b_ced.sum() / gt_len.sum() * 100.0),
        "Ortho-Folded Corpus CER (%)": (a_oed.sum() / gt_olen.sum() * 100.0, b_oed.sum() / gt_olen.sum() * 100.0),
        "True Corpus WER (%)": (a_wed.sum() / gt_w.sum() * 100.0, b_wed.sum() / gt_w.sum() * 100.0),
        "True Mean WER (%)": (float(a_wer_line.mean()), float(b_wer_line.mean())),
    }

    boot_store = {m: {"a": [], "b": [], "d": []} for m in points}
    for start in range(0, n_boot, chunk_size):
        b = min(chunk_size, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))

        s_gt = gt_len[idx].sum(axis=1)
        av = a_ced[idx].sum(axis=1) / s_gt * 100.0
        bv = b_ced[idx].sum(axis=1) / s_gt * 100.0
        boot_store["Strict Corpus CER (%)"]["a"].append(av)
        boot_store["Strict Corpus CER (%)"]["b"].append(bv)
        boot_store["Strict Corpus CER (%)"]["d"].append(bv - av)

        s_ogt = gt_olen[idx].sum(axis=1)
        av = a_oed[idx].sum(axis=1) / s_ogt * 100.0
        bv = b_oed[idx].sum(axis=1) / s_ogt * 100.0
        boot_store["Ortho-Folded Corpus CER (%)"]["a"].append(av)
        boot_store["Ortho-Folded Corpus CER (%)"]["b"].append(bv)
        boot_store["Ortho-Folded Corpus CER (%)"]["d"].append(bv - av)

        s_wgt = gt_w[idx].sum(axis=1)
        av = a_wed[idx].sum(axis=1) / s_wgt * 100.0
        bv = b_wed[idx].sum(axis=1) / s_wgt * 100.0
        boot_store["True Corpus WER (%)"]["a"].append(av)
        boot_store["True Corpus WER (%)"]["b"].append(bv)
        boot_store["True Corpus WER (%)"]["d"].append(bv - av)

        av = a_wer_line[idx].mean(axis=1)
        bv = b_wer_line[idx].mean(axis=1)
        boot_store["True Mean WER (%)"]["a"].append(av)
        boot_store["True Mean WER (%)"]["b"].append(bv)
        boot_store["True Mean WER (%)"]["d"].append(bv - av)

    out = {}
    for m, (ap, bp) in points.items():
        a_all = np.concatenate(boot_store[m]["a"])
        b_all = np.concatenate(boot_store[m]["b"])
        d_all = np.concatenate(boot_store[m]["d"])
        a_lo, a_hi = np.percentile(a_all, [2.5, 97.5])
        b_lo, b_hi = np.percentile(b_all, [2.5, 97.5])
        d_lo, d_hi = np.percentile(d_all, [2.5, 97.5])
        out[m] = {
            "A": ap, "A_lo": float(a_lo), "A_hi": float(a_hi),
            "B": bp, "B_lo": float(b_lo), "B_hi": float(b_hi),
            "Delta": bp - ap, "Delta_lo": float(d_lo), "Delta_hi": float(d_hi),
        }
    return out


def holm_bonferroni(pvalues: list[float], labels: list[str]) -> list[tuple[str, float, float]]:
    """Step-down Holm-Bonferroni adjusted p-values preserving monotonicity (exact from paper1_stats_audit.py)."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = np.asarray(pvalues, dtype=np.float64)[order]
    adj_sorted = np.minimum(sorted_p * np.arange(n, 0, -1), 1.0)
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = adj_sorted
    return [(labels[i], float(pvalues[i]), float(adjusted[i])) for i in range(n)]


def safe_wilcoxon(a_vals: np.ndarray, b_vals: np.ndarray) -> tuple[float, float, float]:
    """Two-sided paired Wilcoxon returning (W_stat, p_val, median_diff_pp)."""
    diff = np.asarray(b_vals, dtype=np.float64) - np.asarray(a_vals, dtype=np.float64)
    med_pp = float(np.median(diff) * 100.0)
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0, 1.0, med_pp
    stat, pval = wilcoxon(a_vals, b_vals, alternative="two-sided")
    return float(stat), float(pval), med_pp


def fmt_p(p: float) -> str:
    if p == 0.0:
        return "0.00e+00"
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Loop Detectors & Maghrebi Taxonomy (EDA/comparative_eda.py + tier1a_decode.py)
# ─────────────────────────────────────────────────────────────────────────────

def arabic_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    arab = sum(1 for c in chars if ARABIC_RANGE[0] <= ord(c) <= ARABIC_RANGE[1])
    return arab / len(chars)


def has_repetition_loop_eda(text: str, gt: str) -> bool:
    """Loop rule from EDA/comparative_eda.py (not present in GT)."""
    if not text.strip():
        return False
    for m in re.finditer(r"(.)\1{3,}", text):
        if m.group(1) * 4 not in gt:
            return True
    for n in range(2, 5):
        for m in re.finditer(r"(.{" + str(n) + r"})\1{2,}", text):
            if m.group(1) * 3 not in gt:
                return True
    return False


def detect_tail_loop_tier1a(text: str) -> tuple[bool, str]:
    """Exact 4-rule deterministic loop detector from scripts/models/hatformer/tier1a_decode.py."""
    if not text:
        return False, ""
    t = " ".join(str(text).split())
    if not t:
        return False, ""

    m_char = re.search(r"(\S)\1{3,}", t)
    if m_char:
        return True, f"char_repeat:{m_char.group(0)}"

    m_sub = re.search(r"([^\s]{2,5})\1{2,}", t)
    if m_sub:
        return True, f"substr_repeat:{m_sub.group(0)}"

    words = t.split()
    for n in (1, 2, 3, 4):
        if len(words) >= 3 * n:
            for i in range(len(words) - 3 * n + 1):
                g1 = tuple(words[i : i + n])
                g2 = tuple(words[i + n : i + 2 * n])
                g3 = tuple(words[i + 2 * n : i + 3 * n])
                if g1 == g2 == g3:
                    return True, f"word_{n}gram_repeat:{' '.join(g1)}"

    if len(words) >= 6:
        win = 7
        for i in range(max(1, len(words) - win + 1)):
            wc = Counter(words[i : i + win])
            top_w, cnt = wc.most_common(1)[0]
            if cnt >= 4 and top_w not in LOOP_WINDOW_EXEMPT_WORDS:
                return True, f"window_burst:{top_w}x{cnt}"

    return False, ""


def garbled_score(text: str) -> float:
    if not text.strip():
        return 1.0
    tokens = text.split()
    single_non_ar = sum(
        1 for t in tokens
        if len(t) == 1
        and not (ARABIC_RANGE[0] <= ord(t[0]) <= ARABIC_RANGE[1])
        and t not in '،؟!.،,()[]{}"\'-'
    )
    return max(single_non_ar / max(len(tokens), 1), 1.0 - arabic_ratio(text))


def classify_maghrebi_cat_line(
    gt: str, pred: str, ins: int, dels: int, subs: int, use_tier1a_loop: bool = True
) -> str:
    """
    Taxonomy matching EDA/comparative_eda.classify_catastrophic_line(),
    optionally using the upgraded 4-rule Tier 1A loop detector.
    """
    gt_len = len(gt)
    pred_len = len(pred)
    len_ratio = pred_len / max(gt_len, 1)
    total_e = ins + dels + subs

    if pred.strip() == "":
        return "blank_output"
    is_loop = detect_tail_loop_tier1a(pred)[0] if use_tier1a_loop else has_repetition_loop_eda(pred, gt)
    if is_loop:
        return "repetition_loop"
    if gt_len < 20:
        return "short_line"
    if len_ratio < 0.40:
        return "severe_truncation"
    ins_rate = ins / max(total_e, 1)
    if len_ratio > 1.60 or ins_rate > 0.60:
        return "severe_hallucination"
    if garbled_score(pred) > 0.40:
        return "random_noise_or_garbled"
    return "plausible_but_wrong_arabic"


# ─────────────────────────────────────────────────────────────────────────────
# Loading & Verification Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_book_preds(directory: Path, prefix: str, col_src: str, col_dst: str) -> pd.DataFrame:
    frames = []
    for bk in sorted(BOOKS.keys()):
        fp = directory / f"{prefix}_{bk}.csv"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing prediction CSV: {fp}")
        sub = pd.read_csv(fp, keep_default_na=False)
        frames.append(sub[["line_uid", "book_id", "gt_normalized", col_src]].rename(columns={col_src: col_dst}))
    return pd.concat(frames, ignore_index=True)


def verify_alignment(df_base: pd.DataFrame, df_other: pd.DataFrame, col_dst: str, label: str) -> pd.Series:
    assert len(df_other) == len(df_base), f"{label}: expected {len(df_base)} rows, got {len(df_other)}"
    base_uids = df_base["line_uid"].tolist()
    other_uids = set(df_other["line_uid"].tolist())
    assert len(other_uids) == len(df_base) and set(base_uids) == other_uids, f"{label}: line_uid set mismatch!"

    merged = df_base[["line_uid", "book_id", "gt_normalized"]].merge(
        df_other[["line_uid", "book_id", "gt_normalized", col_dst]],
        on="line_uid",
        suffixes=("_base", "_new"),
    )
    assert len(merged) == len(df_base), f"{label}: merge cardinality {len(merged)} != {len(df_base)}"
    bk_diff = int((merged["book_id_base"] != merged["book_id_new"]).sum())
    gt_raw_diff = int((merged["gt_normalized_base"] != merged["gt_normalized_new"]).sum())
    gt_norm_diff = int((merged["gt_normalized_base"].apply(norm_nfc_ws) != merged["gt_normalized_new"].apply(norm_nfc_ws)).sum())
    assert bk_diff == 0 and gt_norm_diff == 0, (
        f"{label}: alignment failure! book_diff={bk_diff}, gt_raw_diff={gt_raw_diff}, gt_norm_diff={gt_norm_diff}"
    )
    print(f"  [VERIFIED] {label:<42} N={len(merged):,} | uid=exact | gt_raw_diff={gt_raw_diff} | gt_norm_diff={gt_norm_diff}")
    return merged[col_dst].fillna("").astype(str)


# ─────────────────────────────────────────────────────────────────────────────
# Pair Reporting Helper (Bootstrap 4 metrics + Holm-Bonferroni m=8)
# ─────────────────────────────────────────────────────────────────────────────

def run_pair_analysis(
    df: pd.DataFrame,
    key_a: str,
    label_a: str,
    key_b: str,
    label_b: str,
    delta_label: str,
    section_title: str,
    note: str,
    md: list[str],
    n_boot: int = 10000,
    seed: int = 42,
):
    sep = "=" * 88
    print(f"\n{sep}\n  {section_title}\n{sep}")
    md.append(f"\n---\n\n## {section_title}\n")
    md.append(f"> {note}\n")

    metric_order = [
        "Strict Corpus CER (%)",
        "Ortho-Folded Corpus CER (%)",
        "True Corpus WER (%)",
        "True Mean WER (%)",
    ]
    splits = [("book_03", "Maghrebi"), ("book_05", "Ruq'ah"), ("book_09", "Naskh"), ("OVERALL", "Combined")]

    boot_rows = []
    for bk, script in splits:
        sub = df if bk == "OVERALL" else df[df["book_id"] == bk]
        res = bootstrap_paired_metrics(sub, key_a, key_b, n_boot=n_boot, seed=seed)
        for m_name, vals in res.items():
            boot_rows.append({"Book": bk, "Script": script, "Metric": m_name, **vals})
    df_boot = pd.DataFrame(boot_rows)

    for m_name in metric_order:
        print(f"\n  --- {m_name} ---")
        print(f"  {'Book':<9} {'Script':<9} {label_a + ' [95% CI]':>24}   {label_b + ' [95% CI]':>24}   {delta_label + ' [95% CI]':>24}")
        md.append(f"\n### {m_name}\n")
        md.append(f"| Book | Script | {label_a} | {label_a} 95% CI | {label_b} | {label_b} 95% CI | {delta_label} | $\\Delta$ 95% CI |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for _, r in df_boot[df_boot["Metric"] == m_name].iterrows():
            a_s = f"{r['A']:5.2f} [{r['A_lo']:5.2f}, {r['A_hi']:5.2f}]"
            b_s = f"{r['B']:5.2f} [{r['B_lo']:5.2f}, {r['B_hi']:5.2f}]"
            d_s = f"{r['Delta']:+5.2f} [{r['Delta_lo']:+5.2f}, {r['Delta_hi']:+5.2f}]"
            print(f"  {r['Book']:<9} {r['Script']:<9} {a_s:>24}   {b_s:>24}   {d_s:>24}")
            b_lbl = "**OVERALL**" if r["Book"] == "OVERALL" else f"`{r['Book']}`"
            md.append(
                f"| {b_lbl} | {r['Script']} | {r['A']:.2f}% | [{r['A_lo']:.2f}%, {r['A_hi']:.2f}%] | "
                f"{r['B']:.2f}% | [{r['B_lo']:.2f}%, {r['B_hi']:.2f}%] | "
                f"**{r['Delta']:+.2f}** | [{r['Delta_lo']:+.2f}, {r['Delta_hi']:+.2f}] |"
            )

    # Holm-Bonferroni over m=8 (4 splits x 2 line metrics: CER, True WER)
    raw_ps, p_labels, test_meta = [], [], []
    for bk, script in splits:
        sub = df if bk == "OVERALL" else df[df["book_id"] == bk]
        for m_lbl, suffix in [("CER", "cer"), ("True WER", "wer")]:
            w_stat, p_val, med_pp = safe_wilcoxon(
                sub[f"{key_a}_{suffix}"].to_numpy(),
                sub[f"{key_b}_{suffix}"].to_numpy(),
            )
            wtl_b_wins = int((sub[f"{key_b}_{suffix}"] < sub[f"{key_a}_{suffix}"]).sum())
            wtl_ties = int((sub[f"{key_b}_{suffix}"] == sub[f"{key_a}_{suffix}"]).sum())
            wtl_a_wins = int((sub[f"{key_b}_{suffix}"] > sub[f"{key_a}_{suffix}"]).sum())
            raw_ps.append(p_val)
            p_labels.append(f"{bk} ({script}) — {m_lbl}")
            test_meta.append((bk, script, m_lbl, w_stat, med_pp, wtl_b_wins, wtl_ties, wtl_a_wins))

    holm_res = holm_bonferroni(raw_ps, p_labels)
    print(f"\n  Holm-Bonferroni Adjusted P-Values (m=8 tests: 4 splits x 2 metrics):")
    print(f"  {'Book':<9} {'Script':<9} {'Metric':<9} {'W-stat':>11} {'Med Δ':>8} {'B/Tie/A':>15} {'Raw p':>11} {'Holm Adj p':>11}")
    md.append(f"\n### Holm–Bonferroni Adjusted P-Values ($m = 8$: 4 splits $\\times$ 2 line metrics)\n")
    md.append(f"| Book | Script | Metric | Wilcoxon $W$ | Median {delta_label} | {label_b} W/T/L | Raw $p$ | Holm–Bonferroni Adj $p$ |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for meta_t, (_, raw_p, adj_p) in zip(test_meta, holm_res):
        bk, script, m_lbl, w_stat, med_pp, bw, ties, aw = meta_t
        wtl_str = f"{bw}/{ties}/{aw}"
        sig = "✓" if adj_p < 0.05 else "✗"
        print(f"  {bk:<9} {script:<9} {m_lbl:<9} {w_stat:>11,.0f} {med_pp:>+7.2f}% {wtl_str:>15} {fmt_p(raw_p):>11} {fmt_p(adj_p):>11} {sig}")
        b_lbl = "**OVERALL**" if bk == "OVERALL" else f"`{bk}`"
        md.append(
            f"| {b_lbl} | {script} | {m_lbl} | {w_stat:,.0f} | {med_pp:+.2f} pp | `{wtl_str}` | `{fmt_p(raw_p)}` | **`{fmt_p(adj_p)}`** ({sig}) |"
        )
    return df_boot


# ─────────────────────────────────────────────────────────────────────────────
# Oracle & Complementarity Helper (Section 5)
# ─────────────────────────────────────────────────────────────────────────────

def run_oracle_and_gate(
    df: pd.DataFrame,
    k_key: str,
    k_pred_col: str,
    h_key: str,
    h_name: str,
    h_pred_col: str,
    section_title: str,
    md: list[str],
):
    sep = "=" * 88
    print(f"\n{sep}\n  {section_title}\n{sep}")
    md.append(f"\n---\n\n## {section_title}\n")
    md.append(f"> Paired complementarity between **`Kraken Tier 1A ValOpt`** and **`{h_name}`** (`CER-selected` chooses lower line CER; `WER-selected` chooses lower line WER).\n")

    splits = [("book_03", "Maghrebi"), ("book_05", "Ruq'ah"), ("book_09", "Naskh"), ("OVERALL", "Combined")]

    md.append("\n### Oracle Ceilings, Paired Win/Tie/Loss & 2×2 Catastrophic Transition Matrix\n")
    md.append("| Book | Script | N | K CER | H CER | **Oracle CER** | **Oracle Ortho CER** | K WER | H WER | **CER-Sel Oracle WER** | **WER-Sel Ceiling** | K-Cat | H-Cat | **Oracle Cat** | K-Only Cat | H-Only Cat | Both Cat | Neither | CER W/T/L (K/Tie/H) | WER W/T/L (K/Tie/H) |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    print(f"  {'Book':<8} {'K_CER':>6} {'H_CER':>6} {'OrcCER':>7} {'OrcOrt':>7} {'K_WER':>6} {'H_WER':>6} {'OrcWER_c':>8} {'OrcWER_w':>8} {'K/H/O Cat':>12} {'2x2(K/H/B/N)':>16} {'CER W/T/L':>15}")
    for bk, script in splits:
        sub = df if bk == "OVERALL" else df[df["book_id"] == bk]
        gt_l = sub["gt_len"].to_numpy()
        gt_ol = sub["gt_olen"].to_numpy()
        gt_w = sub["gt_words"].to_numpy()

        k_cer_arr = sub[f"{k_key}_cer"].to_numpy()
        h_cer_arr = sub[f"{h_key}_cer"].to_numpy()
        k_wer_arr = sub[f"{k_key}_wer"].to_numpy()
        h_wer_arr = sub[f"{h_key}_wer"].to_numpy()

        pick_k_cer = k_cer_arr <= h_cer_arr
        pick_k_wer = k_wer_arr <= h_wer_arr

        orc_ced = np.where(pick_k_cer, sub[f"{k_key}_ced"], sub[f"{h_key}_ced"])
        orc_oed = np.where(pick_k_cer, sub[f"{k_key}_oed"], sub[f"{h_key}_oed"])
        orc_wed_c = np.where(pick_k_cer, sub[f"{k_key}_wed"], sub[f"{h_key}_wed"])
        orc_wed_w = np.where(pick_k_wer, sub[f"{k_key}_wed"], sub[f"{h_key}_wed"])

        k_corp_cer = sub[f"{k_key}_ced"].sum() / gt_l.sum() * 100.0
        h_corp_cer = sub[f"{h_key}_ced"].sum() / gt_l.sum() * 100.0
        orc_cer = orc_ced.sum() / gt_l.sum() * 100.0
        orc_ort = orc_oed.sum() / gt_ol.sum() * 100.0

        k_corp_wer = sub[f"{k_key}_wed"].sum() / gt_w.sum() * 100.0
        h_corp_wer = sub[f"{h_key}_wed"].sum() / gt_w.sum() * 100.0
        orc_wer_c = orc_wed_c.sum() / gt_w.sum() * 100.0
        orc_wer_w = orc_wed_w.sum() / gt_w.sum() * 100.0

        k_cat = k_cer_arr > 0.50
        h_cat = h_cer_arr > 0.50
        orc_cat = int((np.minimum(k_cer_arr, h_cer_arr) > 0.50).sum())
        c_both = int((k_cat & h_cat).sum())
        c_k_only = int((k_cat & ~h_cat).sum())
        c_h_only = int((~k_cat & h_cat).sum())
        c_neither = int((~k_cat & ~h_cat).sum())

        cer_wtl = f"{int((k_cer_arr < h_cer_arr).sum())}/{int((k_cer_arr == h_cer_arr).sum())}/{int((k_cer_arr > h_cer_arr).sum())}"
        wer_wtl = f"{int((k_wer_arr < h_wer_arr).sum())}/{int((k_wer_arr == h_wer_arr).sum())}/{int((k_wer_arr > h_wer_arr).sum())}"

        print(
            f"  {bk:<8} {k_corp_cer:>5.2f}% {h_corp_cer:>5.2f}% {orc_cer:>6.2f}% {orc_ort:>6.2f}% "
            f"{k_corp_wer:>5.2f}% {h_corp_wer:>5.2f}% {orc_wer_c:>7.2f}% {orc_wer_w:>7.2f}% "
            f"{int(k_cat.sum()):>3}/{int(h_cat.sum()):>3}/{orc_cat:>3} "
            f"{c_k_only:>3}/{c_h_only:>3}/{c_both:>3}/{c_neither:>4} {cer_wtl:>15}"
        )
        b_lbl = "**OVERALL**" if bk == "OVERALL" else f"`{bk}`"
        md.append(
            f"| {b_lbl} | {script} | {len(sub):,} | {k_corp_cer:.2f}% | {h_corp_cer:.2f}% | "
            f"**{orc_cer:.2f}%** | **{orc_ort:.2f}%** | {k_corp_wer:.2f}% | {h_corp_wer:.2f}% | "
            f"**{orc_wer_c:.2f}%** | **{orc_wer_w:.2f}%** | {int(k_cat.sum())} | {int(h_cat.sum())} | **{orc_cat}** | "
            f"{c_k_only} | {c_h_only} | {c_both} | {c_neither} | `{cer_wtl}` | `{wer_wtl}` |"
        )

    # Disagreement Rate & Oracle Gain Concentration (Step 3 Selective Router Scoping)
    inter_cer = np.array([
        LevDist.distance(k, h) / max(len(k), len(h), 1)
        for k, h in zip(df[k_pred_col], df[h_pred_col])
    ], dtype=np.float64)

    k_ced_all = df[f"{k_key}_ced"].to_numpy()
    h_ced_all = df[f"{h_key}_ced"].to_numpy()
    orc_ced_all = np.minimum(k_ced_all, h_ced_all)
    edit_gain_line = k_ced_all - orc_ced_all  # >= 0 edits saved vs Kraken ValOpt
    total_edit_gain = float(edit_gain_line.sum())
    h_beats_k_all = df[f"{h_key}_cer"].to_numpy() < df[f"{k_key}_cer"].to_numpy()
    total_h_wins = int(h_beats_k_all.sum())

    k_wed_all = df[f"{k_key}_wed"].to_numpy()
    h_wed_all = df[f"{h_key}_wed"].to_numpy()
    orc_wed_all = np.minimum(k_wed_all, h_wed_all)
    word_gain_line = k_wed_all - orc_wed_all
    total_word_gain = float(word_gain_line.sum())

    print(f"\n  Inter-Model Disagreement Thresholds & Oracle Gain Concentration (N={len(df):,}, Total Char Edit Gain vs K-T1A = {int(total_edit_gain):,} chars):")
    md.append(f"\n### Inter-Model Disagreement Thresholds & Step 3 Router Scoping (`inter_model_cer` = `LevDist(K_T1A, H_T1A) / max_len`)\n")
    md.append(f"> Total Oracle character-edit reduction over Kraken Tier 1A ValOpt = **{int(total_edit_gain):,}** chars (word-edit reduction = **{int(total_word_gain):,}** words; lines where `{h_name}` strictly beats Kraken CER = **{total_h_wins:,} / {len(df):,}**).\n")

    def _summarize_strata(strata_defs, title_sub):
        print(f"\n  [{title_sub}]")
        print(f"  {'Stratum':<24} {'N':>5} {'%Lines':>7} {'K_CER':>7} {'H_CER':>7} {'OrcCER':>7} {'Gain(pp)':>9} {'EdSaved':>8} {'%TotGain':>9} {'%TotHWins':>10} {'HWinRate':>9}")
        md.append(f"\n#### {title_sub}\n")
        md.append("| Stratum | Lines ($N$) | % Test | K-T1A CER | H CER | Oracle CER | Within-Stratum Gain (pp) | Char Edits Saved | **% Total Oracle Edit Gain** | **% Total Word Gain** | **% of Total H-Wins** | Stratum H-Win Rate |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for lbl, mask in strata_defs:
            cnt = int(mask.sum())
            pct_l = cnt / len(df) * 100.0
            gt_s = float(df.loc[mask, "gt_len"].sum())
            k_c = float(k_ced_all[mask].sum() / gt_s * 100.0) if gt_s > 0 else 0.0
            h_c = float(h_ced_all[mask].sum() / gt_s * 100.0) if gt_s > 0 else 0.0
            o_c = float(orc_ced_all[mask].sum() / gt_s * 100.0) if gt_s > 0 else 0.0
            g_pp = k_c - o_c
            ed_s = int(edit_gain_line[mask].sum())
            pct_gain = ed_s / total_edit_gain * 100.0 if total_edit_gain > 0 else 0.0
            w_s = int(word_gain_line[mask].sum())
            pct_wgain = w_s / total_word_gain * 100.0 if total_word_gain > 0 else 0.0
            hw = int((h_beats_k_all & mask).sum())
            pct_hw = hw / total_h_wins * 100.0 if total_h_wins > 0 else 0.0
            win_rate = hw / max(cnt, 1) * 100.0
            print(f"  {lbl:<24} {cnt:>5} {pct_l:>6.1f}% {k_c:>6.2f}% {h_c:>6.2f}% {o_c:>6.2f}% {g_pp:>+8.2f} {ed_s:>8,} {pct_gain:>8.1f}% {pct_hw:>9.1f}% {win_rate:>8.1f}%")
            md.append(
                f"| {lbl} | {cnt:,} | {pct_l:.1f}% | {k_c:.2f}% | {h_c:.2f}% | {o_c:.2f}% | "
                f"{g_pp:+.2f} pp | {ed_s:,} | **{pct_gain:.1f}%** | **{pct_wgain:.1f}%** | {pct_hw:.1f}% ({hw}/{total_h_wins}) | {win_rate:.1f}% ({hw}/{cnt}) |"
            )

    cum_strata = [
        ("Low Agreement <= 0.15", inter_cer <= 0.15),
        ("Disagree > 0.15", inter_cer > 0.15),
        ("Disagree > 0.25", inter_cer > 0.25),
        ("Disagree > 0.35", inter_cer > 0.35),
    ]
    disjoint_strata = [
        ("[0.00, 0.15] Low", inter_cer <= 0.15),
        ("(0.15, 0.25] Moderate", (inter_cer > 0.15) & (inter_cer <= 0.25)),
        ("(0.25, 0.35] High", (inter_cer > 0.25) & (inter_cer <= 0.35)),
        ("> 0.35 Very High", inter_cer > 0.35),
    ]
    _summarize_strata(cum_strata, "Cumulative Disagreement Thresholds (> {0.15, 0.25, 0.35})")
    _summarize_strata(disjoint_strata, "Disjoint Disagreement Buckets")
    return inter_cer


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 88)
    print("  PAPER 1 STEP 2 CLOSING AUDIT — TIER 1A ZERO-TRAINING DECODING CALIBRATION")
    print("  AraMS-28k | Official Test N=6,744 & Full Hygiene N=6,673 | Bootstrap n=10,000")
    print("=" * 88)

    # ── 0. Load & Verify Alignment ──────────────────────────────────────────
    print("\n[0] Loading & Verifying 1-to-1 Alignment...")
    df_base = pd.read_csv(PAIRED_CSV, keep_default_na=False)
    assert len(df_base) == 6744, f"Expected 6744 baseline lines, got {len(df_base)}"
    df_base["gt_normalized"] = df_base["gt_normalized"].fillna("").astype(str)
    df_base["kraken_pred"] = df_base["kraken_pred"].fillna("").astype(str)
    df_base["hatformer_pred"] = df_base["hatformer_pred"].fillna("").astype(str)

    # Verify baseline HATFormer beams=3 matches 00_baseline_beams3
    h_00_df = load_book_preds(HAT_00_BASE_DIR, "hatformer_preds", "hatformer_pred", "hat_00base_pred")
    h_00_series = verify_alignment(df_base, h_00_df, "hat_00base_pred", "HATFormer 00_baseline_beams3")
    b00_diff = int((df_base["hatformer_pred"] != h_00_series).sum())
    print(f"  [PARITY] EDA hatformer_pred vs Kaggle 00_baseline_beams3 diffs: {b00_diff}")
    assert b00_diff == 0, f"Baseline HATFormer beams=3 mismatch on {b00_diff} lines!"

    # Load Kraken Tier 1A ValOpt & TestBest, HATFormer 01-BG & 05-ValOpt
    k_vo_df = load_book_preds(KRAKEN_VALOPT_DIR, "kraken_preds", "kraken_pred", "kraken_t1a_pred")
    k_tb_df = load_book_preds(KRAKEN_TESTBEST_DIR, "kraken_preds", "kraken_pred", "kraken_testbest_pred")
    h_01_df = load_book_preds(HAT_01_BG_DIR, "hatformer_preds", "hatformer_pred", "hat_01bg_pred")
    h_05_df = load_book_preds(HAT_05_LFB_DIR, "hatformer_preds", "hatformer_pred", "hat_05loopfb_pred")

    df = df_base[[
        "line_uid", "book_id", "script", "page_id", "image_stem", "image_path",
        "gt_normalized", "is_short_line", "is_damaged_naskh",
        "kraken_pred", "hatformer_pred",
    ]].copy()

    df["kraken_t1a_pred"] = verify_alignment(df_base, k_vo_df, "kraken_t1a_pred", "Kraken Tier 1A ValOpt (bp.2_lb.3_lm.45)")
    df["kraken_testbest_pred"] = verify_alignment(df_base, k_tb_df, "kraken_testbest_pred", "Kraken Tier 1A TestBest (bp.4_lb.3_lm.45)")
    df["hat_01bg_pred"] = verify_alignment(df_base, h_01_df, "hat_01bg_pred", "HATFormer 01_plus_blank_gate")
    df["hat_05loopfb_pred"] = verify_alignment(df_base, h_05_df, "hat_05loopfb_pred", "HATFormer 05_bg_wc_lp1.0_rp1.15_loopfb")

    # Hygiene mask (N=6,673)
    df_audit = pd.read_csv(AUDIT_CSV, keep_default_na=False)
    assert len(df_audit) == 29, f"Expected 29 audit rows, got {len(df_audit)}"
    audit_uids = set(df_audit["line_uid"].tolist())
    hygiene_mask = (~df["line_uid"].isin(audit_uids)) & (~df["page_id"].isin(DUP_FOLIO_PAGES))
    df["in_hygiene_stratum"] = hygiene_mask
    n_hyg = int(hygiene_mask.sum())
    print(f"  [HYGIENE] Official N={len(df):,} | Hygiene-Deduplicated N={n_hyg:,} (excl. 29 audit + {int(df['page_id'].isin(DUP_FOLIO_PAGES).sum())} dup folio lines)")
    assert n_hyg == 6673, f"Expected N=6673, got {n_hyg}"

    # Precompute arrays for all 6 model configurations
    print("\n[0.1] Computing Exact Character, Orthography-Folded, Word & Edit-Op Metrics...")
    model_specs = [
        ("k_base", "Kraken Greedy (Baseline)", "kraken_pred"),
        ("k_t1a", "Kraken Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`)", "kraken_t1a_pred"),
        ("k_tb", "Kraken Tier 1A TestBest (`bw25_bp0.40_lb0.30_lm0.45`)", "kraken_testbest_pred"),
        ("h_base", "HATFormer Baseline (`beams=3`)", "hatformer_pred"),
        ("h_01bg", "HATFormer `01_plus_blank_gate`", "hat_01bg_pred"),
        ("h_05lfb", "HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`)", "hat_05loopfb_pred"),
    ]
    gt_list = df["gt_normalized"].tolist()
    shared_added = False
    for mkey, _, pcol in model_specs:
        arrs = precompute_model_arrays(gt_list, df[pcol].tolist(), mkey)
        for col, vals in arrs.items():
            if col in ("gt_len", "gt_olen", "gt_words"):
                if not shared_added:
                    df[col] = vals
            else:
                df[col] = vals
        shared_added = True

    # Parity assertions
    m_kbase = agg_metrics(df, "k_base")
    m_kt1a = agg_metrics(df, "k_t1a")
    m_hbase = agg_metrics(df, "h_base")
    print(f"  [PARITY] Kraken Greedy   : Strict CER={m_kbase['strict_cer']:.2f}% | Ortho={m_kbase['ortho_cer']:.2f}% | CorpWER={m_kbase['corp_wer']:.2f}% | Cat={m_kbase['cat_n']}")
    print(f"  [PARITY] Kraken Tier 1A  : Strict CER={m_kt1a['strict_cer']:.2f}% | Ortho={m_kt1a['ortho_cer']:.2f}% | CorpWER={m_kt1a['corp_wer']:.2f}% | Cat={m_kt1a['cat_n']}")
    print(f"  [PARITY] HATFormer Base  : Strict CER={m_hbase['strict_cer']:.2f}% | Ortho={m_hbase['ortho_cer']:.2f}% | CorpWER={m_hbase['corp_wer']:.2f}% | Cat={m_hbase['cat_n']}")
    assert abs(m_kbase["strict_cer"] - 22.22) < 0.01, f"Unexpected Kraken Greedy CER: {m_kbase['strict_cer']}"
    assert abs(m_kt1a["strict_cer"] - 19.86) < 0.01, f"Unexpected Kraken Tier 1A CER: {m_kt1a['strict_cer']}"
    assert abs(m_kt1a["corp_wer"] - 49.25) < 0.01, f"Unexpected Kraken Tier 1A WER: {m_kt1a['corp_wer']}"
    assert m_kt1a["cat_n"] == 242, f"Unexpected Kraken Tier 1A Cat: {m_kt1a['cat_n']}"

    # Markdown document collector
    md: list[str] = []
    md.append("# Paper 1 Step 2 — Tier 1A Zero-Training Decoding Calibration Audit Tables\n")
    md.append("**Project:** AraMS-28k Benchmark Audit & Zero-Training Fix (Paper 1, Path B)  ")
    md.append("**Official Test Split:** $N = 6,744$ (`book_03` Maghrebi $n=3,661$, `book_05` Ruq'ah $n=2,027$, `book_09` Naskh $n=1,056$)  ")
    md.append(f"**Full Hygiene-Deduplicated Stratum:** $N = {n_hyg:,}$ (excludes all 29 rows of `arams28k_benchmark_audit_artifact.csv` + 42 duplicated folio lines on `book_03_page_103`/`104`)  ")
    md.append("**Statistical Protocol:** Corpus-weighted paired bootstrap ($n=10,000$, `seed=42`), two-sided paired Wilcoxon signed-rank tests with Holm–Bonferroni ($m=8$ per comparison pair).\n")

    # ── SECTION 1: Dual Stratum Comparison Table ────────────────────────────
    sep = "=" * 88
    print(f"\n{sep}\n  SECTION 1 — Dual Stratum Comparison Table (Official N=6,744 vs. Hygiene N=6,673)\n{sep}")
    md.append("\n---\n\n## 1. Dual Stratum Comparison Table (Official $N=6,744$ vs. Full Hygiene-Deduplicated $N=6,673$)\n")
    md.append("> Clean comparison across all baseline and Tier 1A models on both the official split and the leak-/artifact-/duplicate-free hygiene stratum.\n")

    strata = [
        ("Official Test", "OVERALL", df),
        ("Hygiene-Deduplicated", "OVERALL", df[hygiene_mask]),
    ] + [
        ("Official Test", bk, df[df["book_id"] == bk]) for bk in BOOKS
    ] + [
        ("Hygiene-Deduplicated", bk, df[hygiene_mask & (df["book_id"] == bk)]) for bk in BOOKS
    ]

    md.append("| Stratum | Split | Model | $N$ | Strict CER | Ortho CER | Corp WER | Mean WER | Cat (>50%) |")
    md.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for s_name, bk, dfs in strata:
        bk_disp = "**OVERALL**" if bk == "OVERALL" else f"`{bk}` ({BOOKS[bk]})"
        print(f"\n  [{s_name} — {bk}] (N={len(dfs):,})")
        print(f"  {'Model':<58} {'StrCER':>7} {'OrtCER':>7} {'CrpWER':>7} {'MnWER':>7} {'Cat':>8}")
        for mkey, mlabel, _ in model_specs:
            st = agg_metrics(dfs, mkey)
            print(f"  {mlabel:<58} {st['strict_cer']:>6.2f}% {st['ortho_cer']:>6.2f}% {st['corp_wer']:>6.2f}% {st['mean_wer']:>6.2f}% {st['cat_n']:>4} ({st['cat_pct']:.2f}%)")
            md.append(
                f"| {s_name} | {bk_disp} | {mlabel} | {st['N']:,} | "
                f"{st['strict_cer']:.2f}% | {st['ortho_cer']:.2f}% | {st['corp_wer']:.2f}% | {st['mean_wer']:.2f}% | "
                f"{st['cat_n']} ({st['cat_pct']:.2f}%) |"
            )

    # ── SECTION 2: Pair A — Kraken Gain ─────────────────────────────────────
    run_pair_analysis(
        df=df,
        key_a="k_base",
        label_a="Kraken Greedy",
        key_b="k_t1a",
        label_b="Kraken T1A",
        delta_label="Δ (T1A − Greedy)",
        section_title="2. Pair A — Kraken Gain: Greedy → Tier 1A ValOpt (`bw25_bp0.20_lb0.30_lm0.45`)",
        note="Leak-free CTC prefix beam search (`beam=25, blank_pen=0.20, len_bonus=0.30, lm_wt=0.45`) with 5-gram KN LM trained solely on `split=='train'`. Negative Δ = Tier 1A reduces error.",
        md=md,
    )

    # ── SECTION 3: Pair B — Paradox Reversal ────────────────────────────────
    run_pair_analysis(
        df=df,
        key_a="h_01bg",
        label_a="HAT 01-BG",
        key_b="k_t1a",
        label_b="Kraken T1A",
        delta_label="Δ (K_T1A − H_BG)",
        section_title="3A. Pair B1 — Paradox Reversal: Kraken Tier 1A vs. HATFormer `01_plus_blank_gate` (LP=0.0+BG)",
        note="Tests whether Kraken Tier 1A beats clean HATFormer (`LP=0.0 + Blank-Gate`) simultaneously on Strict CER, Ortho CER, Corpus WER, and Mean WER across all 3 scripts. Negative Δ = Kraken Tier 1A has lower error.",
        md=md,
    )

    run_pair_analysis(
        df=df,
        key_a="h_05lfb",
        label_a="HAT ValOpt",
        key_b="k_t1a",
        label_b="Kraken T1A",
        delta_label="Δ (K_T1A − H_VO)",
        section_title="3B. Pair B2 — Paradox Reversal: Kraken Tier 1A vs. HATFormer Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`)",
        note="Tests whether Kraken Tier 1A beats Val-selected HATFormer Tier 1A simultaneously on CER and WER across all 3 scripts. Negative Δ = Kraken Tier 1A has lower error.",
        md=md,
    )

    # ── SECTION 4: Pair C — HATFormer Tier 1A Diagnostic ────────────────────
    run_pair_analysis(
        df=df,
        key_a="h_base",
        label_a="HAT Base",
        key_b="h_05lfb",
        label_b="HAT ValOpt",
        delta_label="Δ (VO − Base)",
        section_title="4A. Pair C — HATFormer Tier 1A Diagnostic: Baseline → Val-Selected (`05_bg_wc_lp1.0_rp1.15_loopfb`)",
        note="Evaluates why zero-training decoding saturates on HATFormer: `LP=1.0 + RP=1.15 + LoopFB` improves Naskh Corpus CER (`-0.19 pp`, 95% bootstrap CI excludes 0 due to tail-loop rescues, though symmetric 73/71 per-line ranks make Wilcoxon CER non-significant) while significantly degrading Maghrebi CER (`+0.15 pp`) and WER (`+0.57 pp`). Negative Δ = Val-Selected reduces error.",
        md=md,
    )

    # Quick summary of 01_plus_blank_gate vs Baseline (only 3 blank crops differ)
    print("\n  [Note Pair C2] HATFormer Baseline vs. 01_plus_blank_gate differs strictly on the 3 corrupt blank crops:")
    for uid in ["book_05_line_004216", "book_05_line_004741", "book_09_line_006514"]:
        r = df[df["line_uid"] == uid].iloc[0]
        print(f"    {uid}: '{r['hatformer_pred']}' (CER={r['h_base_cer']*100:.1f}%) -> '{r['hat_01bg_pred']}' (CER={r['h_01bg_cer']*100:.1f}%)")
    md.append("\n> **Blank-Gate Surgical Effect (`01_plus_blank_gate` vs. Baseline):** Identical on all 6,741 legible lines ($p=1.0000$); eliminates hallucinated Arabic boilerplate (`صلى الله عليه وسلم`, `ونحو ذلك`) on the 3 verified blank margin crops (`book_05_line_004216`, `004741`, `book_09_line_006514`) with 0 false positives across val+test.\n")

    # ── SECTION 5: Post-Tier-1A Oracle & 2×2 Complementarity + Router Scoping ─
    inter_bg = run_oracle_and_gate(
        df=df,
        k_key="k_t1a",
        k_pred_col="kraken_t1a_pred",
        h_key="h_01bg",
        h_name="HATFormer 01_plus_blank_gate",
        h_pred_col="hat_01bg_pred",
        section_title="5A. Post-Tier-1A Oracle Ceiling, 2×2 Matrix & Disagreement Scoping (Kraken ValOpt + HATFormer 01-BG)",
        md=md,
    )
    inter_vo = run_oracle_and_gate(
        df=df,
        k_key="k_t1a",
        k_pred_col="kraken_t1a_pred",
        h_key="h_05lfb",
        h_name="HATFormer 05_loopfb",
        h_pred_col="hat_05loopfb_pred",
        section_title="5B. Post-Tier-1A Oracle Ceiling, 2×2 Matrix & Disagreement Scoping (Kraken ValOpt + HATFormer ValOpt)",
        md=md,
    )

    df["inter_t1a_01bg_cer"] = inter_bg
    df["inter_t1a_05lfb_cer"] = inter_vo

    # ── SECTION 6: Why Zero-Training Decoding Saturates on HATFormer Maghrebi
    print(f"\n{sep}\n  SECTION 6 — Diagnostic Taxonomy of HATFormer Maghrebi Catastrophic Failures (book_03)\n{sep}")
    md.append("\n---\n\n## 6. Reproducible Diagnostic Table — Why Zero-Training Decoding Saturates on HATFormer Maghrebi (`book_03`)\n")
    md.append("> Classification follows `EDA/comparative_eda.classify_catastrophic_line` (`short_line`: GT `< 20` chars; `severe_truncation`: len ratio `< 0.40`; `severe_hallucination`: len ratio `> 1.60` or insertion share `> 60%`; `random_noise_or_garbled`: garbled score `> 0.40`; remainder: `plausible_but_wrong_arabic`). Checked under both `comparative_eda.py` loop regex and `tier1a_decode.py` 4-rule loop detector.\n")

    b03 = df[df["book_id"] == "book_03"].copy()
    n_b03 = len(b03)
    modes_order = [
        "plausible_but_wrong_arabic",
        "short_line",
        "severe_hallucination",
        "severe_truncation",
        "repetition_loop",
        "random_noise_or_garbled",
        "blank_output",
    ]

    for cfg_lbl, hkey, pcol in [
        ("HATFormer `01_plus_blank_gate` (= Baseline `B=3`, 528 Cat lines)", "h_01bg", "hat_01bg_pred"),
        ("HATFormer `05_bg_wc_lp1.0_rp1.15_loopfb` (Post-Tier 1A, 527 Cat lines)", "h_05lfb", "hat_05loopfb_pred"),
    ]:
        sub_cat = b03[b03[f"{hkey}_cer"] > 0.50].copy()
        n_cat = len(sub_cat)

        # Corpus-wide loop incidence & raw char insertion share on ALL 3,661 Maghrebi lines
        loop_t1a_all = b03[pcol].apply(lambda p: detect_tail_loop_tier1a(p)[0]).to_numpy()
        loop_eda_all = [has_repetition_loop_eda(p, g) for p, g in zip(b03[pcol], b03["gt_normalized"])]
        loop_eda_all = np.asarray(loop_eda_all, dtype=bool)

        tot_ins = int(b03[f"{hkey}_ins"].sum())
        ins_t1a = int(b03.loc[loop_t1a_all, f"{hkey}_ins"].sum())
        ins_eda = int(b03.loc[loop_eda_all, f"{hkey}_ins"].sum())

        tags_t1a = [
            classify_maghrebi_cat_line(g, p, int(i), int(d), int(s), use_tier1a_loop=True)
            for g, p, i, d, s in zip(
                sub_cat["gt_normalized"], sub_cat[pcol],
                sub_cat[f"{hkey}_ins"], sub_cat[f"{hkey}_dels"], sub_cat[f"{hkey}_subs"]
            )
        ]
        tags_eda = [
            classify_maghrebi_cat_line(g, p, int(i), int(d), int(s), use_tier1a_loop=False)
            for g, p, i, d, s in zip(
                sub_cat["gt_normalized"], sub_cat[pcol],
                sub_cat[f"{hkey}_ins"], sub_cat[f"{hkey}_dels"], sub_cat[f"{hkey}_subs"]
            )
        ]
        vc_t1a = pd.Series(tags_t1a).value_counts()
        vc_eda = pd.Series(tags_eda).value_counts()

        print(f"\n  [{cfg_lbl}] Cat = {n_cat}/{n_b03} ({n_cat/n_b03*100:.2f}%)")
        print(f"    Corpus-wide loop incidence (tier1a 4-rule): {int(loop_t1a_all.sum())}/{n_b03} ({loop_t1a_all.mean()*100:.2f}%) | Insertion share: {ins_t1a}/{tot_ins} ({ins_t1a/max(tot_ins,1)*100:.2f}%)")
        print(f"    Corpus-wide loop incidence (eda regex)    : {int(loop_eda_all.sum())}/{n_b03} ({loop_eda_all.mean()*100:.2f}%) | Insertion share: {ins_eda}/{tot_ins} ({ins_eda/max(tot_ins,1)*100:.2f}%)")
        print(f"    {'Failure Category':<28} {'Tier1A-Loop N (%)':>18} {'EDA-Regex N (%)':>18} {'% of All b03':>13}")

        md.append(f"\n### {cfg_lbl}\n")
        md.append(f"- **Catastrophic Maghrebi lines (`CER > 50%`):** **{n_cat} / {n_b03}** (`{n_cat/n_b03*100:.2f}%`)\n")
        md.append(f"- **Corpus-wide Maghrebi loop incidence (`tier1a_decode` 4-rule):** **{int(loop_t1a_all.sum())} / {n_b03}** (`{loop_t1a_all.mean()*100:.2f}%`), accounting for **{ins_t1a:,} / {tot_ins:,}** (**{ins_t1a/max(tot_ins,1)*100:.2f}% < 1%**) of Maghrebi raw character insertions.\n")
        md.append(f"- **Corpus-wide Maghrebi loop incidence (`comparative_eda` regex):** **{int(loop_eda_all.sum())} / {n_b03}** (`{loop_eda_all.mean()*100:.2f}%`), accounting for **{ins_eda:,} / {tot_ins:,}** (**{ins_eda/max(tot_ins,1)*100:.2f}%**) of Maghrebi raw character insertions.\n\n")
        md.append("| Failure Category | Count (4-Rule Loop) | % of Cat Lines | Count (`EDA` Regex) | % of Cat (`EDA`) | % of All `book_03` (3,661) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

        for mode in modes_order:
            c1 = int(vc_t1a.get(mode, 0))
            c2 = int(vc_eda.get(mode, 0))
            if c1 == 0 and c2 == 0 and mode in ("random_noise_or_garbled", "blank_output"):
                continue
            print(f"    {mode:<28} {c1:>4} ({c1/n_cat*100:>5.2f}%)       {c2:>4} ({c2/n_cat*100:>5.2f}%)       {c1/n_b03*100:>6.2f}%")
            md.append(
                f"| `{mode}` | {c1} | {c1/n_cat*100:.2f}% | {c2} | {c2/n_cat*100:.2f}% | {c1/n_b03*100:.2f}% |"
            )
        non_loop_plaus = int(vc_t1a.get("plausible_but_wrong_arabic", 0))
        non_loop_all_cat = int(n_cat - vc_t1a.get("repetition_loop", 0))
        print(f"    -> Non-loop plausible Arabic: {non_loop_plaus}/{n_cat} ({non_loop_plaus/n_cat*100:.2f}%) | Total non-loop catastrophic: {non_loop_all_cat}/{n_cat} ({non_loop_all_cat/n_cat*100:.2f}%)")
        md.append(f"\n> **Saturation Finding:** **`{non_loop_plaus}/{n_cat}` ({non_loop_plaus/n_cat*100:.2f}%)** of catastrophic Maghrebi outputs are **`plausible_but_wrong_arabic`** (grammatically fluent Arabic words of normal length without repetition loops), while total non-loop failures represent `{non_loop_all_cat}/{n_cat}` ({non_loop_all_cat/n_cat*100:.2f}%). Because beam decoding cannot override a miscalibrated encoder-decoder conditional visual likelihood on Maghrebi ligatures, decoding hyperparameters (`LP`, `RP`, `LoopFB`) saturate at `~36.4%–36.5%` CER and motivate cross-model Kraken↔HATFormer arbitration in Step 3.\n")

    # ── Save Merged CSV & Markdown Report ────────────────────────────────────
    save_cols = [
        "line_uid", "book_id", "script", "page_id", "image_stem", "image_path",
        "gt_normalized", "gt_len", "gt_olen", "gt_words",
        "is_short_line", "is_damaged_naskh", "in_hygiene_stratum",
        "kraken_pred", "kraken_t1a_pred", "kraken_testbest_pred",
        "hatformer_pred", "hat_01bg_pred", "hat_05loopfb_pred",
        "k_base_cer", "k_base_wer", "k_t1a_cer", "k_t1a_wer", "k_t1a_ced", "k_t1a_wed",
        "h_base_cer", "h_base_wer", "h_01bg_cer", "h_01bg_wer", "h_01bg_ced", "h_01bg_wed",
        "h_05lfb_cer", "h_05lfb_wer", "h_05lfb_ced", "h_05lfb_wed",
        "inter_t1a_01bg_cer", "inter_t1a_05lfb_cer",
    ]
    df[save_cols].to_csv(OUT_CSV, index=False, encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n{sep}")
    print(f"  [SAVED] Merged Tier 1A paired CSV : {OUT_CSV} ({len(df):,} rows, {len(save_cols)} cols)")
    print(f"  [SAVED] Formatted Markdown Tables : {OUT_MD}")
    print("  ✓ Paper 1 Step 2 Closing Audit Complete.")
    print(sep)


if __name__ == "__main__":
    main()
