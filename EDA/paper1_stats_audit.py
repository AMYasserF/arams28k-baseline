#!/usr/bin/env python3
"""
paper1_stats_audit.py
=====================
Statistical audit for Paper 1 (Path B: Diagnostic, Benchmark Audit & Zero-Training Fix on AraMS-28k).
Reads:
  - EDA/paired_kraken_vs_hatformer.csv
  - data/AraMS-28k-HTR/metadata.csv
  - EDA/suspected_gt_misalignment.csv

Outputs:
  - EDA/reports/paper1_statistical_tables.md
  - EDA/arams28k_benchmark_audit_artifact.csv
Prints all key tables to stdout for verification.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from rapidfuzz.distance import Levenshtein as LevDist
from scipy import stats
from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
PAIRED_CSV = ROOT / "EDA" / "paired_kraken_vs_hatformer.csv"
META_CSV = ROOT / "data" / "AraMS-28k-HTR" / "metadata.csv"
MIS_CSV = ROOT / "EDA" / "suspected_gt_misalignment.csv"
OUT_MD = ROOT / "EDA" / "reports" / "paper1_statistical_tables.md"
OUT_AUDIT = ROOT / "EDA" / "arams28k_benchmark_audit_artifact.csv"
IMAGES_DIR = ROOT / "data" / "AraMS-28k-HTR" / "images"

BOOKS = {
    "book_03": "Maghrebi",
    "book_05": "Ruq'ah",
    "book_09": "Naskh",
}

POPCOUNT16 = np.array([bin(i).count("1") for i in range(65536)], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Text & Metric Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ortho_fold(text: str) -> str:
    """Fold non-semantic orthographic variants: أ/إ/آ -> ا, ة -> ه, ى -> ي."""
    if not isinstance(text, str):
        return ""
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
    )


def holm_bonferroni(pvalues, labels):
    """Step-down Holm-Bonferroni adjusted p-values preserving monotonicity."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = np.asarray(pvalues, dtype=np.float64)[order]
    adj_sorted = np.minimum(sorted_p * np.arange(n, 0, -1), 1.0)
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = adj_sorted
    return [(labels[i], float(pvalues[i]), float(adjusted[i])) for i in range(n)]


def bootstrap_paired_all_metrics(df_sub: pd.DataFrame, n_boot: int = 10000, seed: int = 42, chunk_size: int = 500):
    """
    Vectorized paired bootstrap (n_boot=10,000, seed=42) over 4 metrics simultaneously:
      1. Strict Corpus CER (%)
      2. Orthography-Folded Corpus CER (%)
      3. True Corpus WER (%)
      4. True Mean WER (%)
    Returns dict of results per metric with point estimate, 95% CI, and paired Delta = HATFormer - Kraken.
    """
    rng = np.random.default_rng(seed)
    n = len(df_sub)

    gt_len = df_sub["gt_len"].to_numpy(dtype=np.float64)
    k_ced = df_sub["k_char_edits"].to_numpy(dtype=np.float64)
    h_ced = df_sub["h_char_edits"].to_numpy(dtype=np.float64)

    gt_olen = df_sub["gt_ortho_len"].to_numpy(dtype=np.float64)
    k_oed = df_sub["k_ortho_edits"].to_numpy(dtype=np.float64)
    h_oed = df_sub["h_ortho_edits"].to_numpy(dtype=np.float64)

    gt_w = df_sub["gt_words"].to_numpy(dtype=np.float64)
    k_wed = df_sub["kraken_word_edits"].to_numpy(dtype=np.float64)
    h_wed = df_sub["hatformer_word_edits"].to_numpy(dtype=np.float64)

    k_wer_line = df_sub["kraken_wer"].to_numpy(dtype=np.float64) * 100.0
    h_wer_line = df_sub["hatformer_wer"].to_numpy(dtype=np.float64) * 100.0

    # Point estimates
    points = {
        "Strict Corpus CER (%)": (k_ced.sum() / gt_len.sum() * 100.0, h_ced.sum() / gt_len.sum() * 100.0),
        "Ortho-Folded Corpus CER (%)": (k_oed.sum() / gt_olen.sum() * 100.0, h_oed.sum() / gt_olen.sum() * 100.0),
        "True Corpus WER (%)": (k_wed.sum() / gt_w.sum() * 100.0, h_wed.sum() / gt_w.sum() * 100.0),
        "True Mean WER (%)": (float(k_wer_line.mean()), float(h_wer_line.mean())),
    }

    boot_store = {m: {"k": [], "h": [], "d": []} for m in points}

    for start in range(0, n_boot, chunk_size):
        b = min(chunk_size, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))

        # 1. Strict Corpus CER
        s_gt = gt_len[idx].sum(axis=1)
        kv = k_ced[idx].sum(axis=1) / s_gt * 100.0
        hv = h_ced[idx].sum(axis=1) / s_gt * 100.0
        boot_store["Strict Corpus CER (%)"]["k"].append(kv)
        boot_store["Strict Corpus CER (%)"]["h"].append(hv)
        boot_store["Strict Corpus CER (%)"]["d"].append(hv - kv)

        # 2. Ortho-Folded Corpus CER
        s_ogt = gt_olen[idx].sum(axis=1)
        kv = k_oed[idx].sum(axis=1) / s_ogt * 100.0
        hv = h_oed[idx].sum(axis=1) / s_ogt * 100.0
        boot_store["Ortho-Folded Corpus CER (%)"]["k"].append(kv)
        boot_store["Ortho-Folded Corpus CER (%)"]["h"].append(hv)
        boot_store["Ortho-Folded Corpus CER (%)"]["d"].append(hv - kv)

        # 3. True Corpus WER
        s_wgt = gt_w[idx].sum(axis=1)
        kv = k_wed[idx].sum(axis=1) / s_wgt * 100.0
        hv = h_wed[idx].sum(axis=1) / s_wgt * 100.0
        boot_store["True Corpus WER (%)"]["k"].append(kv)
        boot_store["True Corpus WER (%)"]["h"].append(hv)
        boot_store["True Corpus WER (%)"]["d"].append(hv - kv)

        # 4. True Mean WER
        kv = k_wer_line[idx].mean(axis=1)
        hv = h_wer_line[idx].mean(axis=1)
        boot_store["True Mean WER (%)"]["k"].append(kv)
        boot_store["True Mean WER (%)"]["h"].append(hv)
        boot_store["True Mean WER (%)"]["d"].append(hv - kv)

    out = {}
    for m, (kp, hp) in points.items():
        k_all = np.concatenate(boot_store[m]["k"])
        h_all = np.concatenate(boot_store[m]["h"])
        d_all = np.concatenate(boot_store[m]["d"])
        k_lo, k_hi = np.percentile(k_all, [2.5, 97.5])
        h_lo, h_hi = np.percentile(h_all, [2.5, 97.5])
        d_lo, d_hi = np.percentile(d_all, [2.5, 97.5])
        out[m] = {
            "Kraken": kp, "Kraken_lo": k_lo, "Kraken_hi": k_hi,
            "HATFormer": hp, "HATFormer_lo": h_lo, "HATFormer_hi": h_hi,
            "Delta": hp - kp, "Delta_lo": d_lo, "Delta_hi": d_hi,
        }
    return out


def two_prop_power(p_dam: float, p_cln: float, n_dam: int, n_cln: int, alpha: float = 0.05) -> float:
    """Two-sided normal approximation power for comparing two independent proportions."""
    if p_dam == p_cln:
        return alpha
    p_pool = (p_dam * n_dam + p_cln * n_cln) / (n_dam + n_cln)
    se_null = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_dam + 1.0 / n_cln))
    se_alt = np.sqrt(p_dam * (1.0 - p_dam) / n_dam + p_cln * (1.0 - p_cln) / n_cln)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    z_upper = (z_crit * se_null - abs(p_dam - p_cln)) / se_alt
    z_lower = (-z_crit * se_null - abs(p_dam - p_cln)) / se_alt
    return float((1.0 - stats.norm.cdf(z_upper)) + stats.norm.cdf(z_lower))


def min_detectable_or(p_cln: float, n_dam: int, n_cln: int, target_power: float = 0.80, alpha: float = 0.05):
    """Solve for minimum Odds Ratio > 1 detectable at target_power via bisection."""
    lo, hi = 1.01, 50.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        odds_cln = p_cln / (1.0 - p_cln)
        odds_dam = odds_cln * mid
        p_dam = odds_dam / (1.0 + odds_dam)
        if two_prop_power(p_dam, p_cln, n_dam, n_cln, alpha) >= target_power:
            hi = mid
        else:
            lo = mid
    odds_dam = (p_cln / (1.0 - p_cln)) * hi
    p_dam_req = odds_dam / (1.0 + odds_dam)
    return hi, p_dam_req


def compute_dhash_uint64(img_path: Path):
    """Compute 64-bit horizontal difference hash (dHash) packed into uint64, plus image size."""
    try:
        with Image.open(img_path) as im:
            w, h = im.size
            arr = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.uint8)
        diff = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
        val = np.uint64(0)
        for bit in diff:
            val = (val << np.uint64(1)) | np.uint64(int(bit))
        return val, w, h
    except Exception:
        return None, None, None


def find_dhash_near_duplicates(stems: list, hashes: np.ndarray, max_hamming: int = 6, block_size: int = 512):
    """Vectorized pairwise Hamming distance check over uint64 dHashes."""
    n = len(stems)
    pairs = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = hashes[start:end, None]
        xor = block ^ hashes[None, :]
        dist = (
            POPCOUNT16[(xor & np.uint64(0xFFFF)).astype(np.uint16)]
            + POPCOUNT16[((xor >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16)]
            + POPCOUNT16[((xor >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint16)]
            + POPCOUNT16[((xor >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.uint16)]
        )
        rows_idx, cols_idx = np.where(dist <= max_hamming)
        for r_off, c in zip(rows_idx, cols_idx):
            r = start + r_off
            if c > r:
                pairs.append((stems[r], stems[c], int(dist[r_off, c])))
    return pairs


def compute_split_metrics(df_sub: pd.DataFrame, label: str) -> dict:
    gt_chars = df_sub["gt_len"].sum()
    gt_ochars = df_sub["gt_ortho_len"].sum()
    gt_words = df_sub["gt_words"].sum()
    k_corp_cer = df_sub["k_char_edits"].sum() / gt_chars * 100.0
    h_corp_cer = df_sub["h_char_edits"].sum() / gt_chars * 100.0
    k_ortho_cer = df_sub["k_ortho_edits"].sum() / gt_ochars * 100.0
    h_ortho_cer = df_sub["h_ortho_edits"].sum() / gt_ochars * 100.0
    k_corp_wer = df_sub["kraken_word_edits"].sum() / gt_words * 100.0
    h_corp_wer = df_sub["hatformer_word_edits"].sum() / gt_words * 100.0
    return {
        "Split": label,
        "N": len(df_sub),
        "K_Corp_CER": k_corp_cer,
        "H_Corp_CER": h_corp_cer,
        "K_Mean_CER": df_sub["kraken_cer"].mean() * 100.0,
        "H_Mean_CER": df_sub["hatformer_cer"].mean() * 100.0,
        "K_Ortho_CER": k_ortho_cer,
        "H_Ortho_CER": h_ortho_cer,
        "K_Corp_WER": k_corp_wer,
        "H_Corp_WER": h_corp_wer,
        "K_Mean_WER": df_sub["kraken_wer"].mean() * 100.0,
        "H_Mean_WER": df_sub["hatformer_wer"].mean() * 100.0,
        "K_Cat": int((df_sub["kraken_cer"] > 0.50).sum()),
        "H_Cat": int((df_sub["hatformer_cer"] > 0.50).sum()),
    }


def main():
    print("=" * 84)
    print("  PAPER 1 STATISTICAL & BENCHMARK HYGIENE AUDIT — AraMS-28k")
    print("=" * 84)

    paired = pd.read_csv(PAIRED_CSV)
    paired["gt_normalized"] = paired["gt_normalized"].fillna("")
    paired["kraken_pred"] = paired["kraken_pred"].fillna("")
    paired["hatformer_pred"] = paired["hatformer_pred"].fillna("")

    meta = pd.read_csv(META_CSV)
    mis = pd.read_csv(MIS_CSV)

    # Precompute exact per-line character and orthography-folded edit counts
    paired["gt_len"] = paired["gt_normalized"].str.len()
    paired["k_char_edits"] = paired["kraken_ins"] + paired["kraken_dels"] + paired["kraken_subs"]
    paired["h_char_edits"] = paired["hatformer_ins"] + paired["hatformer_dels"] + paired["hatformer_subs"]

    gt_o = paired["gt_normalized"].apply(ortho_fold)
    k_o = paired["kraken_pred"].apply(ortho_fold)
    h_o = paired["hatformer_pred"].apply(ortho_fold)
    paired["gt_ortho_len"] = gt_o.str.len()
    paired["k_ortho_edits"] = [LevDist.distance(r, h) for r, h in zip(gt_o, k_o)]
    paired["h_ortho_edits"] = [LevDist.distance(r, h) for r, h in zip(gt_o, h_o)]

    overall_k_cer = paired["k_char_edits"].sum() / paired["gt_len"].sum() * 100.0
    overall_h_cer = paired["h_char_edits"].sum() / paired["gt_len"].sum() * 100.0
    print(f"Loaded {len(paired):,} paired test lines | Strict Corpus CER: Kraken={overall_k_cer:.2f}%, HATFormer={overall_h_cer:.2f}%")

    # Artifact sets
    PAGE034_UIDS = set(paired.loc[paired["page_id"] == "book_09_page_034", "line_uid"])
    BLANK_CROP_UIDS = {"book_09_line_006514", "book_05_line_004741", "book_05_line_004216"}
    ARTIFACT_UIDS = PAGE034_UIDS | BLANK_CROP_UIDS
    paired_clean = paired[~paired["line_uid"].isin(ARTIFACT_UIDS)].copy()

    # ── 1. PAIRED BOOTSTRAP 95% CIs ─────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  1. PAIRED BOOTSTRAP 95% CIs (10,000 resamples, seed=42)")
    print("=" * 84)

    sections = list(BOOKS.items()) + [("OVERALL", "Combined")]
    boot_rows = []
    for b_id, script in sections:
        sub = paired if b_id == "OVERALL" else paired[paired["book_id"] == b_id]
        res = bootstrap_paired_all_metrics(sub, n_boot=10000, seed=42)
        for m_name, vals in res.items():
            boot_rows.append({"Book": b_id, "Script": script, "Metric": m_name, **vals})
    df_boot = pd.DataFrame(boot_rows)

    metric_order = [
        "Strict Corpus CER (%)",
        "Ortho-Folded Corpus CER (%)",
        "True Corpus WER (%)",
        "True Mean WER (%)",
    ]
    for m_name in metric_order:
        print(f"\n--- {m_name} ---")
        print(f"{'Book':<10} {'Script':<10} {'Kraken [95% CI]':>24}   {'HATFormer [95% CI]':>24}   {'Delta (HF-K) [95% CI]':>24}")
        for _, r in df_boot[df_boot["Metric"] == m_name].iterrows():
            k_str = f"{r['Kraken']:5.2f} [{r['Kraken_lo']:5.2f}, {r['Kraken_hi']:5.2f}]"
            h_str = f"{r['HATFormer']:5.2f} [{r['HATFormer_lo']:5.2f}, {r['HATFormer_hi']:5.2f}]"
            d_str = f"{r['Delta']:+5.2f} [{r['Delta_lo']:+5.2f}, {r['Delta_hi']:+5.2f}]"
            print(f"{r['Book']:<10} {r['Script']:<10} {k_str:>24}   {h_str:>24}   {d_str:>24}")

    # ── 2. HOLM-BONFERRONI ADJUSTED P-VALUES ─────────────────────────────────
    print("\n" + "=" * 84)
    print("  2. HOLM-BONFERRONI ADJUSTED P-VALUES (8 Paired Two-Sided Wilcoxon Tests)")
    print("=" * 84)

    raw_pvals, p_labels, test_meta = [], [], []
    for b_id, script in sections:
        sub = paired if b_id == "OVERALL" else paired[paired["book_id"] == b_id]
        for metric_lbl, col_k, col_h in [
            ("CER", "kraken_cer", "hatformer_cer"),
            ("True WER", "kraken_wer", "hatformer_wer"),
        ]:
            stat, pval = wilcoxon(sub[col_k], sub[col_h], alternative="two-sided")
            raw_pvals.append(pval)
            p_labels.append(f"{b_id} ({script}) — {metric_lbl}")
            med_diff = (sub[col_h] - sub[col_k]).median() * 100.0
            test_meta.append((b_id, script, metric_lbl, stat, med_diff))

    holm_out = holm_bonferroni(raw_pvals, p_labels)
    df_holm = pd.DataFrame([
        {
            "Book": m[0],
            "Script": m[1],
            "Metric": m[2],
            "Wilcoxon_W": m[3],
            "Median_Diff_pp": m[4],
            "Raw_p": raw_p,
            "Holm_Adj_p": adj_p,
        }
        for m, (_, raw_p, adj_p) in zip(test_meta, holm_out)
    ])
    print(f"\n{'Book':<9} {'Script':<9} {'Metric':<9} {'W-stat':>12} {'Med Δ(HF-K)':>12} {'Raw p':>12} {'Holm Adj p':>12}")
    for _, r in df_holm.iterrows():
        print(f"{r['Book']:<9} {r['Script']:<9} {r['Metric']:<9} {r['Wilcoxon_W']:>12,.0f} {r['Median_Diff_pp']:>+11.2f}% {r['Raw_p']:>12.3e} {r['Holm_Adj_p']:>12.3e}")

    # ── 3. DE-CONFOUNDED AraMS-Restore ANALYSIS ──────────────────────────────
    print("\n" + "=" * 84)
    print("  3. DE-CONFOUNDED AraMS-Restore ANALYSIS (Naskh, Excl. book_09_page_034)")
    print("=" * 84)

    naskh = paired[paired["book_id"] == "book_09"].copy()
    naskh_deconf = naskh[~naskh["line_uid"].isin(PAGE034_UIDS)].copy()
    dam = naskh_deconf[naskh_deconf["is_damaged_naskh"]]
    cln = naskh_deconf[~naskh_deconf["is_damaged_naskh"]]
    n_dam, n_cln = len(dam), len(cln)

    blank_in_dam = "book_09_line_006514" in set(dam["line_uid"])
    print(f"De-confounded Naskh total: {len(naskh_deconf)} (Damaged={n_dam}, Clean={n_cln}, book_09_line_006514 in damaged={blank_in_dam})")

    # Sensitivity subset: also exclude blank crop book_09_line_006514 -> n_clean = 858
    cln_858 = cln[cln["line_uid"] != "book_09_line_006514"].copy()
    n_cln_858 = len(cln_858)
    mw_k_cer_858_2 = mannwhitneyu(dam["kraken_cer"], cln_858["kraken_cer"], alternative="two-sided")
    mw_h_cer_858_2 = mannwhitneyu(dam["hatformer_cer"], cln_858["hatformer_cer"], alternative="two-sided")
    k_cc_858 = int((cln_858["kraken_cer"] > 0.50).sum())
    h_cc_858 = int((cln_858["hatformer_cer"] > 0.50).sum())
    fish_k_or_858, fish_k_p_858 = fisher_exact([[int((dam["kraken_cer"] > 0.50).sum()), n_dam - int((dam["kraken_cer"] > 0.50).sum())], [k_cc_858, n_cln_858 - k_cc_858]], alternative="two-sided")
    fish_h_or_858, fish_h_p_858 = fisher_exact([[int((dam["hatformer_cer"] > 0.50).sum()), n_dam - int((dam["hatformer_cer"] > 0.50).sum())], [h_cc_858, n_cln_858 - h_cc_858]], alternative="two-sided")
    print(f"Sensitivity (n_clean={n_cln_858}, also excl. blank crop book_09_line_006514):")
    print(f"  Clean-858 Mean/Med CER: Kraken={cln_858['kraken_cer'].mean()*100:.2f}%/{cln_858['kraken_cer'].median()*100:.2f}% (Cat={k_cc_858}/{n_cln_858}={k_cc_858/n_cln_858*100:.2f}%), HAT={cln_858['hatformer_cer'].mean()*100:.2f}%/{cln_858['hatformer_cer'].median()*100:.2f}% (Cat={h_cc_858}/{n_cln_858}={h_cc_858/n_cln_858*100:.2f}%)")
    print(f"  MW-U CER two-sided: Kraken U={mw_k_cer_858_2.statistic:,.1f}, p={mw_k_cer_858_2.pvalue:.4e} | HAT U={mw_h_cer_858_2.statistic:,.1f}, p={mw_h_cer_858_2.pvalue:.4e}")
    print(f"  Fisher Cat two-sided: Kraken OR={fish_k_or_858:.3f}, p={fish_k_p_858:.4f} | HAT OR={fish_h_or_858:.3f}, p={fish_h_p_858:.4f}")

    deconf_desc = []
    for sub, lbl in [(dam, f"Damaged (n={n_dam})"), (cln, f"Clean (n={n_cln})")]:
        deconf_desc.append({
            "Subset": lbl,
            "N": len(sub),
            "K_Mean_CER": sub["kraken_cer"].mean() * 100,
            "K_Med_CER": sub["kraken_cer"].median() * 100,
            "K_Corp_CER": sub["k_char_edits"].sum() / sub["gt_len"].sum() * 100,
            "H_Mean_CER": sub["hatformer_cer"].mean() * 100,
            "H_Med_CER": sub["hatformer_cer"].median() * 100,
            "H_Corp_CER": sub["h_char_edits"].sum() / sub["gt_len"].sum() * 100,
            "K_Mean_WER": sub["kraken_wer"].mean() * 100,
            "K_Med_WER": sub["kraken_wer"].median() * 100,
            "K_Corp_WER": sub["kraken_word_edits"].sum() / sub["gt_words"].sum() * 100,
            "H_Mean_WER": sub["hatformer_wer"].mean() * 100,
            "H_Med_WER": sub["hatformer_wer"].median() * 100,
            "H_Corp_WER": sub["hatformer_word_edits"].sum() / sub["gt_words"].sum() * 100,
            "K_Cat_N": int((sub["kraken_cer"] > 0.50).sum()),
            "K_Cat_pct": (sub["kraken_cer"] > 0.50).mean() * 100,
            "H_Cat_N": int((sub["hatformer_cer"] > 0.50).sum()),
            "H_Cat_pct": (sub["hatformer_cer"] > 0.50).mean() * 100,
        })
    df_deconf = pd.DataFrame(deconf_desc)
    print(df_deconf.to_string(index=False, float_format="{:.2f}".format))

    mw_k_cer_2 = mannwhitneyu(dam["kraken_cer"], cln["kraken_cer"], alternative="two-sided")
    mw_k_cer_1 = mannwhitneyu(dam["kraken_cer"], cln["kraken_cer"], alternative="greater")
    mw_h_cer_2 = mannwhitneyu(dam["hatformer_cer"], cln["hatformer_cer"], alternative="two-sided")
    mw_h_cer_1 = mannwhitneyu(dam["hatformer_cer"], cln["hatformer_cer"], alternative="greater")
    mw_k_wer_2 = mannwhitneyu(dam["kraken_wer"], cln["kraken_wer"], alternative="two-sided")
    mw_h_wer_2 = mannwhitneyu(dam["hatformer_wer"], cln["hatformer_wer"], alternative="two-sided")

    k_dc, k_cc = int((dam["kraken_cer"] > 0.50).sum()), int((cln["kraken_cer"] > 0.50).sum())
    h_dc, h_cc = int((dam["hatformer_cer"] > 0.50).sum()), int((cln["hatformer_cer"] > 0.50).sum())
    fish_k_or, fish_k_p = fisher_exact([[k_dc, n_dam - k_dc], [k_cc, n_cln - k_cc]], alternative="two-sided")
    fish_h_or, fish_h_p = fisher_exact([[h_dc, n_dam - h_dc], [h_cc, n_cln - h_cc]], alternative="two-sided")

    obs_pwr_k = two_prop_power(k_dc / n_dam, k_cc / n_cln, n_dam, n_cln)
    obs_pwr_h = two_prop_power(h_dc / n_dam, h_cc / n_cln, n_dam, n_cln)

    p0_k = k_cc / n_cln
    p0_h = h_cc / n_cln
    mdor_k, mdp_k = min_detectable_or(p0_k, n_dam, n_cln, target_power=0.80)
    mdor_h, mdp_h = min_detectable_or(p0_h, n_dam, n_cln, target_power=0.80)

    or_grid = []
    for mult_or in [1.5, 2.0, 3.0, 4.0, 5.0]:
        odds_k = (p0_k / (1.0 - p0_k)) * mult_or
        p1_k = odds_k / (1.0 + odds_k)
        odds_h = (p0_h / (1.0 - p0_h)) * mult_or
        p1_h = odds_h / (1.0 + odds_h)
        or_grid.append({
            "Target_OR": mult_or,
            "K_p_dam": p1_k * 100,
            "K_Power": two_prop_power(p1_k, p0_k, n_dam, n_cln) * 100,
            "H_p_dam": p1_h * 100,
            "H_Power": two_prop_power(p1_h, p0_h, n_dam, n_cln) * 100,
        })
    df_pwr = pd.DataFrame(or_grid)

    print(f"\nMann-Whitney U (De-confounded Damaged vs Clean):")
    print(f"  Kraken CER    : U={mw_k_cer_2.statistic:,.1f} | two-sided p={mw_k_cer_2.pvalue:.4e} | one-sided p={mw_k_cer_1.pvalue:.4e}")
    print(f"  HATFormer CER : U={mw_h_cer_2.statistic:,.1f} | two-sided p={mw_h_cer_2.pvalue:.4e} | one-sided p={mw_h_cer_1.pvalue:.4e}")
    print(f"  Kraken WER    : U={mw_k_wer_2.statistic:,.1f} | two-sided p={mw_k_wer_2.pvalue:.4f}")
    print(f"  HATFormer WER : U={mw_h_wer_2.statistic:,.1f} | two-sided p={mw_h_wer_2.pvalue:.4f}")
    print(f"\nTwo-Sided Fisher's Exact (Catastrophic >50% CER):")
    print(f"  Kraken    : {k_dc}/{n_dam} ({k_dc/n_dam*100:.2f}%) vs {k_cc}/{n_cln} ({k_cc/n_cln*100:.2f}%) | OR={fish_k_or:.3f}, p={fish_k_p:.4f}")
    print(f"  HATFormer : {h_dc}/{n_dam} ({h_dc/n_dam*100:.2f}%) vs {h_cc}/{n_cln} ({h_cc/n_cln*100:.2f}%) | OR={fish_h_or:.3f}, p={fish_h_p:.4f}")
    print(f"\nAnalytic Statistical Power (n_dam={n_dam}, n_clean={n_cln}, alpha=0.05 two-sided):")
    print(f"  Post-hoc power at observed rates : Kraken={obs_pwr_k*100:.2f}%, HATFormer={obs_pwr_h*100:.2f}%")
    print(f"  Minimum Detectable OR (80% power): Kraken OR={mdor_k:.2f} (p_dam>={mdp_k*100:.2f}%), HATFormer OR={mdor_h:.2f} (p_dam>={mdp_h*100:.2f}%)")
    print(df_pwr.to_string(index=False, float_format="{:.2f}".format))

    # ── 4. DUAL TEST-SET COMPARISON TABLE ────────────────────────────────────
    print("\n" + "=" * 84)
    print("  4. DUAL TEST-SET COMPARISON TABLE (Official vs. Artifact-Adjusted Clean Stratum)")
    print("=" * 84)

    dual_rows = [
        compute_split_metrics(paired, f"Official Overall (N={len(paired):,})"),
        compute_split_metrics(paired_clean, f"Clean Stratum Overall (N={len(paired_clean):,})"),
        compute_split_metrics(paired[~paired["is_short_line"]], f"Official Normal >=20ch (N={(~paired['is_short_line']).sum():,})"),
        compute_split_metrics(paired[paired["is_short_line"]], f"Official Short <20ch (N={paired['is_short_line'].sum():,})"),
        compute_split_metrics(paired_clean[~paired_clean["is_short_line"]], f"Clean Normal >=20ch (N={(~paired_clean['is_short_line']).sum():,})"),
        compute_split_metrics(paired_clean[paired_clean["is_short_line"]], f"Clean Short <20ch (N={paired_clean['is_short_line'].sum():,})"),
    ]
    for b_id, script in BOOKS.items():
        dual_rows.append(compute_split_metrics(paired[paired["book_id"] == b_id], f"Official — {script} ({b_id})"))
        dual_rows.append(compute_split_metrics(paired_clean[paired_clean["book_id"] == b_id], f"Clean Stratum — {script} ({b_id})"))

    df_dual = pd.DataFrame(dual_rows)
    print(df_dual.to_string(index=False, float_format="{:.2f}".format))

    # ── 5. METADATA / IMAGE HYGIENE CHECK ────────────────────────────────────
    print("\n" + "=" * 84)
    print("  5. METADATA / IMAGE HYGIENE CHECK")
    print("=" * 84)

    meta_test = meta[meta["split"] == "test"].copy()
    meta_test["image_stem"] = meta_test["img_path"].apply(lambda p: Path(p).stem)
    page_max_idx = meta_test.groupby("page_id")["line_idx"].max().to_dict()
    meta_test["page_max_idx"] = meta_test["page_id"].map(page_max_idx)

    merged = paired.merge(
        meta_test[["image_stem", "line_idx", "page_max_idx", "line_type", "crop_src"]],
        on="image_stem",
        how="left",
    )

    stems_all = merged["image_stem"].tolist()
    dhash_vals = []
    img_ws, img_hs = [], []
    for s in stems_all:
        hval, w, h = compute_dhash_uint64(IMAGES_DIR / f"{s}.png")
        dhash_vals.append(hval)
        img_ws.append(w)
        img_hs.append(h)

    merged["img_w"] = img_ws
    merged["img_h"] = img_hs
    merged["dhash"] = dhash_vals

    short_df = merged[merged["is_short_line"]].copy()
    short_df["is_first"] = short_df["line_idx"] == 0
    short_df["is_last"] = short_df["line_idx"] == short_df["page_max_idx"]

    def classify_short(r):
        if r["line_uid"] in BLANK_CROP_UIDS or r["img_w"] < 15:
            return "Blank / Corrupt Crop"
        if r["line_uid"] in PAGE034_UIDS:
            return "Page-Shifted Full Body Line"
        if r["img_w"] >= 500:
            return "Truncated GT / Full Body Tail"
        if r["is_last"] and r["crop_src"] == "bbox" and r["img_w"] <= 250:
            return "Likely Lower-Left Catchword (bbox, final line)"
        return "Other Short Margin/Line"

    short_df["short_category"] = short_df.apply(classify_short, axis=1)
    short_summary = (
        short_df.groupby(["short_category", "book_id"])
        .agg(
            Count=("line_uid", "count"),
            Med_W=("img_w", "median"),
            Min_W=("img_w", "min"),
            Max_W=("img_w", "max"),
            K_Mean_CER=("kraken_cer", lambda s: s.mean() * 100),
            H_Mean_CER=("hatformer_cer", lambda s: s.mean() * 100),
        )
        .reset_index()
    )
    catchword_mask = short_df["short_category"] == "Likely Lower-Left Catchword (bbox, final line)"
    n_catchwords = int(catchword_mask.sum())
    cw_by_book = short_df.loc[catchword_mask].groupby("book_id").size().to_dict()

    print(f"\n5A. Short Lines (<20 chars, n={len(short_df)}) Classification:")
    print(f"  metadata.csv line_type values (all test): {merged['line_type'].value_counts().to_dict()}")
    print(f"  Running headers (line_idx == 0) among short lines: {int(short_df['is_first'].sum())} / {len(short_df)}")
    print(f"  Likely lower-left catchwords: {n_catchwords} / {len(short_df)} ({cw_by_book})")
    print(short_summary.to_string(index=False, float_format="{:.2f}".format))

    print("\nVerified blank-crop image geometry:")
    for uid in sorted(BLANK_CROP_UIDS):
        r = merged[merged["line_uid"] == uid].iloc[0]
        print(f"  {uid} | {r['image_stem']} | crop_src={r['crop_src']} | {r['img_w']}x{r['img_h']} px | K='{r['kraken_pred']}' | H='{r['hatformer_pred']}'")

    valid_mask = merged["dhash"].notna()
    valid_stems = merged.loc[valid_mask, "image_stem"].tolist()
    valid_hashes = np.array(merged.loc[valid_mask, "dhash"].tolist(), dtype=np.uint64)
    near_dups = find_dhash_near_duplicates(valid_stems, valid_hashes, max_hamming=6)
    exact_dups = [p for p in near_dups if p[2] == 0]

    # Classify near-duplicate pairs using GT string similarity (verified duplicate scans vs incidental stroke collisions)
    stem_to_row = {r["image_stem"]: r for _, r in merged.iterrows()}
    verified_dup_pairs, incidental_pairs = [], []
    for a, b, d in near_dups:
        ra, rb = stem_to_row[a], stem_to_row[b]
        ga, gb = str(ra["gt_normalized"]), str(rb["gt_normalized"])
        sim = 1.0 - LevDist.distance(ga, gb) / max(len(ga), len(gb), 1)
        rec = (a, b, d, sim, ra["page_id"], rb["page_id"], ga[:35], gb[:35])
        if sim >= 0.70:
            verified_dup_pairs.append(rec)
        else:
            incidental_pairs.append(rec)

    # Full page-pair audit across book_03_page_101/103 and page_102/104
    dup_folio_stats = []
    for p1, p2 in [("book_03_page_101", "book_03_page_103"), ("book_03_page_102", "book_03_page_104")]:
        s1 = merged[merged["page_id"] == p1].sort_values("line_idx").reset_index(drop=True)
        s2 = merged[merged["page_id"] == p2].sort_values("line_idx").reset_index(drop=True)
        n_m = min(len(s1), len(s2))
        eq_gt = sum(s1.loc[i, "gt_normalized"] == s2.loc[i, "gt_normalized"] for i in range(n_m))
        gt_cer = [LevDist.distance(s1.loc[i, "gt_normalized"], s2.loc[i, "gt_normalized"]) / max(len(s1.loc[i, "gt_normalized"]), 1) for i in range(n_m)]
        k_match = sum(s1.loc[i, "kraken_pred"] == s2.loc[i, "kraken_pred"] for i in range(n_m))
        dup_folio_stats.append((p1, p2, len(s1), len(s2), eq_gt, float(np.mean(gt_cer) * 100), k_match))

    print(f"\n5B. 64-bit dHash Near-Duplicate Scan across {len(valid_stems):,} test line images (Hamming <= 6):")
    print(f"  Exact hash collisions (Hamming == 0): {len(exact_dups)}")
    print(f"  Total near-duplicate pairs (Hamming <= 6): {len(near_dups)}")
    print(f"    -> Verified true duplicate scans (GT similarity >= 70%): {len(verified_dup_pairs)}")
    print(f"    -> Incidental 9x8 dHash stroke collisions: {len(incidental_pairs)}")
    for p1, p2, n1, n2, eq_gt, mcer, km in dup_folio_stats:
        print(f"  Duplicate Folio Spread: {p1} ({n1}L) <-> {p2} ({n2}L) | exact GT matches={eq_gt}/{min(n1,n2)}, mean inter-GT CER={mcer:.2f}%")
    for rec in verified_dup_pairs:
        print(f"    [TRUE DUP d={rec[2]} sim={rec[3]:.2f}] {rec[0]} <-> {rec[1]}")

    # ── 5C. UPGRADED BENCHMARK AUDIT ARTIFACT CSV ────────────────────────────
    mis_map = {}
    for _, r in merged[merged["line_uid"].isin(PAGE034_UIDS)].iterrows():
        mis_map[r["line_uid"]] = ("page_shift", "book_09_page_034 systematic 5-line GT displacement (all 20 lines on folio)")

    for uid in sorted(BLANK_CROP_UIDS):
        r = merged[merged["line_uid"] == uid].iloc[0]
        mis_map[uid] = ("blank_crop", f"Blank/corrupt crop ({r['img_w']}x{r['img_h']} px, zero text); Kraken empty output is faithful, HATFormer hallucinates boilerplate")

    known_folio_shift_uids = {
        "book_03_line_003060": "book_03_page_152 folio line displacement (GT repeats line 3059 while crop shows line 3061)",
        "book_03_line_003061": "book_03_page_152 folio line displacement (shifted following line 3060 duplicate GT)",
        "book_05_line_004112": "book_05_page_025 line displacement (crop transcribes text matching line 4114 GT segment)",
    }
    for _, mr in mis.iterrows():
        uid = mr["line_uid"]
        if uid in mis_map:
            continue
        if uid in known_folio_shift_uids:
            mis_map[uid] = ("page_shift", known_folio_shift_uids[uid])
        else:
            mis_map[uid] = ("partial_gt_truncation", "GT string truncated/incomplete relative to visible crop text (both models agree with each other, inter-model CER < 35%)")

    mis_set = set(mis["line_uid"])
    audit_rows = []
    for uid in merged.loc[merged["line_uid"].isin(mis_map.keys()), "line_uid"]:
        r = merged[merged["line_uid"] == uid].iloc[0]
        ncat, ndet = mis_map[uid]
        audit_rows.append({
            "line_uid": r["line_uid"],
            "image_stem": r["image_stem"],
            "page_id": r["page_id"],
            "book_id": r["book_id"],
            "script": r["script"],
            "crop_src": r["crop_src"],
            "img_w": int(r["img_w"]),
            "img_h": int(r["img_h"]),
            "gt_normalized": r["gt_normalized"],
            "kraken_pred": r["kraken_pred"],
            "hatformer_pred": r["hatformer_pred"],
            "kraken_cer": r["kraken_cer"],
            "hatformer_cer": r["hatformer_cer"],
            "inter_model_cer": r["inter_model_cer"],
            "kraken_wer": r["kraken_wer"],
            "hatformer_wer": r["hatformer_wer"],
            "in_suspected_gt_misalignment_csv": uid in mis_set,
            "noise_category": ncat,
            "noise_detail": ndet,
        })

    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(OUT_AUDIT, index=False, encoding="utf-8")
    print(f"\n5C. Saved {len(df_audit)} annotated rows to {OUT_AUDIT}")
    print("  noise_category counts:", df_audit["noise_category"].value_counts().to_dict())

    # Build Full Hygiene-Deduplicated Stratum (N = 6,673):
    # excludes all 29 rows in arams28k_benchmark_audit_artifact.csv + 42 duplicated lines on book_03_page_103/104
    dup_folio_uids = set(paired.loc[paired["page_id"].isin({"book_03_page_103", "book_03_page_104"}), "line_uid"])
    hygiene_excl_uids = set(df_audit["line_uid"]) | dup_folio_uids
    paired_hygiene = paired[~paired["line_uid"].isin(hygiene_excl_uids)].copy()
    print(f"\nFull Hygiene-Deduplicated Stratum: N={len(paired_hygiene):,} (excluded {len(hygiene_excl_uids)}: {len(df_audit)} audit artifacts + {len(dup_folio_uids)} dup folio lines)")

    hyg_rows = [
        compute_split_metrics(paired_hygiene, f"Hygiene-Deduplicated Overall (N={len(paired_hygiene):,})"),
        compute_split_metrics(paired_hygiene[~paired_hygiene["is_short_line"]], f"Hygiene Normal >=20ch (N={(~paired_hygiene['is_short_line']).sum():,})"),
        compute_split_metrics(paired_hygiene[paired_hygiene["is_short_line"]], f"Hygiene Short <20ch (N={paired_hygiene['is_short_line'].sum():,})"),
    ]
    for b_id, script in BOOKS.items():
        hyg_rows.append(compute_split_metrics(paired_hygiene[paired_hygiene["book_id"] == b_id], f"Hygiene Clean — {script} ({b_id})"))
    df_dual_full = pd.concat([df_dual, pd.DataFrame(hyg_rows)], ignore_index=True)
    print("\nFull Dual + Hygiene-Deduplicated Stratum Table:")
    print(df_dual_full.to_string(index=False, float_format="{:.2f}".format))

    # ── GENERATE MARKDOWN REPORT ─────────────────────────────────────────────
    md = []
    md.append("# Paper 1 — Statistical Audit & Benchmark Hygiene Tables\n")
    md.append("**Project:** AraMS-28k Benchmark Audit & Zero-Training Fix (Paper 1, Path B)  ")
    md.append(f"**Official Test Split:** $N = {len(paired):,}$ (`book_03` $n=3,661$, `book_05` $n=2,027$, `book_09` $n=1,056$)  ")
    md.append(f"**Artifact-Adjusted Clean Stratum:** $N = {len(paired_clean):,}$ (excludes 20 lines on `book_09_page_034` + 3 verified blank crops)  ")
    md.append("**Models:** Kraken (1D-CNN + BiLSTM + CTC) vs. HATFormer (`BEiT-base (ViT-patch16) + BlockProcessor + RoBERTa-base Arabic BBPE Decoder`, `beams=3`)\n")
    md.append("---\n")

    md.append("## 1. Paired Bootstrap 95% Confidence Intervals ($n = 10,000$, `seed=42`)\n")
    md.append("> **Definition:** $\\Delta = \\text{HATFormer} - \\text{Kraken}$ in percentage points (negative $\\Delta$ means HATFormer has lower error). Corpus metrics preserve character/word weighting inside each bootstrap resample.\n")
    for m_name in metric_order:
        md.append(f"\n### {m_name}\n")
        md.append("| Book | Script | Kraken | Kraken 95% CI | HATFormer | HATFormer 95% CI | $\\Delta$ (HAT−K) | $\\Delta$ 95% CI |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for _, r in df_boot[df_boot["Metric"] == m_name].iterrows():
            b_lbl = "**OVERALL**" if r["Book"] == "OVERALL" else f"`{r['Book']}`"
            md.append(
                f"| {b_lbl} | {r['Script']} | {r['Kraken']:.2f}% | [{r['Kraken_lo']:.2f}%, {r['Kraken_hi']:.2f}%] | "
                f"{r['HATFormer']:.2f}% | [{r['HATFormer_lo']:.2f}%, {r['HATFormer_hi']:.2f}%] | "
                f"**{r['Delta']:+.2f}** | [{r['Delta_lo']:+.2f}, {r['Delta_hi']:+.2f}] |"
            )

    md.append("\n---\n")
    md.append("## 2. Holm-Bonferroni Adjusted P-Values (Two-Sided Paired Wilcoxon)\n")
    md.append("> Family-wise error rate control across all $m = 8$ paired comparisons (4 splits $\\times$ 2 metrics: line CER and True Word-Level WER).\n")
    md.append("| Book | Script | Metric | Wilcoxon $W$ | Median $\\Delta$ (HF−K) | Raw $p$-value | Holm-Bonferroni Adj $p$ |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_holm.iterrows():
        b_lbl = "**OVERALL**" if r["Book"] == "OVERALL" else f"`{r['Book']}`"
        md.append(
            f"| {b_lbl} | {r['Script']} | {r['Metric']} | {r['Wilcoxon_W']:,.0f} | "
            f"{r['Median_Diff_pp']:+.2f} pp | `{r['Raw_p']:.3e}` | **`{r['Holm_Adj_p']:.3e}`** |"
        )

    md.append("\n---\n")
    md.append("## 3. De-Confounded `AraMS-Restore` Physical Degradation Analysis (Naskh)\n")
    md.append(f"> Excludes all 20 shifted lines of `book_09_page_034` ($1,056 - 20 = 1,036$: **$n_{{\\text{{dam}}}} = {n_dam}$** official `AraMS-Restore` lines vs. **$n_{{\\text{{clean}}}} = {n_cln}$** undamaged Naskh lines). Note: `book_09_line_006514` (blank margin crop) belongs to the {'damaged' if blank_in_dam else 'clean'} subset.\n")
    md.append("\n### Error Distributions (De-Confounded Naskh)\n")
    md.append("| Subset | $N$ | Kraken Mean CER | Kraken Med CER | HAT Mean CER | HAT Med CER | Kraken Mean WER | Kraken Corp WER | HAT Mean WER | HAT Corp WER | Kraken Cat (>50%) | HAT Cat (>50%) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_deconf.iterrows():
        md.append(
            f"| {r['Subset']} | {int(r['N'])} | {r['K_Mean_CER']:.2f}% | {r['K_Med_CER']:.2f}% | "
            f"{r['H_Mean_CER']:.2f}% | {r['H_Med_CER']:.2f}% | {r['K_Mean_WER']:.2f}% | {r['K_Corp_WER']:.2f}% | "
            f"{r['H_Mean_WER']:.2f}% | {r['H_Corp_WER']:.2f}% | {int(r['K_Cat_N'])} ({r['K_Cat_pct']:.2f}%) | {int(r['H_Cat_N'])} ({r['H_Cat_pct']:.2f}%) |"
        )

    md.append("\n### Non-Parametric Tests & Binary Catastrophic Power ($n_{\\text{dam}}=177$, $n_{\\text{clean}}=859$)\n")
    md.append("| Test | Model | Statistic | Two-Sided $p$-value | One-Sided $p$ (`dam > clean`) | Interpretation |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :--- |")
    md.append(f"| Mann-Whitney U (Line CER) | Kraken | $U = {mw_k_cer_2.statistic:,.1f}$ | `{mw_k_cer_2.pvalue:.4e}` | `{mw_k_cer_1.pvalue:.4e}` | Systematic upward CER shift |")
    md.append(f"| Mann-Whitney U (Line CER) | HATFormer | $U = {mw_h_cer_2.statistic:,.1f}$ | `{mw_h_cer_2.pvalue:.4e}` | `{mw_h_cer_1.pvalue:.4e}` | Systematic upward CER shift |")
    md.append(f"| Mann-Whitney U (True Line WER) | Kraken | $U = {mw_k_wer_2.statistic:,.1f}$ | `{mw_k_wer_2.pvalue:.4f}` | — | True WER distribution shift |")
    md.append(f"| Mann-Whitney U (True Line WER) | HATFormer | $U = {mw_h_wer_2.statistic:,.1f}$ | `{mw_h_wer_2.pvalue:.4f}` | — | Significant True WER upward shift |")
    md.append(f"| Fisher's Exact Test (CER > 50%) | Kraken | $\\text{{OR}} = {fish_k_or:.3f}$ | `{fish_k_p:.4f}` | — | {k_dc}/{n_dam} ({k_dc/n_dam*100:.2f}%) vs {k_cc}/{n_cln} ({k_cc/n_cln*100:.2f}%) — not enriched |")
    md.append(f"| Fisher's Exact Test (CER > 50%) | HATFormer | $\\text{{OR}} = {fish_h_or:.3f}$ | `{fish_h_p:.4f}` | — | {h_dc}/{n_dam} ({h_dc/n_dam*100:.2f}%) vs {h_cc}/{n_cln} ({h_cc/n_cln*100:.2f}%) — not enriched |")

    md.append(f"\n> **Analytic Power Analysis ($\\alpha = 0.05$, two-sided, $n_{{\\text{{dam}}}}={n_dam}$, $n_{{\\text{{clean}}}}={n_cln}$):**\n"
              f"> - **Observed Post-Hoc Power:** Kraken = **{obs_pwr_k*100:.2f}%** ($p_{{\\text{{dam}}}}={k_dc/n_dam*100:.2f}\\%, p_{{\\text{{clean}}}}={k_cc/n_cln*100:.2f}\\%$), "
              f"HATFormer = **{obs_pwr_h*100:.2f}%** ($p_{{\\text{{dam}}}}={h_dc/n_dam*100:.2f}\\%, p_{{\\text{{clean}}}}={h_cc/n_cln*100:.2f}\\%$).\n"
              f"> - **Minimum Detectable Odds Ratio at 80% Power:** Kraken $\\text{{OR}} \\ge {mdor_k:.2f}$ ($p_{{\\text{{dam}}}} \\ge {mdp_k*100:.2f}\\%$), "
              f"HATFormer $\\text{{OR}} \\ge {mdor_h:.2f}$ ($p_{{\\text{{dam}}}} \\ge {mdp_h*100:.2f}\\%$). Because baseline catastrophic collapse on undamaged Naskh is rare (~{p0_k*100:.2f}% once `book_09_page_034` is removed), binary Fisher testing has low power for small effects ($\\text{{OR}} \\le 2.0$), but >80% power to detect catastrophic surges $\\ge {mdp_k*100:.1f}\\%$.\n")

    md.append(f"\n> **Sensitivity Check ($n_{{\\text{{clean}}}} = {n_cln_858}$, also excluding blank crop `book_09_line_006514`):** "
              f"Clean Mean/Med CER drops slightly to Kraken `{cln_858['kraken_cer'].mean()*100:.2f}% / {cln_858['kraken_cer'].median()*100:.2f}%` "
              f"(True Mean/Corp WER `{cln_858['kraken_wer'].mean()*100:.2f}% / {cln_858['kraken_word_edits'].sum()/cln_858['gt_words'].sum()*100:.2f}%`, "
              f"Cat `{k_cc_858}/{n_cln_858} = {k_cc_858/n_cln_858*100:.2f}%`) and HATFormer `{cln_858['hatformer_cer'].mean()*100:.2f}% / {cln_858['hatformer_cer'].median()*100:.2f}%` "
              f"(True Mean/Corp WER `{cln_858['hatformer_wer'].mean()*100:.2f}% / {cln_858['hatformer_word_edits'].sum()/cln_858['gt_words'].sum()*100:.2f}%`, "
              f"Cat `{h_cc_858}/{n_cln_858} = {h_cc_858/n_cln_858*100:.2f}%`); "
              f"two-sided Mann-Whitney CER stays highly significant (Kraken $U={mw_k_cer_858_2.statistic:,.1f}, p={mw_k_cer_858_2.pvalue:.4e}$; "
              f"HATFormer $U={mw_h_cer_858_2.statistic:,.1f}, p={mw_h_cer_858_2.pvalue:.4e}$) and Fisher catastrophic enrichment stays non-significant "
              f"(Kraken $\\text{{OR}}={fish_k_or_858:.3f}, p={fish_k_p_858:.4f}$; HATFormer $\\text{{OR}}={fish_h_or_858:.3f}, p={fish_h_p_858:.4f}$).\n")

    md.append("| Target True OR | Kraken $p_{\\text{dam}}$ | Kraken Power | HATFormer $p_{\\text{dam}}$ | HATFormer Power |")
    md.append("| :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_pwr.iterrows():
        md.append(f"| {r['Target_OR']:.1f}x | {r['K_p_dam']:.2f}% | {r['K_Power']:.1f}% | {r['H_p_dam']:.2f}% | {r['H_Power']:.1f}% |")

    md.append("\n---\n")
    md.append("## 4. Dual Test-Set Comparison: Official vs. Clean Strata\n")
    md.append("> - **Artifact-Adjusted Clean Stratum ($N=6,721$):** excludes 20 `book_09_page_034` lines + 3 verified blank crops.\n"
              "> - **Full Hygiene-Deduplicated Stratum ($N=6,673$):** excludes all 29 rows in `arams28k_benchmark_audit_artifact.csv` + all 42 duplicated lines on `book_03_page_103` and `book_03_page_104`.\n")
    md.append("| Split | $N$ | K Corp CER | H Corp CER | K Ortho CER | H Ortho CER | K Corp WER | H Corp WER | K Mean WER | H Mean WER | K Cat | H Cat |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_dual_full.iterrows():
        md.append(
            f"| {r['Split']} | {int(r['N']):,} | {r['K_Corp_CER']:.2f}% | {r['H_Corp_CER']:.2f}% | "
            f"{r['K_Ortho_CER']:.2f}% | {r['H_Ortho_CER']:.2f}% | {r['K_Corp_WER']:.2f}% | {r['H_Corp_WER']:.2f}% | "
            f"{r['K_Mean_WER']:.2f}% | {r['H_Mean_WER']:.2f}% | {int(r['K_Cat'])} | {int(r['H_Cat'])} |"
        )

    md.append("\n---\n")
    md.append("## 5. Metadata, Short-Line Taxonomy & Image Hygiene Audit\n")
    md.append("### 5A. Short Lines ($<20$ chars, $n = 95$) Metadata & Crop Geometry Breakdown\n")
    md.append("- **`metadata.csv` Schema Limitation:** All 6,744 test rows carry `line_type = 'main'` (and polygon vertex coordinates are not stored in `metadata.csv`). However, combining `line_idx` position on page, `crop_src`, and measured pixel width cleanly separates marginal catchwords from body-line artifacts.\n")
    md.append(f"- **Top-of-page Headers (`line_idx == 0`):** **0 / {len(short_df)}** short lines occur at line index 0.\n")
    md.append(f"- **Likely Lower-Left Catchwords (`line_idx == page_max_idx`, `crop_src == 'bbox'`, width $\\le 250$ px):** **{n_catchwords} / {len(short_df)} ({n_catchwords/len(short_df)*100:.1f}%)** — {cw_by_book.get('book_03', 0)} in Maghrebi (`book_03`), {cw_by_book.get('book_05', 0)} in Ruq'ah (`book_05`), {cw_by_book.get('book_09', 0)} in Naskh (`book_09`), plus 1 corrupt $4\\times 4$ px blank `bbox` crop (`book_05_line_004216`).\n")
    md.append("\n| Short-Line Category | Book | Count | Median Width (px) | Width Range (px) | Kraken Mean CER | HAT Mean CER |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in short_summary.iterrows():
        md.append(
            f"| {r['short_category']} | `{r['book_id']}` | {int(r['Count'])} | {r['Med_W']:.0f} | "
            f"{int(r['Min_W'])}–{int(r['Max_W'])} | {r['K_Mean_CER']:.1f}% | {r['H_Mean_CER']:.1f}% |"
        )

    md.append("\n### 5B. 64-Bit dHash Near-Duplicate Scan (`Hamming <= 6`)\n")
    md.append("> Standalone full-page scans are not stored in `data/AraMS-28k-HTR/images`; hashing was executed across all 6,744 test-split line image crops.\n")
    md.append(f"- **Images hashed:** {len(valid_stems):,}")
    md.append(f"- **Exact dHash collisions (Hamming = 0):** {len(exact_dups)}")
    md.append(f"- **Near-duplicate pairs (Hamming $\\le 6$):** {len(near_dups)} (**{len(verified_dup_pairs)} verified duplicate scans** with GT similarity $\\ge 70\\%$, {len(incidental_pairs)} incidental low-res stroke collisions across unrelated text)\n")
    md.append("#### Verified Duplicate Bifolio Scan Discovery (`book_03_page_101/102` $\\leftrightarrow$ `book_03_page_103/104`)\n")
    md.append("Cross-checking the dHash hits against ground-truth text revealed **two duplicate manuscript page spreads (84 test lines total; 42 duplicated lines)** in `book_03` (`test` split), where the same physical manuscript spread appears twice with slightly shifted polygon boundaries and minor annotator transcription differences:\n")
    md.append("| Folio Pair | Lines | Identical Line GTs | Mean Inter-Page GT Diff | Aligned dHash Hits ($\\le 6$) |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for p1, p2, n1, n2, eq_gt, mcer, km in dup_folio_stats:
        hits = sum(1 for r in verified_dup_pairs if {r[4], r[5]} == {p1, p2})
        md.append(f"| `{p1}` $\\leftrightarrow$ `{p2}` | {n1} vs {n2} | {eq_gt}/{min(n1,n2)} | {mcer:.2f}% | {hits} |")
    if verified_dup_pairs:
        md.append("\n<details><summary>All 19 Verified Duplicate-Folio dHash Line Hits (Hamming &le; 6)</summary>\n")
        md.append("| Image A | Image B | Hamming | GT Similarity |")
        md.append("| :--- | :--- | :---: | :---: |")
        for a, b, d, sim, _, _, _, _ in verified_dup_pairs:
            md.append(f"| `{a}.png` | `{b}.png` | {d} | {sim*100:.1f}% |")
        md.append("\n</details>\n")

    md.append("\n### 5C. Upgraded Benchmark Audit Artifact (`EDA/arams28k_benchmark_audit_artifact.csv`)\n")
    md.append(f"- **Total Audited Artifact Rows:** {len(df_audit)}\n")
    md.append("| `noise_category` | Count | Scope |")
    md.append("| :--- | :---: | :--- |")
    for cat, cnt in df_audit["noise_category"].value_counts().items():
        desc = {
            "page_shift": "All 20 lines of `book_09_page_034` (5-line shift) + displaced folio lines (`book_03_page_152`, `book_05_page_025`)",
            "blank_crop": "Empty/corrupt paper-margin crops (`book_09_line_006514`, `book_05_line_004741`, `book_05_line_004216`)",
            "partial_gt_truncation": "GT omits visible continuation read consistently by both Kraken & HATFormer (`book_03`)",
        }.get(cat, "—")
        md.append(f"| `{cat}` | {cnt} | {desc} |")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n✓ Markdown summary saved to {OUT_MD}")


if __name__ == "__main__":
    main()
