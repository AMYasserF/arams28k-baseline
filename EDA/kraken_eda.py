# =============================================================================
# Kraken HTR — Comprehensive EDA Script
# Project: Comparative HTR Analysis on AraMS-28k (Kraken vs. HATFormer)
# Books: book_03 (Maghrebi), book_05 (Ruq'ah), book_09 (Naskh)
# Deps: pandas numpy matplotlib seaborn scipy rapidfuzz arabic_reshaper
# =============================================================================

import os, re, sys, warnings, collections
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from rapidfuzz.distance import Levenshtein as LevDist
from rapidfuzz.distance.Levenshtein import editops as lev_editops

# Force UTF-8 output on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import arabic_reshaper
    HAS_RESHAPER = True
except ImportError:
    HAS_RESHAPER = False

warnings.filterwarnings("ignore")

# Use a font that ships with matplotlib for all text
matplotlib.rcParams["font.family"]     = "DejaVu Sans"
matplotlib.rcParams["figure.dpi"]     = 150
matplotlib.rcParams["savefig.dpi"]    = 200
matplotlib.rcParams["axes.unicode_minus"] = False

# ─────────────────────────── PATHS ───────────────────────────────────────────
BASE_DIR = Path("d:/CMP/Research/Baseline_Implementation/arams28k-baseline")
DATA_DIR = BASE_DIR / "EDA" / "kraken_pred"
OUT_DIR  = BASE_DIR / "EDA" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOOKS = {
    "book_03": {"name": "Maghrebi",  "file": DATA_DIR / "kraken_preds_book_03.csv", "color": "#E63946"},
    "book_05": {"name": "Ruq'ah",   "file": DATA_DIR / "kraken_preds_book_05.csv", "color": "#2A9D8F"},
    "book_09": {"name": "Naskh",     "file": DATA_DIR / "kraken_preds_book_09.csv", "color": "#F4A261"},
}
BOOK_NAMES  = [v["name"]  for v in BOOKS.values()]
BOOK_COLORS = [v["color"] for v in BOOKS.values()]

# ─────────────────────────── HELPERS ─────────────────────────────────────────

def save(fig, name):
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {p.name}")
    return p


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0
    return LevDist.distance(ref, hyp) / len(ref)


def wer_line(ref: str, hyp: str) -> float:
    ref_w = ref.split(); hyp_w = hyp.split()
    if not ref_w:
        return 0.0
    return LevDist.distance(" ".join(ref_w), " ".join(hyp_w)) / len(ref.split())


def decompose_edits(ref: str, hyp: str):
    ops  = lev_editops(ref, hyp)
    ins  = sum(1 for op in ops if op.tag == "insert")
    dels = sum(1 for op in ops if op.tag == "delete")
    subs = sum(1 for op in ops if op.tag == "replace")
    return ins, dels, subs


def confusion_pairs(ref: str, hyp: str):
    for op in lev_editops(ref, hyp):
        if op.tag == "replace":
            yield ref[op.src_pos], hyp[op.dest_pos]


def get_ngrams(text: str, n: int):
    t = text.replace(" ", "")
    return [t[i:i+n] for i in range(len(t) - n + 1)]


# ─────────────────────────── LOAD DATA ───────────────────────────────────────
print("\n" + "="*68)
print("  LOADING DATA")
print("="*68)

dfs = {}
for book_id, meta in BOOKS.items():
    df = pd.read_csv(meta["file"])
    df["script"]        = meta["name"]
    df["gt_normalized"] = df["gt_normalized"].fillna("").astype(str)
    df["kraken_pred"]   = df["kraken_pred"].fillna("").astype(str)
    df["cer"]           = df.apply(lambda r: cer(r["gt_normalized"], r["kraken_pred"]), axis=1)
    df["wer"]           = df.apply(lambda r: wer_line(r["gt_normalized"], r["kraken_pred"]), axis=1)
    df["gt_len"]        = df["gt_normalized"].str.len()
    df["pred_len"]      = df["kraken_pred"].str.len()
    df["gt_word_count"] = df["gt_normalized"].str.split().str.len().fillna(0).astype(int)
    df["len_ratio"]     = df["pred_len"] / (df["gt_len"] + 1e-6)

    edits = df.apply(lambda r: decompose_edits(r["gt_normalized"], r["kraken_pred"]), axis=1)
    df[["ins", "dels", "subs"]] = pd.DataFrame(edits.tolist(), index=df.index)
    df["total_edits"] = df["ins"] + df["dels"] + df["subs"]

    dfs[book_id] = df
    print(f"  {book_id} ({meta['name']:12s}): {len(df):>5,} lines | "
          f"CER={df['cer'].mean():.4f} | WER={df['wer'].mean():.4f}")

combined = pd.concat(dfs.values(), ignore_index=True)
print(f"\n  Total lines: {len(combined):,}")


# =============================================================================
# 0 — SUMMARY TABLE
# =============================================================================
print("\n── SECTION 0: SUMMARY TABLE ─────────────────────────────────────────────")

rows = []
for book_id, meta in BOOKS.items():
    df = dfs[book_id]
    total_chars  = df["gt_len"].sum()
    total_edits  = df["total_edits"].sum()
    rows.append({
        "Book":          book_id,
        "Script":        meta["name"],
        "Lines":         len(df),
        "GT Chars":      total_chars,
        "Mean CER":      df["cer"].mean(),
        "Corpus CER":    total_edits / max(total_chars, 1),
        "Median CER":    df["cer"].median(),
        "Mean WER":      df["wer"].mean(),
        "P90 CER":       df["cer"].quantile(0.90),
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False, float_format="{:.4f}".format))

fig, ax = plt.subplots(figsize=(14, 2.8))
ax.axis("off")
display_cols = ["Book", "Script", "Lines", "GT Chars", "Mean CER", "Corpus CER", "Median CER", "Mean WER", "P90 CER"]
tbl_data = summary[display_cols].copy()
for col in ["Mean CER","Corpus CER","Median CER","Mean WER","P90 CER"]:
    tbl_data[col] = tbl_data[col].map("{:.4f}".format)
tbl_data["Lines"]    = tbl_data["Lines"].map("{:,}".format)
tbl_data["GT Chars"] = tbl_data["GT Chars"].map("{:,}".format)
tbl = ax.table(cellText=tbl_data.values, colLabels=tbl_data.columns,
               cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    if r == 0: cell.set_facecolor("#2d3047"); cell.set_text_props(color="white", weight="bold")
    elif r % 2 == 0: cell.set_facecolor("#f0f4f8")
    cell.set_edgecolor("#cccccc")
ax.set_title("Kraken HTR — Dataset & Performance Summary (AraMS-28k)",
             fontsize=12, weight="bold", pad=10)
save(fig, "00_summary_table")


# =============================================================================
# 1 — CER / WER DISTRIBUTIONS
# =============================================================================
print("\n── SECTION 1: CER/WER DISTRIBUTIONS ───────────────────────────────────")

# 1-A violin + boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, metric, label in zip(axes, ["cer","wer"], ["CER","WER"]):
    for i, (book_id, meta) in enumerate(BOOKS.items()):
        vals = dfs[book_id][metric].values
        vp   = ax.violinplot(vals, positions=[i+1], widths=0.6,
                             showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(meta["color"]); body.set_alpha(0.45)
        ax.boxplot(vals, positions=[i+1], widths=0.22,
                   medianprops=dict(color=meta["color"], linewidth=2.5),
                   whiskerprops=dict(color="#555"), capprops=dict(color="#555"),
                   flierprops=dict(marker=".", color=meta["color"], alpha=0.25, markersize=3),
                   boxprops=dict(color="#555"))
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels(BOOK_NAMES, fontsize=12)
    ax.set_ylabel(f"{label} (per line)", fontsize=12)
    ax.set_title(f"{label} Distribution per Script", fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig.suptitle("Kraken — CER & WER per Script (Violin + Box)", fontsize=15, weight="bold")
fig.tight_layout()
save(fig, "01_cer_wer_violin")

# 1-B overlapping histograms + KDE
fig, ax = plt.subplots(figsize=(12, 5))
for book_id, meta in BOOKS.items():
    vals = dfs[book_id]["cer"].clip(0, 1).values
    ax.hist(vals, bins=60, alpha=0.45, color=meta["color"],
            label=meta["name"], density=True, edgecolor="white", linewidth=0.3)
    kde_x = np.linspace(0, 1, 400)
    ax.plot(kde_x, stats.gaussian_kde(vals)(kde_x), color=meta["color"], linewidth=2.2)
ax.set_xlabel("CER", fontsize=12); ax.set_ylabel("Density", fontsize=12)
ax.set_title("CER Distribution by Script — Kraken Predictions", fontsize=14, weight="bold")
ax.legend(fontsize=11); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
save(fig, "02_cer_histogram_kde")

# 1-C cumulative CDF
fig, ax = plt.subplots(figsize=(11, 5))
for book_id, meta in BOOKS.items():
    vals = np.sort(dfs[book_id]["cer"].clip(0, 1).values)
    cdf  = np.arange(1, len(vals)+1) / len(vals)
    ax.plot(vals, cdf, color=meta["color"], linewidth=2.2, label=meta["name"])
for t in [0.05, 0.10, 0.20, 0.30]:
    ax.axvline(t, linestyle="--", color="gray", alpha=0.5, linewidth=0.9)
    ax.text(t+0.003, 0.03, f"{int(t*100)}%", fontsize=8, color="gray")
ax.set_xlabel("CER Threshold"); ax.set_ylabel("Fraction of Lines Below CER")
ax.set_title("Cumulative CER Distribution by Script", fontsize=14, weight="bold")
ax.legend(fontsize=11); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
save(fig, "03_cer_cumulative_cdf")


# =============================================================================
# 2 — EDIT TYPE DECOMPOSITION
# =============================================================================
print("\n── SECTION 2: EDIT DECOMPOSITION ───────────────────────────────────────")

edit_types  = ["ins", "dels", "subs"]
edit_colors = ["#4CAF50", "#f44336", "#FF9800"]
edit_labels = ["Insertions", "Deletions", "Substitutions"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, normalize, title in zip(axes,
        [False, True],
        ["Absolute Edit Counts", "Proportional Edit Mix (%)"]):
    totals = np.array([dfs[b][edit_types].values.sum() for b in BOOKS], dtype=float)
    bottom = np.zeros(3)
    for etype, ecolor, elabel in zip(edit_types, edit_colors, edit_labels):
        vals = np.array([dfs[b][etype].sum() for b in BOOKS], dtype=float)
        if normalize: vals = vals / totals * 100
        ax.bar(BOOK_NAMES, vals, bottom=bottom,
               color=ecolor, label=elabel, edgecolor="white", linewidth=0.5, alpha=0.9)
        bottom += vals
    ax.set_title(title, fontsize=12, weight="bold")
    ax.legend(loc="upper right", fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Count" if not normalize else "%", fontsize=11)
fig.suptitle("Kraken — Edit-Type Decomposition by Script", fontsize=14, weight="bold")
fig.tight_layout(); save(fig, "04_edit_type_stacked")

# Per-line rate box plots
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
for ax, (etype, elabel, ecolor) in zip(axes, zip(edit_types, edit_labels, edit_colors)):
    data = [dfs[b][etype] / (dfs[b]["gt_len"] + 1e-6) for b in BOOKS]
    bp   = ax.boxplot(data, patch_artist=True, notch=True,
                      medianprops=dict(color="white", linewidth=2.5),
                      flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, col in zip(bp["boxes"], BOOK_COLORS):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticklabels(BOOK_NAMES, fontsize=11)
    ax.set_title(elabel, fontsize=12, weight="bold")
    ax.set_ylabel("Rate (edits / GT chars)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Per-Line Edit Rates by Script", fontsize=14, weight="bold")
fig.tight_layout(); save(fig, "05_per_line_edit_rates")


# =============================================================================
# 3 — SEQUENCE LENGTH ANALYSIS
# =============================================================================
print("\n── SECTION 3: SEQUENCE LENGTH vs. ERROR ────────────────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (xcol, xlabel) in zip(axes, [
        ("gt_len",        "GT Line Length (chars)"),
        ("gt_word_count", "GT Word Count")]):
    for book_id, meta in BOOKS.items():
        df = dfs[book_id].copy()
        q_bins = pd.qcut(df[xcol], q=20, duplicates="drop")
        binned = df.groupby(q_bins, observed=True)["cer"].mean()
        mids   = [interval.mid for interval in binned.index]
        ax.plot(mids, binned.values, marker="o", markersize=4,
                linewidth=2, color=meta["color"], label=meta["name"])
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel("Mean CER", fontsize=12)
    ax.set_title(f"CER vs. {xlabel}", fontsize=12, weight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig.suptitle("Sequence Length vs. CER — Kraken (CTC Drift Analysis)",
             fontsize=14, weight="bold")
fig.tight_layout(); save(fig, "06_length_vs_cer")

# Pred vs GT length scatter
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
for ax, book_id in zip(axes, BOOKS):
    df  = dfs[book_id].sample(min(1000, len(dfs[book_id])), random_state=42)
    sc  = ax.scatter(df["gt_len"], df["pred_len"], c=df["cer"],
                     cmap="RdYlGn_r", alpha=0.5, s=10, vmin=0, vmax=0.5)
    lim = max(df["gt_len"].max(), df["pred_len"].max())
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="Perfect")
    ax.set_xlabel("GT Length (chars)", fontsize=10)
    ax.set_ylabel("Pred Length (chars)", fontsize=10)
    ax.set_title(BOOKS[book_id]["name"], fontsize=12, weight="bold",
                 color=BOOKS[book_id]["color"])
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
cbar = fig.colorbar(sc, ax=axes[-1], shrink=0.8)
cbar.set_label("CER", fontsize=10)
fig.suptitle("Predicted vs. GT Length (colored by CER) — Kraken",
             fontsize=14, weight="bold")
fig.tight_layout(); save(fig, "07_pred_gt_length_scatter")


# =============================================================================
# 4 — CHARACTER FREQUENCY & TOP SUBSTITUTIONS
# =============================================================================
print("\n── SECTION 4: CHARACTER ANALYSIS ───────────────────────────────────────")

# 4-A: Top 20 chars per book
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, book_id in zip(axes, BOOKS):
    all_chars = "".join(dfs[book_id]["gt_normalized"].tolist())
    char_cnt  = collections.Counter(c for c in all_chars if c.strip())
    top20     = char_cnt.most_common(20)
    chars, counts = zip(*top20)
    ax.barh(range(20), [c for _, c in top20[::-1]],
            color=BOOKS[book_id]["color"], alpha=0.8)
    ax.set_yticks(range(20))
    ax.set_yticklabels(list(chars[::-1]), fontsize=9)
    ax.set_title(f"{BOOKS[book_id]['name']} — Top 20 GT Chars",
                 fontsize=11, weight="bold")
    ax.set_xlabel("Frequency"); ax.grid(axis="x", alpha=0.3)
fig.suptitle("Most Frequent GT Characters by Script", fontsize=14, weight="bold")
fig.tight_layout(); save(fig, "08_top_gt_chars")

# 4-B: Top substitution pairs
print("  Computing substitution pairs …")
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for ax, book_id in zip(axes, BOOKS):
    df = dfs[book_id]
    conf = collections.Counter()
    for _, row in df.iterrows():
        for rc, hc in confusion_pairs(row["gt_normalized"], row["kraken_pred"]):
            if rc.strip() and hc.strip():
                conf[(rc, hc)] += 1
    top15  = conf.most_common(15)
    if not top15:
        ax.set_visible(False); continue
    labels = [f"[{r}] -> [{h}]" for (r,h), _ in top15]
    vals   = [v for _, v in top15]
    ax.barh(labels[::-1], vals[::-1], color=BOOKS[book_id]["color"], alpha=0.8)
    ax.set_title(f"{BOOKS[book_id]['name']} — Top 15 Substitutions",
                 fontsize=11, weight="bold")
    ax.set_xlabel("Count"); ax.grid(axis="x", alpha=0.3)
fig.suptitle("Top Substitution Pairs (Ref → Hyp) by Script — Kraken",
             fontsize=14, weight="bold")
fig.tight_layout(); save(fig, "09_top_substitution_pairs")


# =============================================================================
# 5 — N-GRAM ERROR RATES
# =============================================================================
print("\n── SECTION 5: N-GRAM ERROR RATES ───────────────────────────────────────")

def ngram_error_rate(df, n):
    total_ref = total_err = 0
    for _, row in df.iterrows():
        ref_ng = set(get_ngrams(row["gt_normalized"], n))
        hyp_ng = set(get_ngrams(row["kraken_pred"], n))
        if not ref_ng: continue
        total_ref += len(ref_ng)
        total_err += len(ref_ng - hyp_ng)
    return total_err / total_ref if total_ref else 0

ngram_res = {}
for book_id, meta in BOOKS.items():
    df = dfs[book_id]
    ngram_res[book_id] = {
        "Unigram ER": df["cer"].mean(),
        "Bigram ER":  ngram_error_rate(df, 2),
        "Trigram ER": ngram_error_rate(df, 3),
    }
    print(f"  {meta['name']:12s} → {ngram_res[book_id]}")

ngram_df = pd.DataFrame(ngram_res).T
ngram_df.index = BOOK_NAMES

fig, ax = plt.subplots(figsize=(10, 6))
x       = np.arange(len(ngram_df))
width   = 0.25
ng_clrs = ["#457B9D", "#1D3557", "#A8DADC"]
for i, (col, clr) in enumerate(zip(ngram_df.columns, ng_clrs)):
    ax.bar(x + (i-1)*width, ngram_df[col].values, width,
           label=col, color=clr, alpha=0.85, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(BOOK_NAMES, fontsize=12)
ax.set_ylabel("Error Rate"); ax.set_title("Character N-gram Error Rates by Script — Kraken",
                                           fontsize=14, weight="bold")
ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
save(fig, "10_ngram_error_rates")


# =============================================================================
# 6 — PERCENTILE OUTLIER CURVES
# =============================================================================
print("\n── SECTION 6: PERCENTILE / OUTLIER CURVES ──────────────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, book_id in zip(axes, BOOKS):
    df  = dfs[book_id]
    srt = df["cer"].sort_values().values
    pct = np.linspace(0, 100, len(srt))
    ax.plot(pct, srt, color=BOOKS[book_id]["color"], linewidth=2)
    for p, col in [(50,"#555"), (90,"orange"), (95,"red")]:
        v = np.percentile(srt, p)
        ax.axhline(v, linestyle="--", color=col, linewidth=1.2,
                   label=f"P{p}: {v:.3f}")
    ax.set_xlabel("Percentile", fontsize=10)
    ax.set_ylabel("CER", fontsize=10)
    ax.set_title(BOOKS[book_id]["name"], fontsize=12, weight="bold",
                 color=BOOKS[book_id]["color"])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig.suptitle("CER Percentile Curves — Identifying Failure Tail",
             fontsize=14, weight="bold")
fig.tight_layout(); save(fig, "11_cer_percentile_curves")


# =============================================================================
# 7 — CER HEATMAP: Length Bucket × Script
# =============================================================================
print("\n── SECTION 7: CER HEATMAP ───────────────────────────────────────────────")

combined["len_bucket"] = pd.cut(
    combined["gt_len"],
    bins=[0,20,40,60,80,100,150,200,500],
    labels=["0-20","21-40","41-60","61-80","81-100","101-150","151-200","200+"])
heatmap_data = (combined
                .groupby(["script","len_bucket"], observed=True)["cer"]
                .mean()
                .unstack("len_bucket"))

fig, ax = plt.subplots(figsize=(13, 4))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white", ax=ax,
            cbar_kws={"label":"Mean CER"})
ax.set_title("Mean CER — Script × Line-Length Bucket — Kraken",
             fontsize=13, weight="bold")
ax.set_xlabel("GT Character Length Bucket", fontsize=11)
ax.set_ylabel("Script", fontsize=11)
fig.tight_layout(); save(fig, "12_cer_heatmap_length_script")


# =============================================================================
# 8 — WER ANALYSIS
# =============================================================================
print("\n── SECTION 8: WER ANALYSIS ──────────────────────────────────────────────")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 8-A mean WER bar + CI
ax = axes[0]
for i, (book_id, meta) in enumerate(BOOKS.items()):
    df  = dfs[book_id]
    mu  = df["wer"].mean()
    sem = df["wer"].std() / np.sqrt(len(df))
    bar = ax.bar(meta["name"], mu, color=meta["color"], alpha=0.85,
                 yerr=sem*1.96, capsize=6, edgecolor="white")
    ax.text(i, mu + sem*1.96 + 0.01, f"{mu:.4f}",
            ha="center", fontsize=11, weight="bold")
ax.set_ylabel("Mean WER"); ax.set_title("Mean WER per Script (95% CI)", weight="bold")
ax.grid(axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

# 8-B WER vs word count
ax = axes[1]
for book_id, meta in BOOKS.items():
    df = dfs[book_id].sample(min(600, len(dfs[book_id])), random_state=0)
    ax.scatter(df["gt_word_count"], df["wer"].clip(0,1),
               color=meta["color"], alpha=0.4, s=8, label=meta["name"])
ax.set_xlabel("GT Word Count"); ax.set_ylabel("WER")
ax.set_title("WER vs. Word Count per Line", weight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

fig.suptitle("Word Error Rate Analysis — Kraken", fontsize=14, weight="bold")
fig.tight_layout(); save(fig, "13_wer_analysis")


# =============================================================================
# 9 — LINE QUALITY TIERS
# =============================================================================
print("\n── SECTION 9: LINE QUALITY TIERS ────────────────────────────────────────")

TIERS = [
    ("Perfect (CER=0)",        lambda d: d["cer"] == 0.0,  "#2ecc71"),
    ("Near-Perfect (CER≤5%)",  lambda d: (d["cer"] > 0) & (d["cer"] <= 0.05), "#a8e6cf"),
    ("Good (5%<CER≤20%)",      lambda d: (d["cer"] > 0.05) & (d["cer"] <= 0.20), "#ffd3b6"),
    ("Poor (20%<CER≤50%)",     lambda d: (d["cer"] > 0.20) & (d["cer"] <= 0.50), "#ff8b94"),
    ("Catastrophic (CER>50%)", lambda d: d["cer"] > 0.50,  "#c0392b"),
]

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(3)
for label, fn, clr in TIERS:
    vals = np.array([fn(dfs[b]).sum() / len(dfs[b]) * 100 for b in BOOKS])
    ax.bar(BOOK_NAMES, vals, bottom=bottom,
           color=clr, label=label, edgecolor="white", linewidth=0.5, alpha=0.9)
    # annotate segments > 3%
    for i, (v, bot) in enumerate(zip(vals, bottom)):
        if v > 3:
            ax.text(i, bot + v/2, f"{v:.1f}%", ha="center", va="center",
                    fontsize=9, weight="bold", color="white")
    bottom += vals
ax.set_ylabel("% of Lines", fontsize=12)
ax.set_title("Line Quality Distribution by Script — Kraken", fontsize=13, weight="bold")
ax.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1.22, 1))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); save(fig, "14_line_quality_tiers")


# =============================================================================
# 10 — CORRELATION HEATMAPS
# =============================================================================
print("\n── SECTION 10: CORRELATION HEATMAPS ────────────────────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
feat_cols = ["cer","wer","gt_len","gt_word_count","ins","dels","subs","len_ratio"]
for ax, book_id in zip(axes, BOOKS):
    corr = dfs[book_id][feat_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.5, linecolor="white",
                ax=ax, cbar=(ax is axes[-1]), cbar_kws={"shrink":0.8})
    ax.set_title(BOOKS[book_id]["name"], fontsize=12, weight="bold",
                 color=BOOKS[book_id]["color"])
fig.suptitle("Feature Correlation Heatmaps by Script — Kraken",
             fontsize=14, weight="bold")
fig.tight_layout(); save(fig, "15_correlation_heatmaps")


# =============================================================================
# 11 — CER DECILE ANALYSIS
# =============================================================================
print("\n── SECTION 11: CER DECILE (10 bins) ────────────────────────────────────")

fig, ax = plt.subplots(figsize=(12, 5))
decile_edges = np.linspace(0, 1, 11)
x = np.arange(10)
width = 0.25
for i, (book_id, meta) in enumerate(BOOKS.items()):
    vals = dfs[book_id]["cer"].clip(0,1).values
    counts, _ = np.histogram(vals, bins=decile_edges)
    pcts = counts / counts.sum() * 100
    ax.bar(x + (i-1)*width, pcts, width,
           label=meta["name"], color=meta["color"], alpha=0.85, edgecolor="white")

bin_labels = [f"{int(e*100)}-{int(decile_edges[j+1]*100)}%" 
              for j, e in enumerate(decile_edges[:-1])]
ax.set_xticks(x); ax.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("% of Lines", fontsize=12)
ax.set_title("CER Decile Distribution by Script — Kraken", fontsize=13, weight="bold")
ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); save(fig, "16_cer_decile_distribution")


# =============================================================================
# 12 — COMBINED DASHBOARD
# =============================================================================
print("\n── SECTION 12: COMBINED DASHBOARD ──────────────────────────────────────")

fig = plt.figure(figsize=(22, 15))
gs  = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.38)

# [0,0:2] CER bar
ax0 = fig.add_subplot(gs[0, 0:2])
for book_id, meta in BOOKS.items():
    df = dfs[book_id]
    ax0.bar(meta["name"], df["cer"].mean(), color=meta["color"], alpha=0.85,
            yerr=df["cer"].std()/np.sqrt(len(df))*1.96, capsize=5)
ax0.set_title("Mean CER per Script", weight="bold")
ax0.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax0.grid(axis="y", alpha=0.3)

# [0,2:4] WER bar
ax1 = fig.add_subplot(gs[0, 2:4])
for book_id, meta in BOOKS.items():
    df = dfs[book_id]
    ax1.bar(meta["name"], df["wer"].mean(), color=meta["color"], alpha=0.85,
            yerr=df["wer"].std()/np.sqrt(len(df))*1.96, capsize=5)
ax1.set_title("Mean WER per Script", weight="bold")
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax1.grid(axis="y", alpha=0.3)

# [1,0:2] Edit proportions
ax2 = fig.add_subplot(gs[1, 0:2])
totals = np.array([dfs[b][edit_types].values.sum() for b in BOOKS], dtype=float)
bottom = np.zeros(3)
for etype, ecolor, elabel in zip(edit_types, edit_colors, edit_labels):
    vals = np.array([dfs[b][etype].sum() for b in BOOKS], dtype=float) / totals * 100
    ax2.bar(BOOK_NAMES, vals, bottom=bottom, color=ecolor, label=elabel,
            edgecolor="white", linewidth=0.5, alpha=0.9)
    bottom += vals
ax2.set_title("Edit Type Mix (%)", weight="bold")
ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3); ax2.set_ylabel("%")

# [1,2:4] CDF
ax3 = fig.add_subplot(gs[1, 2:4])
for book_id, meta in BOOKS.items():
    vals = np.sort(dfs[book_id]["cer"].clip(0,1).values)
    cdf  = np.arange(1, len(vals)+1) / len(vals)
    ax3.plot(vals, cdf, color=meta["color"], linewidth=2, label=meta["name"])
ax3.set_title("CER Cumulative Distribution", weight="bold")
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)
ax3.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

# [2,0:4] N-gram heatmap
ax4 = fig.add_subplot(gs[2, 0:4])
sns.heatmap(ngram_df, annot=True, fmt=".4f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white", ax=ax4,
            cbar_kws={"label":"Error Rate", "shrink":0.6})
ax4.set_title("N-gram Error Rates by Script", weight="bold")

fig.suptitle(
    "KRAKEN HTR — Comprehensive EDA Dashboard\n"
    "AraMS-28k: Maghrebi (book_03) | Ruq'ah (book_05) | Naskh (book_09)",
    fontsize=16, weight="bold")
save(fig, "00_DASHBOARD")


# =============================================================================
# FINAL PRINTOUT
# =============================================================================
print("\n" + "="*68)
print(f"  All figures saved → {OUT_DIR}")
print("="*68)
print("\n  === FINAL SUMMARY ===")
for book_id, meta in BOOKS.items():
    df = dfs[book_id]
    tc = df["gt_len"].sum()
    te = df["total_edits"].sum()
    ins_p  = df["ins"].sum()  / (te+1e-9) * 100
    del_p  = df["dels"].sum() / (te+1e-9) * 100
    sub_p  = df["subs"].sum() / (te+1e-9) * 100
    print(f"\n  {meta['name']:12s} ({book_id})")
    print(f"    Lines:        {len(df):,}")
    print(f"    GT chars:     {tc:,}")
    print(f"    Corpus CER:   {te/tc:.4f}")
    print(f"    Mean CER:     {df['cer'].mean():.4f} ± {df['cer'].std():.4f}")
    print(f"    Mean WER:     {df['wer'].mean():.4f} ± {df['wer'].std():.4f}")
    print(f"    Perfect lines:{(df['cer']==0).sum()} ({(df['cer']==0).mean()*100:.1f}%)")
    print(f"    Catastrophic: {(df['cer']>0.5).sum()} ({(df['cer']>0.5).mean()*100:.1f}%)")
    print(f"    Edit mix:     Ins={ins_p:.1f}% | Del={del_p:.1f}% | Sub={sub_p:.1f}%")
    print(f"    Bigram ER:    {ngram_res[book_id]['Bigram ER']:.4f}")
    print(f"    Trigram ER:   {ngram_res[book_id]['Trigram ER']:.4f}")

print("\n  EDA COMPLETE!")
