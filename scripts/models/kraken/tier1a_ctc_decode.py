#!/usr/bin/env python3
"""
tier1a_ctc_decode.py
====================
Tier 1A Track B: Zero-training CTC decoding calibration for Kraken Epoch 51
(kraken/best_0.8957.safetensors) targeting Ruq'ah's 40.0% deletion bias.

Features:
  - Shadow-safe import of the installed `kraken` package (prevents ./kraken/
    weights directory in the repo from shadowing kraken.lib).
  - Supports both Kraken >= 5.0 `load_any` and direct safetensors `kraken_meta`
    construction for Kraken < 5.0 environments (15/15 exact parity verified).
  - Exact `convert("L")` + `ImageInputTransforms(..., (16, 0), valid_norm=True)`
    preprocessing matching Kraken `rpred` without `bidi_reordering` scrambling
    (raw CTC frame log-probs are cached to disk, then reversed cleanly along
    time `lp[::-1]` during prefix beam search so the forward Arabic char 5-gram LM
    scores prefixes in natural logical RTL reading order).
  - CTC Prefix Beam Search (`beam_width=25`) with sweepable:
      blank_penalty in {0.0, 0.2, 0.4, 0.6}
      length_bonus  in {0.0, 0.05, 0.10}
      lm_weight     in {0.0, 0.15, 0.25}
  - Saves per-book prediction CSVs matching EDA/kraken_pred/kraken_preds_{book_id}.csv
    plus JSON/Markdown tables reporting Strict CER, Ortho-Folded CER, True Corpus/Mean
    WER, Ruq'ah raw deletions, and Catastrophic (>50% CER) counts.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys
import time
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

# Prevent repo root ./kraken/ folder from shadowing installed `kraken` library
_ORIG_SYS_PATH = list(sys.path)
for _shadow in ("", ".", str(REPO_ROOT), os.getcwd()):
    while _shadow in sys.path:
        sys.path.remove(_shadow)

try:
    from kraken.lib.dataset import ImageInputTransforms
    from kraken.lib.vgsl import TorchVGSLModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import `kraken.lib`. Run with Anaconda/environment Python that has "
        "`kraken` installed (e.g. D:\\Anaconda\\anaconda3\\python.exe)."
    ) from exc

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_char_lm import InterpolatedKneserNeyCharLM, norm_text  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein as LevDist
except ImportError:  # pragma: no cover
    LevDist = None

CSV_COLS = ["line_uid", "book_id", "gt_normalized", "kraken_pred"]
DEFAULT_BLANK_PENALTIES = [0.0, 0.2, 0.4, 0.6]
DEFAULT_LENGTH_BONUSES = [0.0, 0.05, 0.10]
DEFAULT_LM_WEIGHTS = [0.0, 0.15, 0.25]


# ─────────────────────────────────────────────────────────────────────────────
# Text & Metric Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ortho_fold(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
    )


def _lev_distance(a: Any, b: Any) -> int:
    if LevDist is not None:
        return int(LevDist.distance(a, b))
    n, m = len(a), len(b)
    if n < m:
        return _lev_distance(b, a)
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * m
        for j, cb in enumerate(b, start=1):
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1))
        prev = curr
    return prev[m]


def _char_deletions(ref: str, hyp: str) -> int:
    if LevDist is not None:
        dels = 0
        for op in LevDist.opcodes(ref, hyp):
            if op.tag == "delete":
                dels += op.src_end - op.src_start
        return int(dels)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    i, j, dels = n, m, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            j -= 1
    return dels


def evaluate_predictions(df_preds: pd.DataFrame, pred_col: str = "kraken_pred") -> dict[str, Any]:
    rows = []
    for _, r in df_preds.iterrows():
        ref = norm_text(str(r["gt_normalized"]))
        hyp = norm_text(str(r[pred_col] if pd.notna(r[pred_col]) else ""))
        c_ed = _lev_distance(ref, hyp)
        o_ref = ortho_fold(ref)
        o_hyp = ortho_fold(hyp)
        o_ed = _lev_distance(o_ref, o_hyp)
        ref_w = ref.split()
        hyp_w = hyp.split()
        w_ed = _lev_distance(ref_w, hyp_w)
        c_del = _char_deletions(ref, hyp)
        rows.append({
            "book_id": r["book_id"],
            "gt_len": len(ref),
            "char_edits": c_ed,
            "line_cer": c_ed / max(len(ref), 1),
            "gt_ortho_len": len(o_ref),
            "ortho_edits": o_ed,
            "gt_words": len(ref_w),
            "word_edits": w_ed,
            "line_wer": w_ed / max(len(ref_w), 1),
            "char_dels": c_del,
        })
    m = pd.DataFrame(rows)

    def _agg(sub: pd.DataFrame) -> dict[str, Any]:
        if len(sub) == 0:
            return {
                "n": 0, "strict_corpus_cer_pct": 0.0, "ortho_corpus_cer_pct": 0.0,
                "true_corpus_wer_pct": 0.0, "true_mean_wer_pct": 0.0,
                "catastrophic_count": 0, "catastrophic_rate_pct": 0.0, "raw_deletions": 0,
            }
        cat_n = int((sub["line_cer"] > 0.50).sum())
        return {
            "n": int(len(sub)),
            "strict_corpus_cer_pct": round(100.0 * float(sub["char_edits"].sum()) / max(int(sub["gt_len"].sum()), 1), 4),
            "ortho_corpus_cer_pct": round(100.0 * float(sub["ortho_edits"].sum()) / max(int(sub["gt_ortho_len"].sum()), 1), 4),
            "true_corpus_wer_pct": round(100.0 * float(sub["word_edits"].sum()) / max(int(sub["gt_words"].sum()), 1), 4),
            "true_mean_wer_pct": round(100.0 * float(sub["line_wer"].mean()), 4),
            "catastrophic_count": cat_n,
            "catastrophic_rate_pct": round(100.0 * cat_n / len(sub), 4),
            "raw_deletions": int(sub["char_dels"].sum()),
        }

    out = {"overall": _agg(m), "per_book": {}}
    for bk in sorted(m["book_id"].unique()):
        out["per_book"][bk] = _agg(m[m["book_id"] == bk])
    out["overall"]["ruqah_raw_deletions"] = int(
        out["per_book"].get("book_05", {}).get("raw_deletions", 0)
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kraken Model Loader, Manifest Alignment, and Logit Extraction Cache
# ─────────────────────────────────────────────────────────────────────────────

def load_kraken_vgsl(ckpt_path: str | Path) -> TorchVGSLModel:
    """
    Load Kraken checkpoint (supports both Kraken >= 5.0 `load_any` and
    safetensors `kraken_meta` construction for Kraken < 5.0 environments).
    """
    ckpt = Path(ckpt_path)
    try:
        from kraken.lib.models import load_any
        return load_any(str(ckpt))
    except Exception:
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(ckpt), framework="pt", device="cpu") as f:
            meta_raw = f.metadata().get("kraken_meta")
        if not meta_raw:
            raise RuntimeError(f"Missing kraken_meta inside {ckpt}")
        meta_obj = json.loads(meta_raw)
        _, model_meta = next(iter(meta_obj.items()))
        kwargs = {
            k: v for k, v in model_meta.items()
            if k in ("one_channel_mode", "seg_type", "model_type")
        }
        nn = TorchVGSLModel(vgsl=model_meta["vgsl"], codec=model_meta["codec"], **kwargs)
        raw_sd = load_file(str(ckpt))
        cleaned_sd = {k.split(".nn.", 1)[1]: v for k, v in raw_sd.items() if ".nn." in k}
        nn.nn.load_state_dict(cleaned_sd, strict=True)
        return nn


def build_label_to_char(nn: TorchVGSLModel) -> dict[int, str]:
    l2c: dict[int, str] = {}
    for ch, labels in nn.codec.c2l.items():
        for idx in labels:
            l2c[int(idx)] = str(ch)
    return l2c


def load_kraken_eval_df(data_dir: str | Path, split: str, limit: int = 0) -> pd.DataFrame:
    data_path = Path(data_dir)
    meta = pd.read_csv(data_path / "metadata.csv")
    sub = meta[meta["split"] == split].copy()
    if "line_type" in sub.columns:
        sub = sub[sub["line_type"].fillna("main") == "main"]
    sub = sub.reset_index(drop=True)

    images_dir = data_path / "images"
    resolved_paths, stems = [], []
    for p in sub["img_path"]:
        rel = Path(str(p).replace("\\", "/"))
        cand = data_path / rel
        if not cand.is_file():
            cand = images_dir / rel.name
        if not cand.is_file():
            raise FileNotFoundError(f"Missing image for {p}")
        resolved_paths.append(str(cand))
        stems.append(cand.stem)

    sub["resolved_path"] = resolved_paths
    sub["image_stem"] = stems
    sub["gt_normalized"] = [norm_text(t) for t in sub["gt_text"]]

    manifest_txt = data_path / "test_manifest.txt"
    if split == "test" and manifest_txt.is_file():
        lines = [ln.strip() for ln in manifest_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        stem_to_idx = {Path(ln.replace("\\", "/")).stem: i for i, ln in enumerate(lines)}
        sub["line_uid"] = [
            f"{bk}_line_{stem_to_idx.get(st, i):06d}"
            for i, (bk, st) in enumerate(zip(sub["book_id"], sub["image_stem"]))
        ]
    else:
        tag = "line" if split == "test" else split
        sub["line_uid"] = [f"{bk}_{tag}_{i:06d}" for i, bk in enumerate(sub["book_id"])]

    if limit and limit > 0 and limit < len(sub):
        # Deterministic stratified selection across books so --limit covers Maghrebi, Ruq'ah, Naskh
        books = list(dict.fromkeys(sub["book_id"].tolist()))
        per_book = max(1, limit // len(books))
        selected = []
        for bk in books:
            selected.extend(sub.index[sub["book_id"] == bk].tolist()[:per_book])
        if len(selected) < limit:
            used = set(selected)
            for idx in sub.index.tolist():
                if idx not in used:
                    selected.append(idx)
                    if len(selected) >= limit:
                        break
        sub = sub.loc[sorted(selected[:limit])].reset_index(drop=True)
    return sub


def extract_or_load_logprobs(
    df: pd.DataFrame,
    ckpt_path: str | Path,
    cache_path: str | Path,
    split: str,
    device_str: str = "auto",
) -> tuple[dict[str, np.ndarray], dict[int, str]]:
    """
    Extract frame-level CTC log-probabilities once and cache float16 arrays by
    line_uid to disk. Returns `(logprobs_by_uid, label_to_char)`.
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cached_lps: dict[str, np.ndarray] = {}
    cached_l2c: dict[int, str] | None = None
    if cache_file.is_file():
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        if payload.get("split") == split and payload.get("ckpt_name") == Path(ckpt_path).name:
            cached_lps = payload.get("log_probs", {})
            cached_l2c = payload.get("l2c")

    needed = df[~df["line_uid"].isin(cached_lps)].copy()
    if len(needed) == 0 and cached_l2c is not None:
        print(f"[CACHE HIT] Loaded all {len(df)} {split} CTC log-prob matrices from {cache_file}")
        return {uid: cached_lps[uid] for uid in df["line_uid"]}, cached_l2c

    nn = load_kraken_vgsl(ckpt_path)
    l2c = build_label_to_char(nn)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    nn.nn.to(device).eval()

    batch, ch, h, w = nn.input
    valid_norm = getattr(nn, "seg_type", "bbox") != "baselines"
    tf = ImageInputTransforms(batch, h, w, ch, (16, 0), valid_norm=valid_norm)

    if len(needed) > 0:
        t0 = time.time()
        print(f"[EXTRACT] Computing Kraken CTC log-probs for {len(needed)}/{len(df)} lines on {device}...")
        with torch.no_grad():
            for done_i, (_, row) in enumerate(needed.iterrows(), start=1):
                with Image.open(row["resolved_path"]) as im:
                    gray = im.convert("L")
                ten = tf(gray).unsqueeze(0).to(device)
                logits, _ = nn.nn(ten, None)  # (1, C, 1, T)
                lp = logits[0, :, 0, :].T.log_softmax(dim=-1)  # (T, C) in visual LTR order
                cached_lps[row["line_uid"]] = lp.cpu().half().numpy()
                if done_i % 500 == 0 or done_i == len(needed):
                    print(f"  processed {done_i}/{len(needed)} lines ({time.time() - t0:.1f}s)", flush=True)

        tmp_file = cache_file.with_suffix(".pt.tmp")
        torch.save(
            {
                "split": split,
                "ckpt_name": Path(ckpt_path).name,
                "l2c": l2c,
                "log_probs": cached_lps,
            },
            tmp_file,
        )
        os.replace(tmp_file, cache_file)
        print(f"[CACHE SAVED] {len(cached_lps)} entries -> {cache_file}")

    return {uid: cached_lps[uid] for uid in df["line_uid"]}, l2c


# ─────────────────────────────────────────────────────────────────────────────
# CTC Prefix Beam Search (with Blank Penalty, Length Bonus, Char 5-gram LM)
# ─────────────────────────────────────────────────────────────────────────────

def _logaddexp(a: float, b: float) -> float:
    if a <= -1e17:
        return b
    if b <= -1e17:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def greedy_ctc_decode(lp_raw: np.ndarray, l2c: dict[int, str], reverse_rtl: bool = True) -> str:
    lp = lp_raw[::-1] if reverse_rtl else lp_raw
    ids = lp.argmax(axis=-1).tolist()
    out, prev = [], 0
    for idx in ids:
        if idx != 0 and idx != prev:
            out.append(l2c[int(idx)])
        prev = idx
    return norm_text("".join(out))


def ctc_prefix_beam_search(
    lp_raw: np.ndarray,
    l2c: dict[int, str],
    beam_width: int = 25,
    blank_penalty: float = 0.0,
    length_bonus: float = 0.0,
    lm: InterpolatedKneserNeyCharLM | None = None,
    lm_weight: float = 0.0,
    top_k_per_frame: int = 10,
    reverse_rtl: bool = True,
) -> str:
    """
    CTC Prefix Beam Search over frame log-probabilities `lp_raw` of shape (T, C).

    By default `reverse_rtl=True` reverses frames `lp_raw[::-1]` before decoding so
    that emissions happen in natural logical RTL Arabic reading order without UBA
    reordering distortion and can be scored directly left-to-right by the forward
    character 5-gram LM.
    """
    lp = np.asarray(lp_raw, dtype=np.float32)
    if reverse_rtl:
        lp = lp[::-1]

    T, C = lp.shape
    if T == 0:
        return ""

    neg_inf = -1e18
    use_lm = (lm is not None) and (lm_weight != 0.0)
    lm_order_m1 = (lm.order - 1) if use_lm else 0

    # Pre-select top non-blank candidates per frame cleanly outside inner loop
    nonblank = lp[:, 1:]  # shape (T, C - 1)
    k_eff = min(top_k_per_frame, C - 1)
    top_idx_rel = np.argpartition(nonblank, -k_eff, axis=1)[:, -k_eff:]
    top_vals = np.take_along_axis(nonblank, top_idx_rel, axis=1)

    # Pure acoustic CTC state: prefix tuple[int, ...] -> (p_b, p_nb)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, neg_inf)}
    # Memoized metadata per prefix inside this utterance: prefix -> (text_str, char_len, lm_ln_prob)
    meta: dict[tuple[int, ...], tuple[str, int, float]] = {(): ("", 0, 0.0)}

    for t in range(T):
        lp_b = float(lp[t, 0]) - blank_penalty
        next_b: dict[tuple[int, ...], float] = {}
        next_nb: dict[tuple[int, ...], float] = {}

        # Filter frame candidates: skip non-blank classes > 10.0 nats below penalized blank
        cutoff = lp_b - 10.0
        row_ids = top_idx_rel[t]
        row_vals = top_vals[t]
        cand_list = [(int(cid) + 1, float(val)) for cid, val in zip(row_ids, row_vals) if val >= cutoff]
        # Always include the single best non-blank class if blank penalty is non-zero
        if not cand_list:
            best_pos = int(np.argmax(row_vals))
            if row_vals[best_pos] > lp_b - 14.0:
                cand_list = [(int(row_ids[best_pos]) + 1, float(row_vals[best_pos]))]

        for p, (p_b, p_nb) in beams.items():
            p_tot = _logaddexp(p_b, p_nb)
            # 1. Transition via blank keeps prefix p unchanged
            next_b[p] = _logaddexp(next_b.get(p, neg_inf), p_tot + lp_b)

            last_c = p[-1] if p else -1
            for c, lp_c in cand_list:
                if c == last_c:
                    # Staying non-blank keeps prefix p; coming from blank starts new char c
                    next_nb[p] = _logaddexp(next_nb.get(p, neg_inf), p_nb + lp_c)
                    if p_b > neg_inf:
                        p_new = p + (c,)
                        next_nb[p_new] = _logaddexp(next_nb.get(p_new, neg_inf), p_b + lp_c)
                        if p_new not in meta:
                            prev_txt, prev_len, prev_lm = meta[p]
                            ch_str = l2c[c]
                            new_lm = prev_lm
                            if use_lm:
                                hist = prev_txt[-lm_order_m1:]
                                for sub_ch in ch_str:
                                    new_lm += lm.score_char(hist, sub_ch)
                                    hist = (hist + sub_ch)[-lm_order_m1:]
                            meta[p_new] = (prev_txt + ch_str, prev_len + len(ch_str), new_lm)
                else:
                    p_new = p + (c,)
                    next_nb[p_new] = _logaddexp(next_nb.get(p_new, neg_inf), p_tot + lp_c)
                    if p_new not in meta:
                        prev_txt, prev_len, prev_lm = meta[p]
                        ch_str = l2c[c]
                        new_lm = prev_lm
                        if use_lm:
                            hist = prev_txt[-lm_order_m1:]
                            for sub_ch in ch_str:
                                new_lm += lm.score_char(hist, sub_ch)
                                hist = (hist + sub_ch)[-lm_order_m1:]
                        meta[p_new] = (prev_txt + ch_str, prev_len + len(ch_str), new_lm)

        # Rank all active prefixes by acoustic CTC + LM + length bonus and prune to beam_width
        all_prefixes = set(next_b.keys()) | set(next_nb.keys())
        scored: list[tuple[float, tuple[int, ...], float, float]] = []
        for q in all_prefixes:
            qb = next_b.get(q, neg_inf)
            qnb = next_nb.get(q, neg_inf)
            _, qlen, qlm = meta[q]
            rank_s = _logaddexp(qb, qnb) + lm_weight * qlm + length_bonus * qlen
            scored.append((rank_s, q, qb, qnb))

        scored.sort(key=lambda item: item[0], reverse=True)
        beams = {q: (qb, qnb) for _, q, qb, qnb in scored[:beam_width]}

    best_q = max(
        beams.keys(),
        key=lambda q: _logaddexp(beams[q][0], beams[q][1])
        + lm_weight * meta[q][2]
        + length_bonus * meta[q][1],
    )
    return norm_text(meta[best_q][0])


# ─────────────────────────────────────────────────────────────────────────────
# Sweep Runner & CLI
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_LM: InterpolatedKneserNeyCharLM | None = None
_WORKER_L2C: dict[int, str] = {}


def _init_ctc_worker(lm_path_str: str, l2c: dict[int, str]) -> None:
    global _WORKER_LM, _WORKER_L2C
    _WORKER_L2C = l2c
    _WORKER_LM = InterpolatedKneserNeyCharLM.load_pickle(lm_path_str) if lm_path_str else None


def _decode_worker_item(task: tuple[np.ndarray, int, float, float, float]) -> str:
    lp_raw, beam_width, bp, lb, lmw = task
    return ctc_prefix_beam_search(
        lp_raw=lp_raw,
        l2c=_WORKER_L2C,
        beam_width=beam_width,
        blank_penalty=bp,
        length_bonus=lb,
        lm=_WORKER_LM,
        lm_weight=lmw,
        reverse_rtl=True,
    )


def save_kraken_preds(df: pd.DataFrame, preds: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "line_uid": df["line_uid"],
        "book_id": df["book_id"],
        "gt_normalized": df["gt_normalized"],
        "kraken_pred": preds,
    })
    for bk, sub in out_df.groupby("book_id", sort=True):
        sub[CSV_COLS].reset_index(drop=True).to_csv(
            out_dir / f"kraken_preds_{bk}.csv", index=False, encoding="utf-8"
        )


def write_sweep_reports(results: list[dict[str, Any]], out_root: Path, split: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "ctc_sweep_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Kraken CTC Tier 1A Calibration Sweep (`split={split}`)\n",
        "| Config | Beam | BlankPen | LenBonus | LM_Wt | Strict CER | Ortho CER | Corpus WER | Mean WER | Cat (>50%) | Ruq'ah Del | Time(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        c = r["config"]
        ov = r["overall"]
        lines.append(
            f"| `{c['tag']}` | {c['beam_width']} | {c['blank_penalty']:.2f} | {c['length_bonus']:.2f} | {c['lm_weight']:.2f} | "
            f"{ov['strict_corpus_cer_pct']:.2f}% | {ov['ortho_corpus_cer_pct']:.2f}% | "
            f"{ov['true_corpus_wer_pct']:.2f}% | {ov['true_mean_wer_pct']:.2f}% | "
            f"{ov['catastrophic_count']} ({ov['catastrophic_rate_pct']:.2f}%) | "
            f"{ov['ruqah_raw_deletions']} | {r['elapsed_sec']:.1f} |"
        )
    all_books = sorted({bk for r in results for bk in r.get("per_book", {}).keys()})
    book_labels = {"book_03": "Maghrebi", "book_05": "Ruq'ah", "book_09": "Naskh"}
    book_hdr = " | ".join(
        f"`{bk}` ({book_labels[bk]})" if bk in book_labels else f"`{bk}`"
        for bk in all_books
    )
    lines.append("\n## Per-Book Strict CER (%) / Raw Deletions\n")
    lines.append(f"| Config | {book_hdr} |")
    lines.append("|---|" + "|".join(["---:"] * len(all_books)) + "|")
    for r in results:
        c = r["config"]
        pb = r["per_book"]
        cells = []
        for bk in all_books:
            if bk in pb:
                b = pb[bk]
                cells.append(f"{b['strict_corpus_cer_pct']:.2f}% (Del={b['raw_deletions']})")
            else:
                cells.append("n/a")
        lines.append(f"| `{c['tag']}` | " + " | ".join(cells) + " |")

    md = "\n".join(lines) + "\n"
    (out_root / "ctc_sweep_summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)


def _parse_float_list(csv_vals: str) -> list[float]:
    return [float(x.strip()) for x in csv_vals.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tier 1A Track B: Kraken CTC Prefix Beam Search + Blank/Length/LM Calibration.")
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "kraken" / "best_0.8957.safetensors"))
    ap.add_argument("--lm-path", default=str(REPO_ROOT / "kraken" / "char_lm_5gram_train.pkl"))
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "AraMS-28k-HTR"))
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "kraken" / "ctc_cache"))
    ap.add_argument("--output-dir", default="runs/tier1a_kraken_ctc")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--beam-width", type=int, default=25)
    ap.add_argument("--blank-penalties", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--length-bonuses", default="0.0,0.05,0.10")
    ap.add_argument("--lm-weights", default="0.0,0.15,0.25")
    ap.add_argument("--verify-greedy", action="store_true", help="Also verify exact greedy CTC parity.")
    ap.add_argument("--single-grid", action="store_true", help="Use only first value of each parameter list (quick smoke).")
    ap.add_argument("--workers", type=int, default=0, help="Worker processes for CTC beam search (0=auto).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    df = load_kraken_eval_df(args.data_dir, split=args.split, limit=args.limit)
    ckpt = Path(args.checkpoint)
    cache_file = Path(args.cache_dir) / f"ctc_logprobs_{args.split}_{ckpt.stem}.pt"
    log_probs, l2c = extract_or_load_logprobs(
        df=df, ckpt_path=ckpt, cache_path=cache_file, split=args.split, device_str=args.device
    )

    lm: InterpolatedKneserNeyCharLM | None = None
    if args.lm_path and Path(args.lm_path).is_file():
        lm = InterpolatedKneserNeyCharLM.load_pickle(args.lm_path)
        print(f"[LM LOADED] order={lm.order}, vocab={len(lm.vocab)} from {args.lm_path}")

    uids = df["line_uid"].tolist()
    if args.verify_greedy:
        greedy_preds = [greedy_ctc_decode(log_probs[u], l2c, reverse_rtl=True) for u in uids]
        g_df = pd.DataFrame({
            "line_uid": df["line_uid"], "book_id": df["book_id"],
            "gt_normalized": df["gt_normalized"], "kraken_pred": greedy_preds,
        })
        gm = evaluate_predictions(g_df, pred_col="kraken_pred")
        ov = gm["overall"]
        print(
            f"[GREEDY] Strict CER={ov['strict_corpus_cer_pct']:.2f}% | "
            f"Ortho CER={ov['ortho_corpus_cer_pct']:.2f}% | "
            f"Corpus WER={ov['true_corpus_wer_pct']:.2f}% | Ruq'ah Del={ov['ruqah_raw_deletions']}"
        )
        # If EDA baseline CSV exists and split=='test', compare exact matches
        eda_dir = REPO_ROOT / "EDA" / "kraken_pred"
        if args.split == "test" and eda_dir.is_dir():
            eda_df = pd.concat(
                [pd.read_csv(p, keep_default_na=False) for p in sorted(eda_dir.glob("kraken_preds_book_*.csv"))],
                ignore_index=True,
            )
            merged = g_df.merge(eda_df[["line_uid", "kraken_pred"]], on="line_uid", suffixes=("_new", "_eda"))
            exact = int((merged["kraken_pred_new"] == merged["kraken_pred_eda"]).sum())
            print(f"[GREEDY PARITY vs EDA] exact_matches={exact}/{len(merged)}")

    bps = _parse_float_list(args.blank_penalties)
    lbs = _parse_float_list(args.length_bonuses)
    lmws = _parse_float_list(args.lm_weights)
    if args.single_grid:
        bps, lbs, lmws = bps[:1], lbs[:1], lmws[:1]
    if lm is None:
        lmws = [0.0]

    workers = args.workers if args.workers > 0 else (min(os.cpu_count() or 1, 6) if len(uids) >= 128 else 1)
    lm_init_arg = str(args.lm_path) if (lm is not None and Path(args.lm_path).is_file()) else ""

    out_root = Path(args.output_dir)
    all_results: list[dict[str, Any]] = []
    pool = (
        ProcessPoolExecutor(max_workers=workers, initializer=_init_ctc_worker, initargs=(lm_init_arg, l2c))
        if workers > 1 else None
    )
    try:
        for bp in bps:
            for lb in lbs:
                for lmw in lmws:
                    t0 = time.time()
                    tag = f"bw{args.beam_width}_bp{bp:.2f}_lb{lb:.2f}_lm{lmw:.2f}"
                    if pool is not None:
                        tasks = [(log_probs[u], args.beam_width, bp, lb, lmw) for u in uids]
                        preds = list(pool.map(_decode_worker_item, tasks, chunksize=16))
                    else:
                        preds = [
                            ctc_prefix_beam_search(
                                lp_raw=log_probs[u],
                                l2c=l2c,
                                beam_width=args.beam_width,
                                blank_penalty=bp,
                                length_bonus=lb,
                                lm=lm,
                                lm_weight=lmw,
                                reverse_rtl=True,
                            )
                            for u in uids
                        ]
                    out_df = pd.DataFrame({
                        "line_uid": df["line_uid"],
                        "book_id": df["book_id"],
                        "gt_normalized": df["gt_normalized"],
                        "kraken_pred": preds,
                    })
                    save_kraken_preds(out_df, preds, out_root / tag)
                    m = evaluate_predictions(out_df, pred_col="kraken_pred")
                    m["config"] = {
                        "tag": tag, "beam_width": args.beam_width,
                        "blank_penalty": bp, "length_bonus": lb, "lm_weight": lmw,
                    }
                    m["elapsed_sec"] = round(time.time() - t0, 2)
                    (out_root / tag / "eval_summary.json").write_text(
                        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    all_results.append(m)
                    ov = m["overall"]
                    print(
                        f"[SWEEP] {tag} -> Strict CER={ov['strict_corpus_cer_pct']:.2f}% | "
                        f"Ortho={ov['ortho_corpus_cer_pct']:.2f}% | WER={ov['true_corpus_wer_pct']:.2f}% | "
                        f"RuqahDel={ov['ruqah_raw_deletions']} | Cat={ov['catastrophic_count']} ({m['elapsed_sec']:.1f}s)",
                        flush=True,
                    )
    finally:
        if pool is not None:
            pool.shutdown()

    write_sweep_reports(all_results, out_root, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
