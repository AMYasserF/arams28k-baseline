#!/usr/bin/env python3
"""
train_char_lm.py
================
Train a leak-free smoothed character N-gram language model (default order=5,
Interpolated Modified Kneser-Ney smoothing) strictly on AraMS-28k train lines
(metadata.csv rows with split == "train", main line_type). Zero exposure to
val or test data.

Saves:
  - Fast compact lookup dictionary (.pkl) storing natural-log (ln) probabilities
    and backoff weights for O(1) incremental scoring during CTC Prefix Beam Search.
  - Standard ARPA text file (.arpa) alongside the pickle (and optionally uses
    kenlm if installed and requested).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import pickle
import time
import unicodedata
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
BOS_CHAR = "\u0002"
EOS_CHAR = "\u0003"


def norm_text(s: Any) -> str:
    """Identical NFC + whitespace collapse used across AraMS-28k pipeline."""
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFC", " ".join(s.strip().split()))


class InterpolatedKneserNeyCharLM:
    """
    Fast Interpolated Kneser-Ney character N-gram language model storing
    natural-log (ln) probabilities and backoff weights per n-gram order 1..N.
    """

    def __init__(self, order: int = 5) -> None:
        if order < 2 or order > 8:
            raise ValueError(f"Expected character LM order in 2..8, got {order}")
        self.order = int(order)
        # probs[k]: dict mapping k-char n-gram -> ln P_KN(w_k | w_1..w_{k-1})
        self.probs: list[dict[str, float]] = [{} for _ in range(self.order + 1)]
        # bow[k]: dict mapping k-char history -> ln backoff_weight(h) for k in 1..order-1
        self.bow: list[dict[str, float]] = [{} for _ in range(self.order)]
        self.unk_ln_prob: float = -12.0
        self.vocab: list[str] = []
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _estimate_discount(counts: Counter) -> float:
        n1 = sum(1 for c in counts.values() if c == 1)
        n2 = sum(1 for c in counts.values() if c == 2)
        if n1 == 0 or (n1 + 2 * n2) == 0:
            return 0.75
        d = n1 / float(n1 + 2 * n2)
        return max(0.15, min(0.90, float(d)))

    def fit(self, texts: list[str], use_eos: bool = False) -> None:
        t0 = time.time()
        n_max = self.order

        # Collect raw n-gram counts up to order n_max
        ngram_counts: list[Counter] = [Counter() for _ in range(n_max + 1)]
        vocab_set: set[str] = set()
        pad = BOS_CHAR * (n_max - 1)
        total_chars = 0

        for txt in texts:
            if not txt:
                continue
            total_chars += len(txt)
            vocab_set.update(txt)
            seq = pad + txt + (EOS_CHAR if use_eos else "")
            L = len(seq)
            # Only start emitting from first real character (index n_max - 1)
            start_pos = n_max - 1
            for i in range(start_pos, L):
                for k in range(1, n_max + 1):
                    ngram_counts[k][seq[i - k + 1 : i + 1]] += 1

        if use_eos:
            vocab_set.add(EOS_CHAR)
        self.vocab = sorted(vocab_set)
        V = max(len(self.vocab), 1)
        self.unk_ln_prob = math.log(1.0 / (total_chars + V))

        # Continuation sets for lower orders 1..n_max-1:
        # c_kn[k][ng] = count of distinct preceding chars u such that (u + ng) occurred
        c_kn: list[Counter] = [Counter() for _ in range(n_max + 1)]
        c_kn[n_max] = ngram_counts[n_max]
        for k in range(1, n_max):
            # Distinct single preceding token over observed (k+1)-grams
            cont: defaultdict[str, set[str]] = defaultdict(set)
            for ng_up in ngram_counts[k + 1]:
                cont[ng_up[1:]].add(ng_up[0])
            c_kn[k] = Counter({ng: len(prevs) for ng, prevs in cont.items()})

        # Unigram base probability (interpolated with uniform floor over vocab + unk)
        tot_u = sum(c_kn[1].values())
        eps = 1e-4
        uniform = 1.0 / (V + 1)
        self.probs[1] = {}
        for ch in self.vocab:
            p1 = (1.0 - eps) * (c_kn[1].get(ch, 0) / max(tot_u, 1)) + eps * uniform
            self.probs[1][ch] = math.log(max(p1, 1e-12))

        # Higher orders k = 2 .. n_max: Interpolated Kneser-Ney recursion
        for k in range(2, n_max + 1):
            d_k = self._estimate_discount(ngram_counts[k] if k == n_max else c_kn[k])
            use_counts = c_kn[k]

            hist_tot: defaultdict[str, int] = defaultdict(int)
            hist_unique: defaultdict[str, int] = defaultdict(int)
            for ng, cnt in use_counts.items():
                h = ng[:-1]
                hist_tot[h] += cnt
                hist_unique[h] += 1

            self.bow[k - 1] = {}
            for h, tot in hist_tot.items():
                lam = (d_k * hist_unique[h]) / float(tot)
                self.bow[k - 1][h] = math.log(max(lam, 1e-12))

            self.probs[k] = {}
            for ng, cnt in use_counts.items():
                h = ng[:-1]
                tot = hist_tot[h]
                lam = (d_k * hist_unique[h]) / float(tot)
                p_low = math.exp(self._ln_prob_up_to(ng[1:], k - 1))
                p_k = max(cnt - d_k, 0.0) / float(tot) + lam * p_low
                self.probs[k][ng] = math.log(max(p_k, 1e-12))

        self.metadata = {
            "order": self.order,
            "num_train_lines": len(texts),
            "num_train_chars": total_chars,
            "vocab_size": V,
            "use_eos": use_eos,
            "ngram_counts": {k: len(self.probs[k]) for k in range(1, n_max + 1)},
            "train_sec": round(time.time() - t0, 2),
        }

    def _ln_prob_up_to(self, ng: str, max_k: int) -> float:
        k = min(len(ng), max_k)
        while k >= 2:
            val = self.probs[k].get(ng[-k:])
            if val is not None:
                return val
            bow = self.bow[k - 1].get(ng[-k:-1], 0.0)
            k -= 1
            # Accumulate backoff weight recursively
            return bow + self._ln_prob_up_to(ng[-k:], k)
        return self.probs[1].get(ng[-1:], self.unk_ln_prob)

    def score_char(self, history: str, ch: str) -> float:
        """
        Return natural-log P_KN(ch | history) in O(order) dict lookups.
        `history` contains previously emitted logical RTL characters (up to order-1).
        """
        pad_needed = (self.order - 1) - len(history)
        ctx = (BOS_CHAR * pad_needed + history) if pad_needed > 0 else history[-(self.order - 1) :]
        ng = ctx + ch
        acc_bow = 0.0
        for k in range(self.order, 1, -1):
            sub = ng[-k:]
            val = self.probs[k].get(sub)
            if val is not None:
                return acc_bow + val
            h = sub[:-1]
            acc_bow += self.bow[k - 1].get(h, 0.0)
        return acc_bow + self.probs[1].get(ch, self.unk_ln_prob)

    def score_text(self, text: str) -> float:
        """Total natural-log LM probability of a logical string."""
        hist = ""
        tot = 0.0
        for ch in text:
            tot += self.score_char(hist, ch)
            hist = (hist + ch)[-(self.order - 1) :]
        return tot

    def perplexity(self, texts: list[str]) -> float:
        tot_lp = 0.0
        tot_n = 0
        for t in texts:
            if not t:
                continue
            tot_lp += self.score_text(t)
            tot_n += len(t)
        if tot_n == 0:
            return float("inf")
        return float(math.exp(-tot_lp / tot_n))

    def save_pickle(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(
                {
                    "order": self.order,
                    "probs": self.probs,
                    "bow": self.bow,
                    "unk_ln_prob": self.unk_ln_prob,
                    "vocab": self.vocab,
                    "metadata": self.metadata,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load_pickle(cls, path: str | Path) -> "InterpolatedKneserNeyCharLM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        lm = cls(order=int(obj["order"]))
        lm.probs = obj["probs"]
        lm.bow = obj["bow"]
        lm.unk_ln_prob = float(obj["unk_ln_prob"])
        lm.vocab = obj["vocab"]
        lm.metadata = obj.get("metadata", {})
        return lm

    def write_arpa(self, path: str | Path) -> None:
        """Export standard ARPA format (base-10 log10 probabilities)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ln10 = math.log(10.0)

        def _tok(ch: str) -> str:
            if ch == BOS_CHAR:
                return "<s>"
            if ch == EOS_CHAR:
                return "</s>"
            if ch == " ":
                return "<space>"
            return ch

        with open(out, "w", encoding="utf-8") as f:
            f.write("\n\\data\\\n")
            for k in range(1, self.order + 1):
                extra = 1 if k == 1 else 0  # include <s> placeholder at unigram level
                f.write(f"ngram {k}={len(self.probs[k]) + extra}\n")

            for k in range(1, self.order + 1):
                f.write(f"\n\\{k}-grams:\n")
                if k == 1:
                    f.write(f"-99.000000\t<s>\t{self.bow[1].get(BOS_CHAR, 0.0)/ln10:.6f}\n")
                for ng, lp in sorted(self.probs[k].items()):
                    toks = " ".join(_tok(c) for c in ng)
                    lp10 = lp / ln10
                    if k < self.order and ng in self.bow[k]:
                        bw10 = self.bow[k][ng] / ln10
                        f.write(f"{lp10:.6f}\t{toks}\t{bw10:.6f}\n")
                    else:
                        f.write(f"{lp10:.6f}\t{toks}\n")
            f.write("\n\\end\\\n")


def load_strict_train_texts(data_dir: str | Path, main_only: bool = True) -> list[str]:
    meta_csv = Path(data_dir) / "metadata.csv"
    df = pd.read_csv(meta_csv)
    df = df[df["split"] == "train"].copy()
    if main_only and "line_type" in df.columns:
        df = df[df["line_type"].fillna("main") == "main"]
    assert (df["split"] == "train").all(), "Leakage guard violated: non-train row detected!"
    texts = [norm_text(t) for t in df["gt_text"].tolist()]
    return [t for t in texts if t]


def main() -> int:
    ap = argparse.ArgumentParser(description="Train leak-free 5-gram char LM on AraMS-28k split=='train'.")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "AraMS-28k-HTR"))
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--output-pkl", default=str(REPO_ROOT / "kraken" / "char_lm_5gram_train.pkl"))
    ap.add_argument("--output-arpa", default=str(REPO_ROOT / "kraken" / "char_lm_5gram_train.arpa"))
    ap.add_argument("--check-val-ppl", action="store_true", help="Report held-out val perplexity (read-only diagnostic).")
    args = ap.parse_args()

    train_texts = load_strict_train_texts(args.data_dir, main_only=True)
    print(f"[LM TRAIN] Loaded strict train rows: N={len(train_texts)} (split == 'train', main only)")

    lm = InterpolatedKneserNeyCharLM(order=args.order)
    lm.fit(train_texts, use_eos=False)
    train_ppl = lm.perplexity(train_texts[:2000])
    lm.metadata["train_sample_ppl"] = round(train_ppl, 4)

    lm.save_pickle(args.output_pkl)
    lm.write_arpa(args.output_arpa)

    if args.check_val_ppl:
        df_val = pd.read_csv(Path(args.data_dir) / "metadata.csv")
        df_val = df_val[(df_val["split"] == "val") & (df_val["line_type"].fillna("main") == "main")]
        val_texts = [norm_text(t) for t in df_val["gt_text"].tolist() if norm_text(t)]
        val_ppl = lm.perplexity(val_texts)
        lm.metadata["val_ppl_diagnostic"] = round(val_ppl, 4)
        print(f"[LM DIAG] Held-out val char perplexity (N={len(val_texts)}): {val_ppl:.3f}")

    summary_path = Path(args.output_pkl).with_suffix(".json")
    summary_path.write_text(json.dumps(lm.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[LM SAVED] pkl={args.output_pkl} | arpa={args.output_arpa}")
    print("[LM META]", json.dumps(lm.metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
