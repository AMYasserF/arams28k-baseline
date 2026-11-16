"""Pure-Python helpers for the HATFormer trainer (no torch import on purpose,
so everything here can be unit-tested on any machine).

Contents
  - edit distance / CER / exact-match metrics
  - learning-rate schedule factors (paper: linear warm-up over 500 steps)
  - resume-checkpoint discovery (works across Kaggle sessions)
  - training-history container + curve plotting (LR, loss, accuracy, CER)
"""
import csv
import json
import math
import os
import time

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def norm_text(s):
    """Collapse whitespace so stray double-spaces never count as errors."""
    return " ".join(str(s).split())


def edit_distance(a, b):
    """Levenshtein distance, O(len(a)*len(b)) time, O(len(b)) memory."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_error_rate(preds, refs):
    """Corpus-level CER = total edits / total reference characters."""
    errs = chars = 0
    for p, r in zip(preds, refs):
        p, r = norm_text(p), norm_text(r)
        errs += edit_distance(p, r)
        chars += max(1, len(r))
    return errs / max(1, chars)


def exact_match(preds, refs):
    if not refs:
        return 0.0
    return sum(norm_text(p) == norm_text(r) for p, r in zip(preds, refs)) / len(refs)


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #

def lr_factor(step, warmup, kind="inverse_sqrt", max_steps=20000):
    """Multiplier applied to the base LR after `step` optimizer steps.

    Every schedule starts with the paper's *linear warm-up* (0 -> 1 over
    `warmup` steps).  The paper does not state the post-warm-up decay; the repo
    (and the original HATFormer code) use inverse-sqrt, which is the default.
    """
    if warmup > 0 and step < warmup:
        return step / warmup
    if kind == "inverse_sqrt":
        return math.sqrt(warmup / max(step, 1)) if warmup > 0 else 1.0 / math.sqrt(max(step, 1))
    if kind == "linear":
        return max(0.0, (max_steps - step) / max(1, max_steps - warmup))
    if kind == "cosine":
        prog = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    if kind == "constant":
        return 1.0
    raise ValueError(f"unknown lr_scheduler '{kind}'")


# --------------------------------------------------------------------------- #
# Deterministic data order (so a resumed run sees exactly the batches it would have seen)
# --------------------------------------------------------------------------- #

def epoch_batches(n_items, batch_size, seed, epoch, skip_batches=0):
    """Index batches for `epoch`; the shuffle depends only on (seed, epoch)."""
    import random
    order = list(range(n_items))
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    batches = [order[i:i + batch_size] for i in range(0, n_items, batch_size)]
    return batches[skip_batches:]


# --------------------------------------------------------------------------- #
# Resume discovery
# --------------------------------------------------------------------------- #

CKPT_NAME = "ckpt_last.pt"
CKPT_META = "ckpt_last.json"


def _read_step(ckpt_path):
    meta = os.path.join(os.path.dirname(ckpt_path), CKPT_META)
    try:
        with open(meta, encoding="utf-8") as f:
            return int(json.load(f).get("step", 0))
    except Exception:
        return -1  # unknown: still a candidate, but loses to any known step


def find_resume_checkpoint(search_roots, max_depth=7, skip_dirs=("images", "__pycache__", ".git")):
    """Return the path of the most advanced `ckpt_last.pt` under any root.

    On Kaggle the previous session's output shows up under /kaggle/input/...
    (read-only), while a same-session restart leaves it in /kaggle/working.
    Highest recorded step wins; ties go to the earliest root in the list.
    """
    cands = []
    for prio, root in enumerate(search_roots):
        if not root or not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - base_depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            if CKPT_NAME in filenames:
                p = os.path.join(dirpath, CKPT_NAME)
                cands.append((_read_step(p), -prio, p))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][2]


# --------------------------------------------------------------------------- #
# History + plots
# --------------------------------------------------------------------------- #

class History:
    """Flat list of dict records: {'kind': 'train'|'val', 'step': int, ...}."""

    def __init__(self, records=None):
        self.records = list(records or [])

    def add(self, kind, step, **kw):
        rec = {"kind": kind, "step": int(step), "t": round(time.time(), 1)}
        rec.update({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in kw.items()})
        self.records.append(rec)

    def series(self, kind, key):
        xs, ys = [], []
        for r in self.records:
            if r["kind"] == kind and key in r and r[key] is not None:
                xs.append(r["step"])
                ys.append(r[key])
        return xs, ys

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        tmp = os.path.join(out_dir, "history.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.records, f)
        os.replace(tmp, os.path.join(out_dir, "history.json"))
        keys = sorted({k for r in self.records for k in r})
        first = [k for k in ("kind", "step") if k in keys]
        keys = first + [k for k in keys if k not in first]
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.records:
                w.writerow(r)


def _moving_avg(vals, k):
    out, run = [], []
    for v in vals:
        run.append(v)
        if len(run) > k:
            run.pop(0)
        out.append(sum(run) / len(run))
    return out


def plot_training_curves(history, out_dir, title="HATFormer fine-tuning on AraMS-28k"):
    """Write training_curves.png (+ lr_curve.png, loss_curve.png, accuracy_curve.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    H = history
    lr_x, lr_y = H.series("train", "lr")
    tl_x, tl_y = H.series("train", "train_loss")
    ta_x, ta_y = H.series("train", "train_token_acc")
    vl_x, vl_y = H.series("val", "val_loss")
    vc_x, vc_y = H.series("val", "val_cer")
    vca_x, vca_y = H.series("val", "val_char_acc")
    vex_x, vex_y = H.series("val", "val_exact_acc")
    vta_x, vta_y = H.series("val", "val_token_acc")
    sp_x, sp_y = H.series("train", "sec_per_step")

    def panel_lr(ax):
        ax.plot(lr_x, lr_y, color="tab:purple")
        ax.set_title("Learning rate"); ax.set_xlabel("optimizer step"); ax.set_ylabel("lr")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0)); ax.grid(alpha=.3)

    def panel_loss(ax):
        if tl_x:
            ax.plot(tl_x, tl_y, color="tab:blue", alpha=.25, label="train (raw)")
            ax.plot(tl_x, _moving_avg(tl_y, 10), color="tab:blue", label="train (avg of 10)")
        if vl_x:
            ax.plot(vl_x, vl_y, "o-", color="tab:red", label="validation")
        ax.set_title("Loss (cross-entropy)"); ax.set_xlabel("optimizer step"); ax.set_ylabel("loss")
        ax.grid(alpha=.3); ax.legend()

    def panel_acc(ax):
        if vca_x:
            ax.plot(vca_x, [100 * v for v in vca_y], "o-", color="tab:green", label="val char accuracy (1-CER)")
        if vex_x:
            ax.plot(vex_x, [100 * v for v in vex_y], "s-", color="tab:orange", label="val exact-line accuracy")
        if vta_x:
            ax.plot(vta_x, [100 * v for v in vta_y], "^-", color="tab:red", label="val token accuracy")
        if ta_x:
            ax.plot(ta_x, [100 * v for v in _moving_avg(ta_y, 10)], color="tab:blue", label="train token accuracy")
        ax.set_title("Accuracy (%)"); ax.set_xlabel("optimizer step"); ax.set_ylabel("%")
        ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.legend(fontsize=8)

    def panel_cer(ax):
        if vc_x:
            ax.plot(vc_x, [100 * v for v in vc_y], "o-", color="tab:red")
            i = min(range(len(vc_y)), key=lambda j: vc_y[j])
            ax.scatter([vc_x[i]], [100 * vc_y[i]], s=120, facecolors="none", edgecolors="k", zorder=5)
            ax.annotate(f"best {100 * vc_y[i]:.2f}% @ {vc_x[i]}", (vc_x[i], 100 * vc_y[i]),
                        textcoords="offset points", xytext=(8, 10), fontsize=8)
        ax.set_title("Validation CER (%) - greedy"); ax.set_xlabel("optimizer step"); ax.set_ylabel("CER %")
        ax.grid(alpha=.3)

    def panel_speed(ax):
        ax.plot(sp_x, sp_y, color="tab:gray")
        ax.set_title("Seconds / optimizer step"); ax.set_xlabel("optimizer step"); ax.grid(alpha=.3)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, fn in zip(axes.ravel(), (panel_lr, panel_loss, panel_acc, panel_cer, panel_speed)):
        fn(ax)
    axes.ravel()[5].axis("off")
    last_val = next((r for r in reversed(H.records) if r["kind"] == "val"), None)
    if last_val:
        txt = (f"last validation @ step {last_val['step']}\n"
               f"  loss      {last_val.get('val_loss', float('nan')):.4f}\n"
               f"  CER       {100 * last_val.get('val_cer', float('nan')):.2f} %\n"
               f"  char acc  {100 * last_val.get('val_char_acc', float('nan')):.2f} %\n"
               f"  exact acc {100 * last_val.get('val_exact_acc', float('nan')):.2f} %")
        axes.ravel()[5].text(0.02, 0.95, txt, va="top", family="monospace", fontsize=12)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=110)
    plt.close(fig)

    for name, fns, size in (("lr_curve.png", [panel_lr], (7, 4)),
                            ("loss_curve.png", [panel_loss], (8, 4.5)),
                            ("accuracy_curve.png", [panel_acc], (8, 4.5))):
        f, a = plt.subplots(1, 1, figsize=size)
        fns[0](a)
        f.tight_layout()
        f.savefig(os.path.join(out_dir, name), dpi=110)
        plt.close(f)
