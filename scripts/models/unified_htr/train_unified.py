#!/usr/bin/env python3
"""Resumable two-stage trainer & evaluator for Unified Script-Aware Hybrid CTC/Attention HTR.

Reuses AraMS-28k official manifests (`train=19,739`, `val=1,486`, `test=6,744`),
HATFormer `hat_utils` / `seed` helpers, and Paper 1 metric conventions (`Strict CER`,
`Ortho CER`, `True Corpus WER`, `Catastrophic >50%`, plus `Aux CTC CER`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedTokenizerFast, TrOCRConfig
import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[2]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))


def _load_hat_module(mod_name: str):
    mod_path = REPO_ROOT / "scripts" / "models" / "hatformer" / f"{mod_name}.py"
    spec = importlib.util.spec_from_file_location(f"_hat_{mod_name}", str(mod_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hat_utils = _load_hat_module("hat_utils")
_hat_seed = _load_hat_module("seed")

CKPT_META = _hat_utils.CKPT_META
CKPT_NAME = _hat_utils.CKPT_NAME
History = _hat_utils.History
edit_distance_py = _hat_utils.edit_distance
epoch_batches = _hat_utils.epoch_batches
find_resume_checkpoint = _hat_utils.find_resume_checkpoint
set_seed = _hat_seed.set_seed

from ctc_vocab import CTCCharVocab  # noqa: E402
from data_utils import BOOK_TO_SCRIPT_NAME, norm_text  # noqa: E402
from dataset import UnifiedHTRDataset, unified_collate_fn  # noqa: E402
from model import (  # noqa: E402
    TROCR_BOS_ID,
    TROCR_EOS_ID,
    TROCR_PAD_ID,
    UnifiedHTRModel,
    load_trocr_decoder,
)

try:
    from rapidfuzz.distance import Levenshtein as _LevDist
except ImportError:  # pragma: no cover
    _LevDist = None

STOP = {"flag": False, "reason": ""}
PAPER_TEST_CER = {"book_03": 37.88, "book_05": 13.26, "book_09": 25.37, "overall": 26.74}


def log(*args: Any) -> None:
    print(*args, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics & Environment
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


def _lev(a: Any, b: Any) -> int:
    if _LevDist is not None:
        return int(_LevDist.distance(a, b))
    return int(edit_distance_py(a, b))


def compute_htr_metrics(
    preds: list[str],
    refs: list[str],
    ctc_preds: list[str] | None = None,
    books: list[str] | None = None,
) -> dict[str, Any]:
    """Compute Paper 1 identical Strict CER, Ortho CER, True Corpus/Mean WER, Catastrophic, & CTC CER."""
    rows = []
    for idx, (pred_raw, ref_raw) in enumerate(zip(preds, refs)):
        p = norm_text(pred_raw)
        r = norm_text(ref_raw)
        c_ed = _lev(p, r)
        o_p, o_r = ortho_fold(p), ortho_fold(r)
        o_ed = _lev(o_p, o_r)
        pw, rw = p.split(), r.split()
        w_ed = _lev(rw, pw)
        ctc_p = norm_text(ctc_preds[idx]) if ctc_preds is not None else ""
        ctc_ed = _lev(ctc_p, r) if ctc_preds is not None else 0
        rows.append({
            "book_id": books[idx] if books is not None else "ALL",
            "gt_len": len(r),
            "gt_len_nz": max(1, len(r)),
            "char_edits": c_ed,
            "line_cer": c_ed / max(1, len(r)),
            "ortho_len": max(1, len(o_r)),
            "ortho_edits": o_ed,
            "gt_words": max(1, len(rw)),
            "word_edits": w_ed,
            "line_wer": w_ed / max(1, len(rw)),
            "ctc_edits": ctc_ed,
        })
    df_m = pd.DataFrame(rows)

    def _agg(sub: pd.DataFrame) -> dict[str, Any]:
        n = len(sub)
        if n == 0:
            return {
                "n": 0, "strict_corpus_cer_pct": 0.0, "ortho_corpus_cer_pct": 0.0,
                "true_corpus_wer_pct": 0.0, "true_mean_wer_pct": 0.0,
                "ctc_corpus_cer_pct": 0.0, "catastrophic_count": 0, "catastrophic_rate_pct": 0.0,
                "exact_match": 0.0,
            }
        tot_gt = int(sub["gt_len"].sum())
        tot_gt_nz = int(sub["gt_len_nz"].sum())
        denom_c = max(tot_gt if tot_gt > 0 else tot_gt_nz, 1)
        cat_n = int((sub["line_cer"] > 0.50).sum())
        exact_n = int((sub["char_edits"] == 0).sum())
        return {
            "n": n,
            "strict_corpus_cer_pct": round(100.0 * float(sub["char_edits"].sum()) / denom_c, 4),
            "ortho_corpus_cer_pct": round(100.0 * float(sub["ortho_edits"].sum()) / max(int(sub["ortho_len"].sum()), 1), 4),
            "true_corpus_wer_pct": round(100.0 * float(sub["word_edits"].sum()) / max(int(sub["gt_words"].sum()), 1), 4),
            "true_mean_wer_pct": round(100.0 * float(sub["line_wer"].mean()), 4),
            "ctc_corpus_cer_pct": round(100.0 * float(sub["ctc_edits"].sum()) / denom_c, 4) if ctc_preds is not None else 0.0,
            "catastrophic_count": cat_n,
            "catastrophic_rate_pct": round(100.0 * cat_n / n, 4),
            "exact_match": round(exact_n / n, 4),
        }

    out: dict[str, Any] = {"overall": _agg(df_m), "per_book": {}}
    if books is not None:
        for bk in sorted(df_m["book_id"].unique()):
            agg_b = _agg(df_m[df_m["book_id"] == bk])
            agg_b["script"] = BOOK_TO_SCRIPT_NAME.get(bk, bk)
            out["per_book"][bk] = agg_b
    return out


def pick_device(allow_cpu: bool = False) -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        prop = torch.cuda.get_device_properties(0)
        log(f"[GPU] {prop.name} | {prop.total_memory / 2**30:.1f} GiB | capability {prop.major}.{prop.minor} "
            f"| torch {torch.__version__} | cuda {torch.version.cuda}")
        x = torch.randn(256, 256, device=dev)
        torch.cuda.synchronize()
        _ = (x @ x).sum().item()
        return dev
    if allow_cpu:
        log("[WARN] No CUDA GPU visible — running on CPU (appropriate for local smoke tests only).")
        return torch.device("cpu")
    raise RuntimeError("No CUDA GPU detected. Pass --allow-cpu for local smoke tests or enable Kaggle GPU T4 x2.")


def build_tokenizer(tokenizer_path: str | None = None, tiny_fallback: bool = False) -> PreTrainedTokenizerFast:
    cands = []
    if tokenizer_path:
        cands.append(Path(tokenizer_path))
    cands.extend([
        REPO_ROOT / "scripts" / "models" / "hatformer" / "tokenizer" / "tokenizer.json",
        REPO_ROOT / "checkpoints" / "hatformer" / "original" / "hatformer-muharaf" / "tokenizer.json",
    ])
    for c in cands:
        if c.is_file() and c.stat().st_size > 1000:
            tok = PreTrainedTokenizerFast(tokenizer_file=str(c))
            tok.add_special_tokens({
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "cls_token": "<s>",
                "bos_token": "<s>",
            })
            return tok

    if tiny_fallback:
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
        pre = pre_tokenizers.ByteLevel(add_prefix_space=False)
        for b_idx in range(256):
            ch_tok = pre_tokenizers.ByteLevel.alphabet()[b_idx]
            if ch_tok not in vocab:
                vocab[ch_tok] = len(vocab)
        bpe = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
        bpe.pre_tokenizer = pre
        bpe.post_processor = processors.RobertaProcessing(sep=("</s>", 2), cls=("<s>", 0), add_prefix_space=False)
        tok = PreTrainedTokenizerFast(tokenizer_object=bpe)
        tok.add_special_tokens({
            "bos_token": "<s>", "pad_token": "<pad>", "eos_token": "</s>",
            "unk_token": "<unk>", "mask_token": "<mask>", "cls_token": "<s>",
        })
        log("[WARN] Built in-memory byte-level fallback tokenizer (--tiny-decoder-test mode).")
        return tok

    raise FileNotFoundError(
        "Tokenizer file `tokenizer.json` not found locally. Pass --tokenizer <path_to_tokenizer.json> "
        "or use --tiny-decoder-test for local offline CPU smoke testing."
    )


def build_model_from_cfg(
    cfg: dict[str, Any],
    ctc_vocab: CTCCharVocab,
    tok: PreTrainedTokenizerFast,
    device: torch.device,
    init_path: str = "hatformer-muharaf",
    tiny_decoder_test: bool = False,
) -> UnifiedHTRModel:
    mcfg = cfg["model"]
    common_kw = dict(
        ctc_vocab_size=ctc_vocab.num_classes,
        use_multiscale=bool(mcfg.get("use_multiscale", True)),
        use_film=bool(mcfg.get("use_film", True)),
        use_ctc=bool(mcfg.get("use_ctc", True)),
        use_script_loss=bool(mcfg.get("use_script_loss", True)),
        film_dropout=float(mcfg.get("film_dropout", 0.2)),
        spec_aug_prob=float(mcfg.get("spec_aug_prob", 0.0)),
        spec_aug_num_masks=int(mcfg.get("spec_aug_num_masks", 2)),
        spec_aug_max_ratio=float(mcfg.get("spec_aug_max_ratio", 0.08)),
        enc_feat_dropout=float(mcfg.get("enc_feat_dropout", 0.0)),
        label_smoothing=float(mcfg.get("label_smoothing", 0.0)),
        context_dropout=float(mcfg.get("context_dropout", 0.10)),
        lambda_ctc=float(mcfg.get("lambda_ctc", 0.30)),
        alpha_script=float(mcfg.get("alpha_script", 0.05)),
        pretrained_backbone=(False if tiny_decoder_test else bool(mcfg.get("pretrained_backbone", True))),
    )
    if tiny_decoder_test:
        dec_cfg = TrOCRConfig(
            vocab_size=max(len(tok), 512),
            d_model=64,
            decoder_layers=2,
            decoder_attention_heads=4,
            decoder_ffn_dim=128,
            cross_attention_hidden_size=768,
            max_position_embeddings=128,
            is_decoder=True,
            add_cross_attention=True,
            tie_word_embeddings=True,
            bos_token_id=TROCR_BOS_ID,
            pad_token_id=TROCR_PAD_ID,
            eos_token_id=TROCR_EOS_ID,
            decoder_start_token_id=TROCR_BOS_ID,
            use_cache=False,
        )
        model = UnifiedHTRModel(decoder_config=dec_cfg, **common_kw)
    else:
        decoder = load_trocr_decoder(init_path, torch_dtype=torch.float32)
        model = UnifiedHTRModel(decoder=decoder, **common_kw)

    model = model.float().to(device)
    bad = {str(p.dtype) for p in model.parameters() if p.dtype != torch.float32}
    assert not bad, f"Model contains non-fp32 master params: {bad}"
    n_tot = sum(p.numel() for p in model.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    log(f"[MODEL] UnifiedHTRModel | total={n_tot/1e6:.2f}M (encoder={(n_tot-n_dec)/1e6:.2f}M, decoder={n_dec/1e6:.2f}M) "
        f"| ms={common_kw['use_multiscale']} film={common_kw['use_film']}(drop={common_kw['film_dropout']:.2f}) "
        f"ctc={common_kw['use_ctc']} script={common_kw['use_script_loss']} "
        f"| spec={common_kw['spec_aug_prob']:.2f} enc_drop={common_kw['enc_feat_dropout']:.2f} "
        f"ctx_drop={common_kw['context_dropout']:.2f} ls={common_kw['label_smoothing']:.2f} | tiny={tiny_decoder_test}")
    return model


def save_unified_best(
    out_dir: str | Path,
    model: UnifiedHTRModel,
    tok: PreTrainedTokenizerFast,
    ctc_vocab: CTCCharVocab,
    cfg: dict[str, Any],
    subdir_name: str = "best",
) -> Path:
    dest = Path(out_dir) / subdir_name
    tmp = Path(out_dir) / f"{subdir_name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), tmp / "unified_model.pt")
    ctc_vocab.to_json(tmp / "ctc_vocab.json")
    tok.save_pretrained(str(tmp))
    model_meta = {
        "model_args": {
            "ctc_vocab_size": model.ctc_vocab_size,
            "num_scripts": model.num_scripts,
            "use_multiscale": model.use_multiscale,
            "use_film": model.use_film,
            "use_ctc": model.use_ctc,
            "use_script_loss": model.use_script_loss,
            "film_dropout": model.film_dropout,
            "spec_aug_prob": model.spec_aug_prob,
            "spec_aug_num_masks": model.spec_aug_num_masks,
            "spec_aug_max_ratio": model.spec_aug_max_ratio,
            "enc_feat_dropout": model.enc_feat_dropout,
            "label_smoothing": model.label_smoothing,
            "context_dropout": model.context_dropout,
            "lambda_ctc": model.lambda_ctc,
            "alpha_script": model.alpha_script,
        },
        "image": cfg["image"],
        "data": cfg["data"],
        "decoder_config": model.decoder.config.to_dict(),
    }
    (tmp / "model_config.json").write_text(json.dumps(model_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return dest


def load_unified_from_checkpoint_dir(
    ckpt_dir: str | Path,
    device: torch.device,
    tokenizer_override: str | None = None,
) -> tuple[UnifiedHTRModel, PreTrainedTokenizerFast, CTCCharVocab, dict[str, Any]]:
    cdir = Path(ckpt_dir)
    meta = json.loads((cdir / "model_config.json").read_text(encoding="utf-8"))
    ctc_vocab = CTCCharVocab.from_json(cdir / "ctc_vocab.json")

    tok_cand = Path(tokenizer_override) if tokenizer_override else (cdir / "tokenizer.json")
    if tok_cand.is_file():
        tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_cand))
        tok.add_special_tokens({
            "pad_token": "<pad>", "eos_token": "</s>", "cls_token": "<s>", "bos_token": "<s>"
        })
    else:
        tok = PreTrainedTokenizerFast.from_pretrained(str(cdir))

    dec_dict = dict(meta["decoder_config"])
    dec_dict.update({
        "is_decoder": True,
        "add_cross_attention": True,
        "bos_token_id": TROCR_BOS_ID,
        "pad_token_id": TROCR_PAD_ID,
        "eos_token_id": TROCR_EOS_ID,
        "decoder_start_token_id": TROCR_BOS_ID,
    })
    dec_cfg = TrOCRConfig(**dec_dict)
    margs = dict(meta["model_args"])
    model = UnifiedHTRModel(decoder_config=dec_cfg, pretrained_backbone=False, **margs)
    sd = torch.load(cdir / "unified_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.float().to(device).eval()
    return model, tok, ctc_vocab, meta


# ─────────────────────────────────────────────────────────────────────────────
# Two-Stage Optimizer & Cosine LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_warmup_factor(local_step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    eff_warm = min(max(0, int(warmup_steps)), max(1, total_steps // 2)) if warmup_steps > 0 else 0
    if eff_warm > 0 and local_step < eff_warm:
        return float(local_step + 1) / float(eff_warm)
    denom = max(1, total_steps - eff_warm)
    prog = min(1.0, max(0.0, float(local_step - eff_warm) / float(denom)))
    min_ratio = min(1.0, max(0.0, float(min_lr) / max(float(base_lr), 1e-12)))
    cos_val = 0.5 * (1.0 + math.cos(math.pi * prog))
    return min_ratio + (1.0 - min_ratio) * cos_val


def build_stage_optimizer_and_scheduler(
    model: UnifiedHTRModel,
    tcfg: dict[str, Any],
    stage: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    wd = float(tcfg["weight_decay"])
    betas = tuple(tcfg["betas"])
    eps = float(tcfg["eps"])
    min_lr = float(tcfg.get("min_lr", 1e-6))
    stage1_steps = int(tcfg["stage1_steps"])
    max_steps = int(tcfg["max_steps"])

    if stage == 1:
        model.set_decoder_frozen(True)
        enc_params = [p for n, p in model.named_parameters() if not n.startswith("decoder.") and p.requires_grad]
        s1_lr = float(tcfg["stage1_lr"])
        s1_warm = int(tcfg["stage1_warmup"])
        opt = torch.optim.AdamW([{"params": enc_params, "lr": s1_lr, "name": "encoder"}], lr=s1_lr, betas=betas, eps=eps, weight_decay=wd)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda s: _cosine_warmup_factor(s, s1_warm, max(1, stage1_steps), s1_lr, min_lr)
        )
        return opt, sched

    model.set_decoder_frozen(False)
    enc_params = [p for n, p in model.named_parameters() if not n.startswith("decoder.") and p.requires_grad]
    dec_params = [p for n, p in model.named_parameters() if n.startswith("decoder.") and p.requires_grad]
    enc_lr = float(tcfg["stage2_encoder_lr"])
    dec_lr = float(tcfg["stage2_decoder_lr"])
    s2_warm = int(tcfg["stage2_warmup"])
    s2_total = max(1, max_steps - stage1_steps)

    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": enc_lr, "name": "encoder"},
            {"params": dec_params, "lr": dec_lr, "name": "decoder"},
        ],
        betas=betas,
        eps=eps,
        weight_decay=wd,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=[
            lambda s: _cosine_warmup_factor(s, s2_warm, s2_total, enc_lr, min_lr),
            lambda s: _cosine_warmup_factor(s, s2_warm, s2_total, dec_lr, min_lr),
        ],
    )
    return opt, sched


# ─────────────────────────────────────────────────────────────────────────────
# Plotting & Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def plot_unified_training_curves(history: History, out_dir: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    H = history

    tr_s, tr_tot = H.series("train", "train_loss")
    _, tr_ce = H.series("train", "train_loss_ce")
    _, tr_ctc = H.series("train", "train_loss_ctc")
    lr_s, lr_enc = H.series("train", "lr_encoder")
    _, lr_dec = H.series("train", "lr_decoder")

    v_s, v_cer = H.series("val", "val_cer_pct")
    _, v_ocer = H.series("val", "val_ortho_cer_pct")
    _, v_ctcc = H.series("val", "val_ctc_cer_pct")
    _, v_wer = H.series("val", "val_wer_pct")
    _, v_loss = H.series("val", "val_loss")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    ax = axes[0, 0]
    if tr_s:
        ax.plot(tr_s, tr_tot, label="train total", color="#1f77b4", alpha=0.8)
        ax.plot(tr_s, tr_ce, label="train CE", color="#2ca02c", alpha=0.7)
        ax.plot(tr_s, tr_ctc, label="train CTC", color="#ff7f0e", alpha=0.7)
    if v_s:
        ax.plot(v_s, v_loss, "o-", label="val total", color="#d62728")
    ax.set_title("Losses (Total / CE / CTC)"); ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 1]
    if v_s:
        ax.plot(v_s, v_cer, "o-", label="Strict CER (%)", color="#d62728")
        ax.plot(v_s, v_ocer, "s--", label="Ortho CER (%)", color="#2ca02c")
        ax.plot(v_s, v_ctcc, "^:", label="Aux CTC CER (%)", color="#ff7f0e")
    ax.set_title("Validation CER (%)"); ax.set_xlabel("step"); ax.set_ylabel("%"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 2]
    if v_s:
        ax.plot(v_s, v_wer, "o-", label="True Corpus WER (%)", color="#9467bd")
    ax.set_title("Validation True Corpus WER (%)"); ax.set_xlabel("step"); ax.set_ylabel("%"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    if lr_s:
        ax.plot(lr_s, lr_enc, label="encoder LR", color="#1f77b4")
        if any(x > 0 for x in lr_dec):
            ax.plot(lr_s, lr_dec, label="decoder LR", color="#e377c2")
    ax.set_title("Differential Learning Rates"); ax.set_xlabel("step"); ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 1]
    sp_s, sp_v = H.series("train", "sec_per_step")
    if sp_s:
        ax.plot(sp_s, sp_v, color="#7f7f7f")
    ax.set_title("Seconds / Optimizer Step"); ax.set_xlabel("step"); ax.grid(alpha=0.3)

    axes[1, 2].axis("off")
    last_val = next((r for r in reversed(H.records) if r["kind"] == "val"), None)
    if last_val:
        txt = (
            f"Last Val @ step {last_val['step']} (Stage {last_val.get('stage', '?')})\n"
            f"  val loss     : {last_val.get('val_loss', float('nan')):.4f}\n"
            f"  Strict CER   : {last_val.get('val_cer_pct', float('nan')):.2f}%\n"
            f"  Ortho CER    : {last_val.get('val_ortho_cer_pct', float('nan')):.2f}%\n"
            f"  Aux CTC CER  : {last_val.get('val_ctc_cer_pct', float('nan')):.2f}%\n"
            f"  Corpus WER   : {last_val.get('val_wer_pct', float('nan')):.2f}%"
        )
        axes[1, 2].text(0.02, 0.95, txt, va="top", family="monospace", fontsize=11)

    fig.suptitle("Unified Script-Aware Hybrid CTC/Attention HTR on AraMS-28k", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path / "training_curves.png", dpi=110)
    plt.close(fig)


def _disk_free_gb(path: str | Path) -> float:
    try:
        return shutil.disk_usage(path).free / 2**30
    except Exception:
        return 1e9


def save_training_state(
    out_dir: str | Path,
    model: UnifiedHTRModel,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    st: dict[str, Any],
    history: History,
    cfg: dict[str, Any],
    reason: str,
) -> bool:
    out_p = Path(out_dir)
    ckpt_path = out_p / CKPT_NAME
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict(),
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "step": int(st["step"]),
        "stage": int(st["stage"]),
        "epoch": int(st["epoch"]),
        "batch_in_epoch": int(st["batch_in_epoch"]),
        "best_val_cer": float(st["best_val_cer"]),
        "history": history.records,
        "config": cfg,
        "rng_states": {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "format": 2,
    }
    tmp = ckpt_path.with_suffix(".pt.tmp")
    t0 = time.time()
    try:
        torch.save(payload, tmp)
        os.replace(tmp, ckpt_path)
        (out_p / CKPT_META).write_text(
            json.dumps({
                "step": st["step"], "stage": st["stage"], "best_val_cer": st["best_val_cer"],
                "reason": reason, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }),
            encoding="utf-8",
        )
    except OSError as exc:
        if tmp.exists():
            tmp.unlink()
        log(f"  [WARN] checkpoint save failed ({exc}); continuing.")
        return False
    log(f"  [CKPT] step {st['step']} (Stage {st['stage']}) saved ({reason}) in {time.time() - t0:.1f}s -> {ckpt_path}")
    return True


def load_training_state(
    ckpt_path: str | Path,
    model: UnifiedHTRModel,
    tcfg: dict[str, Any],
    scaler: Any,
    st: dict[str, Any],
    history: History,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    log(f"[RESUME] loading {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"], strict=True)

    ck_step = int(ck["step"])
    ck_stage = int(ck.get("stage", 1 if ck_step < int(tcfg["stage1_steps"]) else 2))
    opt, sched = build_stage_optimizer_and_scheduler(model, tcfg, stage=ck_stage)
    opt.load_state_dict(ck["optimizer_state"])
    sched.load_state_dict(ck["scheduler_state"])

    if scaler.is_enabled() and ck.get("scaler_state"):
        scaler.load_state_dict(ck["scaler_state"])

    st.update(
        step=ck_step,
        stage=ck_stage,
        epoch=int(ck["epoch"]),
        batch_in_epoch=int(ck["batch_in_epoch"]),
        best_val_cer=float(ck["best_val_cer"]),
    )
    history.records = list(ck.get("history", []))
    try:
        r = ck["rng_states"]
        random.setstate(r["py"])
        np.random.set_state(r["np"])
        torch.set_rng_state(r["torch"])
        if r.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(r["cuda"])
    except Exception as exc:
        log(f"  (rng restore skipped: {exc})")

    log(f"[RESUME] continuing from step {st['step']} (Stage {st['stage']}, epoch {st['epoch']}, "
        f"batch {st['batch_in_epoch']}), best Strict CER {100.0 * st['best_val_cer']:.2f}%")
    del ck
    return opt, sched


def write_status(out_dir: str | Path, st: dict[str, Any], cfg: dict[str, Any], finished: bool, reason: str) -> None:
    d = {
        "finished": bool(finished),
        "step": int(st["step"]),
        "stage": int(st["stage"]),
        "max_steps": int(cfg["train"]["max_steps"]),
        "best_cer": None if st["best_val_cer"] == float("inf") else float(st["best_val_cer"]),
        "reason": reason,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (Path(out_dir) / "status.json").write_text(json.dumps(d, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation & Export
# ─────────────────────────────────────────────────────────────────────────────

def _autocast(device: torch.device, use_amp: bool):
    return torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    non_block = device.type == "cuda"
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=non_block),
        "encoder_attention_mask": batch["encoder_attention_mask"].to(device, non_blocking=non_block),
        "enc_lengths": batch["enc_lengths"].to(device, non_blocking=non_block),
        "decoder_labels": batch["decoder_labels"].to(device, non_blocking=non_block),
        "ctc_targets": batch["ctc_targets"].to(device, non_blocking=non_block),
        "ctc_target_lengths": batch["ctc_target_lengths"].to(device, non_blocking=non_block),
        "script_ids": batch["script_ids"].to(device, non_blocking=non_block),
    }


@torch.no_grad()
def evaluate(
    model: UnifiedHTRModel,
    tok: PreTrainedTokenizerFast,
    ctc_vocab: CTCCharVocab,
    ds: UnifiedHTRDataset,
    device: torch.device,
    cfg: dict[str, Any],
    use_amp: bool,
    num_beams: int = 1,
    max_gen_lines: int | None = None,
    loss_too: bool = True,
    want_samples: int = 3,
    return_preds: bool = False,
) -> dict[str, Any]:
    model.eval()
    tcfg = cfg["train"]
    dcfg = cfg["data"]
    bs = int(tcfg.get("eval_batch_size", 32))
    nw = int(tcfg.get("num_workers", 0)) if device.type == "cuda" else 0
    pin = device.type == "cuda"

    res: dict[str, Any] = {}
    if loss_too:
        tot_loss = tot_ce = tot_ctc = 0.0
        tot_n = 0
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, collate_fn=unified_collate_fn)
        for batch in loader:
            mb = _move_batch(batch, device)
            b_sz = mb["pixel_values"].shape[0]
            with _autocast(device, use_amp):
                out = model(**mb)
            tot_loss += float(out["loss"].item()) * b_sz
            tot_ce += float(out["loss_ce"].item()) * b_sz
            tot_ctc += float(out["loss_ctc"].item()) * b_sz
            tot_n += b_sz
        res["val_loss"] = tot_loss / max(1, tot_n)
        res["val_loss_ce"] = tot_ce / max(1, tot_n)
        res["val_loss_ctc"] = tot_ctc / max(1, tot_n)

    n = len(ds)
    idx = (
        list(range(n))
        if (max_gen_lines is None or max_gen_lines >= n)
        else sorted(set(np.linspace(0, n - 1, max_gen_lines).astype(int).tolist()))
    )
    gen_ds = ds if len(idx) == n else Subset(ds, idx)
    gen_loader = DataLoader(gen_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, collate_fn=unified_collate_fn)

    preds, ctc_preds, refs, books, uids = [], [], [], [], []
    max_new = int(dcfg.get("max_text_length", 64))
    for batch in gen_loader:
        mb = _move_batch(batch, device)
        with _autocast(device, use_amp):
            d_txt, c_txt = model.generate(
                pixel_values=mb["pixel_values"],
                encoder_attention_mask=mb["encoder_attention_mask"],
                enc_lengths=mb["enc_lengths"],
                tokenizer=tok,
                ctc_vocab=ctc_vocab,
                num_beams=num_beams,
                max_new_tokens=max_new,
                return_ctc=True,
            )
        preds.extend(d_txt)
        ctc_preds.extend(c_txt)
        refs.extend(batch["gt_normalized"])
        books.extend(batch["book_id"])
        uids.extend(batch["line_uid"])

    metrics = compute_htr_metrics(preds, refs, ctc_preds=ctc_preds, books=books)
    ov = metrics["overall"]
    res.update({
        "val_cer": ov["strict_corpus_cer_pct"] / 100.0,
        "val_cer_pct": ov["strict_corpus_cer_pct"],
        "val_ortho_cer_pct": ov["ortho_corpus_cer_pct"],
        "val_wer_pct": ov["true_corpus_wer_pct"],
        "val_mean_wer_pct": ov["true_mean_wer_pct"],
        "val_ctc_cer_pct": ov["ctc_corpus_cer_pct"],
        "val_catastrophic_count": ov["catastrophic_count"],
        "val_exact_acc": ov["exact_match"],
        "n_gen": ov["n"],
        "per_book": metrics["per_book"],
        "samples": list(zip(preds[:want_samples], ctc_preds[:want_samples], refs[:want_samples])),
    })
    if return_preds:
        res["preds"] = preds
        res["ctc_preds"] = ctc_preds
        res["refs"] = refs
        res["books"] = books
        res["line_uids"] = uids
    model.train()
    return res


def export_unified_predictions(
    res: dict[str, Any],
    save_preds_dir: str | Path,
    split: str,
    num_beams: int,
    ckpt: str,
) -> dict[str, Any]:
    out_dir = Path(save_preds_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.DataFrame({
        "line_uid": res["line_uids"],
        "book_id": res["books"],
        "gt_normalized": [norm_text(r) for r in res["refs"]],
        "unified_pred": [norm_text(p) for p in res["preds"]],
        "ctc_pred": [norm_text(c) for c in res["ctc_preds"]],
    })
    df_all.to_csv(out_dir / "unified_preds_all.csv", index=False, encoding="utf-8-sig")

    for bk, sub in df_all.groupby("book_id", sort=True):
        sub.to_csv(out_dir / f"unified_preds_{bk}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "checkpoint": str(ckpt),
        "split": split,
        "num_beams": int(num_beams),
        "overall": {
            "lines": int(res["n_gen"]),
            "strict_corpus_cer_pct": res["val_cer_pct"],
            "ortho_corpus_cer_pct": res["val_ortho_cer_pct"],
            "true_corpus_wer_pct": res["val_wer_pct"],
            "true_mean_wer_pct": res["val_mean_wer_pct"],
            "ctc_corpus_cer_pct": res["val_ctc_cer_pct"],
            "catastrophic_count": res["val_catastrophic_count"],
        },
        "per_book": res["per_book"],
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[EXPORT] Wrote unified_preds_all.csv + per-book CSVs + eval_summary.json -> {out_dir}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Preflight, Training Loop, & Eval Entry Points
# ─────────────────────────────────────────────────────────────────────────────

def install_signal_handlers() -> None:
    def _h(signum: int, _frm: Any) -> None:
        STOP["flag"], STOP["reason"] = True, f"signal {signum}"
        log(f"\n[SIGNAL] {signum} received — will save checkpoint and exit cleanly at step boundary.")
    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _h)
        except Exception:
            pass


def make_datasets(
    cfg: dict[str, Any],
    tok: PreTrainedTokenizerFast,
    ctc_vocab: CTCCharVocab,
    data_dir: str | Path,
    split_names: list[str],
    limit: int = 0,
) -> dict[str, UnifiedHTRDataset]:
    icfg, dcfg, tcfg = cfg["image"], cfg["data"], cfg["train"]
    out: dict[str, UnifiedHTRDataset] = {}
    for name in split_names:
        split_val = tcfg.get(f"split_{name}", name)
        use_aug = bool(dcfg.get("augment_train", False)) and (split_val == "train")
        ds = UnifiedHTRDataset(
            data_dir=data_dir,
            split=split_val,
            tokenizer=tok,
            ctc_vocab=ctc_vocab,
            height=int(icfg.get("height", 64)),
            min_width=int(icfg.get("min_width", 16)),
            max_width=int(icfg.get("max_width", 1280)),
            rtl_flip=bool(icfg.get("rtl_flip", True)),
            max_text_length=int(dcfg.get("max_text_length", 64)),
            main_only=bool(dcfg.get("main_only", True)),
            exclude_corrupt_train=bool(dcfg.get("exclude_corrupt_train", False)),
            p_blank=float(dcfg.get("p_blank", 0.0)),
            augment=use_aug,
            limit=limit,
            seed=int(tcfg.get("seed", 1337)),
        )
        log(f"[DATA] {name:5s}: {len(ds):6d} lines | books={sorted(set(ds.book_ids))} "
            f"| augment={ds.augment} | exclude_corrupt={dcfg.get('exclude_corrupt_train', False)} "
            f"| p_blank={dcfg.get('p_blank', 0.0)}")
        out[name] = ds
    return out


def preflight(cfg: dict[str, Any], args: argparse.Namespace, device: torch.device) -> None:
    use_amp = (cfg["train"]["precision"] == "amp_fp16") and (device.type == "cuda")
    tok = build_tokenizer(cfg.get("tokenizer_path"), tiny_fallback=args.tiny_decoder_test)
    ctc_vocab = CTCCharVocab()
    model = build_model_from_cfg(
        cfg, ctc_vocab, tok, device, init_path=cfg["_init_path"], tiny_decoder_test=args.tiny_decoder_test
    )
    d = make_datasets(cfg, tok, ctc_vocab, args.data_dir, ["train", "val"], limit=args.limit or 16)
    tr_ds, va_ds = d["train"], d["val"]
    loader = DataLoader(tr_ds, batch_size=min(4, len(tr_ds)), shuffle=False, collate_fn=unified_collate_fn)
    batch = _move_batch(next(iter(loader)), device)
    model.eval()
    with torch.no_grad(), _autocast(device, use_amp):
        out = model(**batch)
    log(f"[PREFLIGHT] Batch pv={tuple(batch['pixel_values'].shape)} | enc={tuple(out['enc_features'].shape)} "
        f"| total_loss={float(out['loss']):.3f} (CE={float(out['loss_ce']):.3f}, CTC={float(out['loss_ctc']):.3f})")
    r = evaluate(model, tok, ctc_vocab, va_ds, device, cfg, use_amp, num_beams=1, max_gen_lines=8, loss_too=False, want_samples=2)
    log(f"[PREFLIGHT] Val check (n={r['n_gen']}): Strict CER={r['val_cer_pct']:.2f}% | CTC CER={r['val_ctc_cer_pct']:.2f}%")
    for p, cp, rf in r["samples"]:
        log(f"  pred: {p}\n  ctc : {cp}\n  ref : {rf}")
    log("[PREFLIGHT OK]")


def train(cfg: dict[str, Any], args: argparse.Namespace, device: torch.device) -> bool:
    tcfg = cfg["train"]
    set_seed(int(tcfg["seed"]))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = (tcfg["precision"] == "amp_fp16") and (device.type == "cuda")

    tok = build_tokenizer(cfg.get("tokenizer_path"), tiny_fallback=args.tiny_decoder_test)
    ctc_vocab = CTCCharVocab()
    model = build_model_from_cfg(
        cfg, ctc_vocab, tok, device, init_path=cfg["_init_path"], tiny_decoder_test=args.tiny_decoder_test
    )
    ds_map = make_datasets(cfg, tok, ctc_vocab, args.data_dir, ["train", "val"], limit=args.limit or 0)
    tr_ds, va_ds = ds_map["train"], ds_map["val"]

    stage1_steps = int(tcfg["stage1_steps"])
    max_steps = int(tcfg["max_steps"])
    init_stage = 1 if stage1_steps > 0 else 2
    opt, sched = build_stage_optimizer_and_scheduler(model, tcfg, stage=init_stage)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    st: dict[str, Any] = {
        "step": 0, "stage": init_stage, "epoch": 0, "batch_in_epoch": 0, "best_val_cer": float("inf")
    }
    history = History()

    if args.resume != "no":
        roots = [str(out_dir), "/kaggle/input"] if args.resume == "auto" else [args.resume]
        skip = ("images", "__pycache__", ".git") + (() if "smoke" in out_dir.name else ("smoke_run",))
        ck = (
            args.resume
            if (args.resume != "auto" and args.resume.endswith(".pt"))
            else find_resume_checkpoint(roots, skip_dirs=skip)
        )
        if ck:
            opt, sched = load_training_state(ck, model, tcfg, scaler, st, history)
            src_best = Path(ck).parent / "best"
            dst_best = out_dir / "best"
            if src_best.is_dir() and not dst_best.is_dir():
                shutil.copytree(src_best, dst_best)
        else:
            log(f"[RESUME] No checkpoint found — starting fresh in Stage {init_stage} (step 0).")

    # If checkpoint stopped exactly on stage1_steps boundary while training was not finished, promote now
    if st["stage"] == 1 and st["step"] >= stage1_steps and st["step"] < max_steps:
        st["stage"] = 2
        opt, sched = build_stage_optimizer_and_scheduler(model, tcfg, stage=2)
        log(f"[STAGE 2] Unfroze TrOCR decoder on resume (enc_lr={tcfg['stage2_encoder_lr']:.1e}, dec_lr={tcfg['stage2_decoder_lr']:.1e})")

    accum, bs = int(tcfg["grad_accum"]), int(tcfg["batch_size"])
    nw = int(tcfg.get("num_workers", 0)) if device.type == "cuda" else 0
    pin = device.type == "cuda"
    session_start = float(os.environ.get("SESSION_START_TS", time.time()))
    deadline = session_start + float(tcfg["time_budget_hours"]) * 3600.0 if tcfg.get("time_budget_hours") else None
    install_signal_handlers()

    log(f"[PLAN] train={len(tr_ds)} | val={len(va_ds)} | bs={bs}x{accum} | steps {st['step']} -> {max_steps} "
        f"| Stage {st['stage']} (stage1_steps={stage1_steps}) | amp={use_amp}")

    model.train()
    done = st["step"] >= max_steps
    bad_streak = 0
    reason = "finished" if done else ""
    micro = st["batch_in_epoch"]
    t_log, steps_at_log = time.time(), st["step"]
    acc_tot = acc_ce = acc_ctc = acc_scr = 0.0
    acc_n = 0

    while not done:
        batches = epoch_batches(len(tr_ds), bs, int(tcfg["seed"]), st["epoch"], st["batch_in_epoch"])
        if not batches:
            st["epoch"] += 1
            st["batch_in_epoch"] = 0
            micro = 0
            continue
        loader = DataLoader(tr_ds, batch_sampler=batches, num_workers=nw, pin_memory=pin, collate_fn=unified_collate_fn)
        for batch in loader:
            mb = _move_batch(batch, device)
            with _autocast(device, use_amp):
                out = model(**mb)
                loss = out["loss"]
            if not torch.isfinite(loss):
                bad_streak += 1
                opt.zero_grad(set_to_none=True)
                log(f"  [WARN] non-finite loss at step {st['step']} (streak {bad_streak})")
                if bad_streak >= 25:
                    raise RuntimeError("25 consecutive non-finite losses — aborting.")
                st["batch_in_epoch"] += 1
                micro += 1
                continue
            bad_streak = 0
            scaler.scale(loss / accum).backward()
            acc_tot += float(loss.detach().item())
            acc_ce += float(out["loss_ce"].detach().item())
            acc_ctc += float(out["loss_ctc"].detach().item())
            acc_scr += float(out["loss_script"].detach().item())
            acc_n += 1
            st["batch_in_epoch"] += 1
            micro += 1

            if micro % accum != 0:
                continue

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(tcfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            st["step"] += 1
            step = st["step"]

            if step % int(tcfg["log_every_steps"]) == 0 or step == 1:
                n = max(1, acc_n)
                lrs = sched.get_last_lr()
                lr_enc = float(lrs[0])
                lr_dec = float(lrs[1]) if len(lrs) > 1 else 0.0
                now = time.time()
                sps = (now - t_log) / max(1, step - steps_at_log)
                history.add(
                    "train", step, stage=st["stage"],
                    train_loss=acc_tot / n, train_loss_ce=acc_ce / n,
                    train_loss_ctc=acc_ctc / n, train_loss_script=acc_scr / n,
                    lr_encoder=lr_enc, lr_decoder=lr_dec, sec_per_step=sps,
                )
                log(f"  [S{st['stage']}] step {step:>6}/{max_steps} | loss {acc_tot/n:.4f} "
                    f"(CE {acc_ce/n:.3f}, CTC {acc_ctc/n:.3f}) | lr_enc {lr_enc:.2e} lr_dec {lr_dec:.2e} | {sps:.2f}s/step")
                acc_tot = acc_ce = acc_ctc = acc_scr = 0.0
                acc_n = 0
                t_log, steps_at_log = now, step

            if step % int(tcfg["val_every_steps"]) == 0:
                r = evaluate(
                    model, tok, ctc_vocab, va_ds, device, cfg, use_amp,
                    num_beams=int(tcfg.get("val_num_beams", 1)),
                    max_gen_lines=tcfg.get("val_max_lines"),
                    loss_too=True, want_samples=2,
                )
                history.add(
                    "val", step, stage=st["stage"],
                    val_loss=r["val_loss"], val_cer=r["val_cer"],
                    val_cer_pct=r["val_cer_pct"], val_ortho_cer_pct=r["val_ortho_cer_pct"],
                    val_ctc_cer_pct=r["val_ctc_cer_pct"], val_wer_pct=r["val_wer_pct"],
                )
                log(f"  [VAL S{st['stage']}] step {step} | loss {r['val_loss']:.4f} | CER {r['val_cer_pct']:.2f}% "
                    f"| Ortho {r['val_ortho_cer_pct']:.2f}% | CTC {r['val_ctc_cer_pct']:.2f}% | WER {r['val_wer_pct']:.2f}%")
                if r["val_cer"] < st["best_val_cer"]:
                    st["best_val_cer"] = r["val_cer"]
                    if _disk_free_gb(out_dir) > 2.0:
                        best_p = save_unified_best(out_dir, model, tok, ctc_vocab, cfg, "best")
                        log(f"  [BEST] new best Strict CER {100.0 * st['best_val_cer']:.2f}% -> {best_p}")
                history.save(str(out_dir))
                plot_unified_training_curves(history, out_dir / "plots")
                model.train()

            # Stage 1 -> Stage 2 unfreeze transition
            if st["stage"] == 1 and step >= stage1_steps and step < max_steps:
                st["stage"] = 2
                opt, sched = build_stage_optimizer_and_scheduler(model, tcfg, stage=2)
                log(f"  [STAGE 2] Unfroze TrOCR decoder (enc_lr={tcfg['stage2_encoder_lr']:.1e}, "
                    f"dec_lr={tcfg['stage2_decoder_lr']:.1e})")

            if step % int(tcfg["ckpt_every_steps"]) == 0:
                save_training_state(out_dir, model, opt, sched, scaler, st, history, cfg, "periodic")
            if step % int(tcfg["val_every_steps"]) == 0 or step % int(tcfg["ckpt_every_steps"]) == 0:
                t_log, steps_at_log = time.time(), step

            if deadline and time.time() > deadline and not STOP["flag"]:
                STOP["flag"], STOP["reason"] = True, f"time budget {tcfg['time_budget_hours']} h reached"
            if STOP["flag"]:
                reason, done = STOP["reason"], True
                break
            if step >= max_steps:
                reason, done = "finished", True
                break
        else:
            st["epoch"] += 1
            st["batch_in_epoch"] = 0
            micro = 0

    finished = st["step"] >= max_steps
    history.save(str(out_dir))
    plot_unified_training_curves(history, out_dir / "plots")
    if finished:
        if not (out_dir / "best" / "unified_model.pt").is_file():
            save_unified_best(out_dir, model, tok, ctc_vocab, cfg, "best")
        save_unified_best(out_dir, model, tok, ctc_vocab, cfg, "last")
    save_training_state(out_dir, model, opt, sched, scaler, st, history, cfg, reason or "stop")
    write_status(out_dir, st, cfg, finished, reason)
    log(f"[STATUS] finished={finished} | step={st['step']}/{max_steps} | stage={st['stage']} | reason={reason}")
    return finished


def eval_only(cfg: dict[str, Any], args: argparse.Namespace, device: torch.device) -> None:
    ckpt_dir = Path(args.checkpoint or (Path(args.output_dir) / "best"))
    use_amp = (cfg["train"]["precision"] == "amp_fp16") and (device.type == "cuda")
    model, tok, ctc_vocab, meta = load_unified_from_checkpoint_dir(ckpt_dir, device, tokenizer_override=args.tokenizer)
    ds_map = make_datasets(cfg, tok, ctc_vocab, args.data_dir, [args.split], limit=args.limit or 0)
    ds = ds_map[args.split]
    r = evaluate(
        model, tok, ctc_vocab, ds, device, cfg, use_amp,
        num_beams=int(args.num_beams), max_gen_lines=None,
        loss_too=True, want_samples=3, return_preds=True,
    )
    log(f"\n== {args.split.upper()} EVAL | ckpt={ckpt_dir} | beams={args.num_beams} ==")
    for bk, bv in r["per_book"].items():
        log(f"  {bk} ({bv['script']:10s}) n={bv['n']:4d} | CER={bv['strict_corpus_cer_pct']:6.2f}% "
            f"| Ortho={bv['ortho_corpus_cer_pct']:6.2f}% | WER={bv['true_corpus_wer_pct']:6.2f}% "
            f"| CTC={bv['ctc_corpus_cer_pct']:6.2f}% | Cat={bv['catastrophic_count']}")
    log(f"  OVERALL n={r['n_gen']} | Strict CER={r['val_cer_pct']:.2f}% | Ortho={r['val_ortho_cer_pct']:.2f}% "
        f"| WER={r['val_wer_pct']:.2f}% | CTC CER={r['val_ctc_cer_pct']:.2f}% | Cat={r['val_catastrophic_count']}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preds_dir:
        export_unified_predictions(r, args.save_preds_dir, split=args.split, num_beams=args.num_beams, ckpt=str(ckpt_dir))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train/evaluate Unified Script-Aware Hybrid CTC/Attention HTR.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "ocr_unified_htr.yaml"))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--init-path", help="Path/HF ID for TrOCR decoder initialization (default hatformer-muharaf)")
    ap.add_argument("--tokenizer", help="Path to 50,265-token HATFormer tokenizer.json")
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--stage1-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--val-every", type=int)
    ap.add_argument("--ckpt-every", type=int)
    ap.add_argument("--val-max-lines", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--time-budget-hours", type=float)
    ap.add_argument("--precision", choices=["amp_fp16", "fp32"])
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--checkpoint", help="Saved checkpoint directory (e.g. <output-dir>/best)")
    ap.add_argument("--save-preds-dir", help="Destination folder for prediction CSVs & eval_summary.json")
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--no-multiscale", action="store_true")
    ap.add_argument("--no-ctc", action="store_true")
    ap.add_argument("--no-script-loss", action="store_true")
    ap.add_argument("--no-augment", action="store_true", help="Disable train-time visual augmentations")
    ap.add_argument("--tiny-decoder-test", action="store_true", help="CPU smoke test using tiny 2-layer TrOCRConfig")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    t = cfg["train"]
    for key, val in (
        ("max_steps", args.max_steps), ("stage1_steps", args.stage1_steps),
        ("batch_size", args.batch_size), ("grad_accum", args.grad_accum),
        ("val_every_steps", args.val_every), ("ckpt_every_steps", args.ckpt_every),
        ("val_max_lines", args.val_max_lines), ("time_budget_hours", args.time_budget_hours),
        ("precision", args.precision),
    ):
        if val is not None:
            t[key] = val

    m = cfg["model"]
    if args.no_film:
        m["use_film"] = False
        m["use_script_loss"] = False
    if args.no_multiscale:
        m["use_multiscale"] = False
    if args.no_ctc:
        m["use_ctc"] = False
    if args.no_script_loss:
        m["use_script_loss"] = False
    if args.no_augment:
        cfg.setdefault("data", {})["augment_train"] = False

    if args.tokenizer:
        cfg["tokenizer_path"] = args.tokenizer
    cfg["_init_path"] = args.init_path or cfg.get("checkpoints", {}).get("muharaf", "hatformer-muharaf")

    device = pick_device(allow_cpu=args.allow_cpu)
    if args.preflight:
        preflight(cfg, args, device)
    elif args.eval_only:
        eval_only(cfg, args, device)
    else:
        train(cfg, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
