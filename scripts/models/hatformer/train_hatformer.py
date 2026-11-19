#!/usr/bin/env python3
"""Fine-tune HATFormer on AraMS-28k-HTR - Kaggle-ready, resumable, GPU-checked.

Why this file differs from the original trainer (see docs/KAGGLE_RUN_GUIDE.md):
  * The model is forced to float32 master weights and trained with fp16 *autocast*
    + GradScaler.  transformers>=5 loads checkpoints in the dtype they were saved
    in (float16), and the old loop trained those fp16 weights directly with
    AdamW(eps=1e-6) -> numerically broken -> loss went UP and predictions were empty.
  * Full training state (model, AdamW, LR schedule, grad-scaler, RNG, data position,
    metric history) is checkpointed atomically every `ckpt_every_steps` and when the
    time budget is reached, so a new Kaggle session continues exactly where the last
    one stopped.
  * LR / loss / accuracy / CER curves are written during training.

Typical use (the notebook does this for you):
    python train_hatformer.py --experiment muharaf_ours \
        --data-dir /kaggle/temp/AraMS-28k-HTR --output-dir /kaggle/working/hatformer_run
    python train_hatformer.py ... --preflight          # 1-minute sanity check
    python train_hatformer.py ... --eval-only --checkpoint <dir>/best --split test
"""
import argparse
import json
import os
import random
import shutil
import signal
import sys
import time

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import HatformerLineDataset, load_manifest_split  # noqa: E402
from hat_utils import (CKPT_META, CKPT_NAME, History, char_error_rate,  # noqa: E402
                       edit_distance, epoch_batches, exact_match, find_resume_checkpoint,
                       lr_factor, norm_text, plot_training_curves)
from seed import set_seed  # noqa: E402

# Paper (AraMS-28k, Table 4) HATFormer test CER, for the final comparison printout.
PAPER_TEST_CER = {"book_03": 37.88, "book_05": 13.26, "book_09": 25.37, "overall": 26.74}
BOOK_SCRIPT = {"book_03": "Maghrebi", "book_05": "Ruq'ah", "book_09": "Naskh"}
EXPECTED_LINES = {"train": 19739, "val": 1486, "test": 6746}

STOP = {"flag": False, "reason": ""}
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    torch.multiprocessing.set_sharing_strategy("file_system")   # avoids 'too many open files' / shm errors
except Exception:
    pass


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# Environment / model
# --------------------------------------------------------------------------- #

def pick_device(allow_cpu=False):
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        p = torch.cuda.get_device_properties(0)
        log(f"[GPU] {p.name} | {p.total_memory / 2**30:.1f} GiB | capability {p.major}.{p.minor} "
            f"| torch {torch.__version__} | cuda {torch.version.cuda}")
        # Prove kernels really run on this GPU (catches 'no kernel image' on unsupported cards)
        x = torch.randn(256, 256, device=dev)
        torch.cuda.synchronize(); _ = (x @ x).sum().item()
        return dev
    if allow_cpu:
        log("[WARN] No GPU - running on CPU (only OK for smoke tests).")
        return torch.device("cpu")
    raise RuntimeError(
        "No CUDA GPU visible to PyTorch. On Kaggle: Notebook sidebar -> Session options -> "
        "Accelerator -> 'GPU T4 x2' (or P100), then restart the session.")


def build_model_and_tokenizer(cfg, init_path, device):
    from transformers import PreTrainedTokenizerFast, VisionEncoderDecoderModel

    tok = PreTrainedTokenizerFast(tokenizer_file=cfg["tokenizer_path"])
    tok.add_special_tokens({"pad_token": "<pad>", "eos_token": "</s>",
                            "cls_token": "<s>", "bos_token": "<s>"})
    try:                                    # transformers >= 4.56 / 5.x
        model = VisionEncoderDecoderModel.from_pretrained(init_path, dtype=torch.float32)
    except TypeError:                       # older releases
        model = VisionEncoderDecoderModel.from_pretrained(init_path, torch_dtype=torch.float32)
    model = model.float()                   # belt and braces: fp32 master weights, always
    bad = {str(p.dtype) for p in model.parameters() if p.dtype != torch.float32}
    assert not bad, f"model still has non-fp32 params: {bad}"

    vocab = model.config.decoder.vocab_size
    ids = {"bos": tok.bos_token_id, "eos": tok.eos_token_id, "pad": tok.pad_token_id}
    assert all(v is not None and v < vocab for v in ids.values()), \
        f"tokenizer special ids {ids} do not fit decoder vocab {vocab} - wrong tokenizer.json?"
    # match the original HATFormer recipe: decoder starts from <s>, pad aligned to our tokenizer
    model.config.decoder_start_token_id = ids["bos"]
    model.config.pad_token_id = ids["pad"]
    model.config.eos_token_id = ids["eos"]
    model.config.vocab_size = vocab
    gc = model.generation_config
    gc.decoder_start_token_id, gc.pad_token_id, gc.eos_token_id = ids["bos"], ids["pad"], ids["eos"]
    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    log(f"[MODEL] {init_path} | {n / 1e6:.1f}M params | fp32 master weights | special ids {ids} | vocab {vocab}")
    return model, tok


def make_datasets(cfg, tok, data_dir, split_names, limit=None):
    tcfg, dcfg = cfg["train"], cfg["data"]
    manifest = os.path.join(data_dir, "metadata.csv")
    common = dict(tokenizer=tok, max_text_length=dcfg["max_text_length"],
                  image_cfg=cfg["image"], root=data_dir)
    out = {}
    for name in split_names:
        df = load_manifest_split(manifest, tcfg.get(f"split_{name}", name), main_only=dcfg.get("main_only", True))
        if limit:
            df = df.head(limit)
        exp = EXPECTED_LINES.get(name)
        note = "" if (limit or exp is None or len(df) == exp) else f"  (!! paper release has {exp})"
        log(f"[DATA] {name:5s}: {len(df):6d} lines, books={sorted(df['book_id'].unique())}{note}")
        out[name] = (df, HatformerLineDataset(df, **common))
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _loader(ds, bs, nw, shuffle=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False)


def _autocast(use_amp):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)


@torch.no_grad()
def evaluate(model, tok, ds, device, cfg, use_amp, num_beams=1, max_gen_lines=None,
             loss_too=True, want_samples=3, books=None, return_preds=False):
    """Teacher-forced loss/token-accuracy on `ds` + generation CER on an evenly
    spaced subset of `max_gen_lines` lines (all lines if None).
    Returns a dict (also `per_book` CER if `books` given)."""
    model.eval()
    tcfg, dcfg = cfg["train"], cfg["data"]
    bs = tcfg.get("eval_batch_size", 32)
    nw = tcfg["num_workers"]
    res = {}

    if loss_too:
        tot_loss = tot_tok = tot_ok = 0.0
        for batch in _loader(ds, bs, nw):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            lab = batch["labels"].to(device, non_blocking=True)
            with _autocast(use_amp):
                out = model(pixel_values=pv, labels=lab, interpolate_pos_encoding=True)
            m = lab != -100
            n = int(m.sum())
            tot_loss += float(out.loss) * n
            tot_ok += int(((out.logits.argmax(-1) == lab) & m).sum())
            tot_tok += n
        res["val_loss"] = tot_loss / max(1, tot_tok)
        res["val_token_acc"] = tot_ok / max(1, tot_tok)

    n = len(ds)
    idx = list(range(n)) if (max_gen_lines is None or max_gen_lines >= n) \
        else sorted(set(np.linspace(0, n - 1, max_gen_lines).astype(int).tolist()))
    gen_ds = ds if len(idx) == n else Subset(ds, idx)
    texts = [ds.texts[i] for i in idx]
    book_of = [ds.df.iloc[i]["book_id"] for i in idx] if books else None

    preds = []
    gen_kw = dict(num_beams=num_beams, max_new_tokens=dcfg["max_text_length"])
    if num_beams > 1:
        gen_kw["length_penalty"] = 0.0
    for batch in _loader(gen_ds, bs, nw):
        pv = batch["pixel_values"].to(device, non_blocking=True)
        with _autocast(use_amp):
            out = model.generate(pv, **gen_kw)
        preds += tok.batch_decode(out.tolist(), skip_special_tokens=True)
    refs = [norm_text(t) for t in texts]
    preds = [norm_text(p) for p in preds]
    res["val_cer"] = char_error_rate(preds, refs)
    res["val_char_acc"] = max(0.0, 1.0 - res["val_cer"])
    res["val_exact_acc"] = exact_match(preds, refs)
    res["n_gen"] = len(refs)
    res["empty_pred_frac"] = sum(1 for p in preds if not p) / max(1, len(preds))
    res["samples"] = list(zip(preds[:want_samples], refs[:want_samples]))
    if books:
        pb = {}
        for b in sorted(set(book_of)):
            ii = [i for i, bb in enumerate(book_of) if bb == b]
            pb[b] = {"cer": char_error_rate([preds[i] for i in ii], [refs[i] for i in ii]), "lines": len(ii)}
        res["per_book"] = pb
        res["overall_cer"] = res["val_cer"]
    if return_preds:
        res["preds"] = preds
        res["refs"] = refs
        res["idx"] = idx
    model.train()
    return res


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def _disk_free_gb(path):
    try:
        return shutil.disk_usage(path).free / 2**30
    except Exception:
        return 1e9


def save_weights_dir(model, tok, path):
    """HF-format weights (fp32) - used for 'best' and 'last' exports."""
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def save_training_state(out_dir, model, opt, sched, scaler, st, history, cfg, reason):
    """Atomic: write ckpt_last.pt.tmp -> os.replace -> sidecar json."""
    path = os.path.join(out_dir, CKPT_NAME)
    need = (sum(p.numel() for p in model.parameters()) * 4 * 3) / 2**30 * 1.1
    if os.path.exists(path):
        need *= 2          # old ckpt stays on disk until the new one is fully written
    if _disk_free_gb(out_dir) < need:
        log(f"  [WARN] only {_disk_free_gb(out_dir):.1f} GiB free, need ~{need:.1f} - skipping checkpoint")
        return False
    payload = {
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(), "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "step": st["step"], "epoch": st["epoch"], "batch_in_epoch": st["batch_in_epoch"],
        "best_cer": st["best_cer"], "history": history.records, "cfg": cfg,
        "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        "format": 2,
    }
    tmp = path + ".tmp"
    t0 = time.time()
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
        with open(os.path.join(out_dir, CKPT_META), "w") as f:
            json.dump({"step": st["step"], "best_cer": st["best_cer"], "reason": reason,
                       "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    except OSError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        log(f"  [WARN] checkpoint save failed ({e}); training continues.")
        return False
    log(f"  [CKPT] step {st['step']} saved ({reason}) in {time.time() - t0:.0f}s -> {path}")
    return True


def load_training_state(path, model, opt, sched, scaler, st, history):
    log(f"[RESUME] loading {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    if scaler.is_enabled() and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    st.update(step=ck["step"], epoch=ck["epoch"], batch_in_epoch=ck["batch_in_epoch"], best_cer=ck["best_cer"])
    history.records = list(ck["history"])
    try:
        r = ck["rng"]
        random.setstate(r["py"]); np.random.set_state(r["np"]); torch.set_rng_state(r["torch"])
        if r.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(r["cuda"])
    except Exception as e:
        log(f"  (rng restore skipped: {e})")
    log(f"[RESUME] continuing from step {st['step']} (epoch {st['epoch']}, batch {st['batch_in_epoch']}), "
        f"best CER so far {100 * st['best_cer']:.2f}%")
    del ck


def write_status(out_dir, st, cfg, finished, reason):
    d = {"finished": finished, "step": st["step"], "max_steps": cfg["train"]["max_steps"],
         "best_cer": None if st["best_cer"] == float("inf") else st["best_cer"], "reason": reason,
         "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "status.json"), "w") as f:
        json.dump(d, f, indent=2)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def install_signal_handlers():
    def _h(signum, _frm):
        STOP["flag"], STOP["reason"] = True, f"signal {signum}"
        log(f"\n[SIGNAL] {signum} received - will checkpoint and exit at the next step boundary")
    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _h)
        except Exception:
            pass


def print_sample_block(prefix, samples):
    for p, r in samples:
        log(f"{prefix}pred: {p}\n{prefix}ref : {r}")


def preflight(cfg, args, device):
    """~1 minute: proves model/tokenizer/data/GPU are wired correctly BEFORE a long run."""
    use_amp = cfg["train"]["precision"] == "amp_fp16" and device.type == "cuda"
    exp = cfg["experiments"][args.experiment]
    model, tok = build_model_and_tokenizer(cfg, cfg["_init_path"], device)
    d = make_datasets(cfg, tok, args.data_dir, ["train", "val"], limit=args.limit)
    tr_ds, va_ds = d["train"][1], d["val"][1]
    s = tr_ds.texts[0]
    ids = tok(s).input_ids
    log(f"[TOKENS] text  : {s}\n         ids   : {ids}\n         decode: {tok.decode(ids)}")
    log(f"         labels: {tr_ds[0]['labels'].tolist()[:len(ids) + 2]}")
    model.train()
    b = next(iter(_loader(tr_ds, 8, 0)))
    pv, lab = b["pixel_values"].to(device), b["labels"].to(device)
    with torch.no_grad(), _autocast(use_amp):
        loss = model(pixel_values=pv, labels=lab, interpolate_pos_encoding=True).loss
    log(f"[CHECK] loss of the *pretrained* init on 8 train lines: {loss.item():.3f} "
        f"(random-init would be ~{np.log(model.config.decoder.vocab_size):.1f})")
    r = evaluate(model, tok, va_ds, device, cfg, use_amp, max_gen_lines=16, loss_too=False, want_samples=4)
    log(f"[CHECK] greedy CER of the pretrained init on 16 val lines: {100 * r['val_cer']:.1f}%  "
        f"(empty predictions: {100 * r['empty_pred_frac']:.0f}%)")
    print_sample_block("        ", r["samples"])
    log("[PREFLIGHT OK] GPU, fp32 weights, tokenizer, data paths and generation all work.")


def train(cfg, args, device):
    tcfg = cfg["train"]
    set_seed(tcfg["seed"])
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    use_amp = tcfg["precision"] == "amp_fp16" and device.type == "cuda"
    log(f"== HATFormer fine-tune | experiment={args.experiment} | precision={'fp16-autocast+fp32 weights' if use_amp else 'fp32'} "
        f"| out={out_dir} ==")

    model, tok = build_model_and_tokenizer(cfg, cfg["_init_path"], device)
    d = make_datasets(cfg, tok, args.data_dir, ["train", "val"], limit=args.limit)
    (_, train_ds), (_, val_ds) = d["train"], d["val"]

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], betas=tuple(tcfg["betas"]),
                            eps=tcfg["eps"], weight_decay=tcfg["weight_decay"])
    kind, warm, max_steps = tcfg["lr_scheduler"], tcfg["warmup_steps"], tcfg["max_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warm, kind, max_steps))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    st = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "best_cer": float("inf")}
    history = History()

    if args.resume != "no":
        roots = [out_dir, "/kaggle/input"] if args.resume == "auto" else [args.resume]
        skip = ("images", "__pycache__", ".git") + (() if "smoke" in os.path.basename(out_dir.rstrip("/")) else ("smoke_run",))
        ck = args.resume if (args.resume != "auto" and args.resume.endswith(".pt")) \
            else find_resume_checkpoint(roots, skip_dirs=skip)
        if ck:
            load_training_state(ck, model, opt, sched, scaler, st, history)
            src_best = os.path.join(os.path.dirname(ck), "best")
            if os.path.isdir(src_best) and not os.path.isdir(os.path.join(out_dir, "best")):
                log(f"[RESUME] copying previous best/ model from {src_best}")
                shutil.copytree(src_best, os.path.join(out_dir, "best"))
        else:
            log("[RESUME] no checkpoint found - starting from the pretrained init (step 0)")

    accum, bs = tcfg["grad_accum"], tcfg["batch_size"]
    n_batches = (len(train_ds) + bs - 1) // bs
    session_start = float(os.environ.get("SESSION_START_TS", time.time()))
    deadline = session_start + tcfg["time_budget_hours"] * 3600 if tcfg["time_budget_hours"] else None
    log(f"[PLAN] {len(train_ds)} train lines | batch {bs} x accum {accum} | {n_batches} batches/epoch | "
        f"steps {st['step']} -> {max_steps} | lr {tcfg['lr']:.1e} ({kind}, warmup {warm}) | "
        f"time budget {tcfg['time_budget_hours']} h")
    install_signal_handlers()

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    done = st["step"] >= max_steps
    bad_streak = 0
    t_log, steps_at_log = time.time(), st["step"]
    acc_loss = torch.zeros((), device=device); acc_ok = torch.zeros((), device=device)
    acc_tok = torch.zeros((), device=device); acc_n = 0
    reason = "finished" if done else ""
    micro = st["batch_in_epoch"]

    while not done:
        batches = epoch_batches(len(train_ds), bs, tcfg["seed"], st["epoch"], st["batch_in_epoch"])
        if not batches:
            st["epoch"] += 1; st["batch_in_epoch"] = 0; micro = 0
            continue
        loader = DataLoader(train_ds, batch_sampler=batches, num_workers=tcfg["num_workers"],
                            pin_memory=device.type == "cuda")
        for batch in loader:
            pv = batch["pixel_values"].to(device, non_blocking=True)
            lab = batch["labels"].to(device, non_blocking=True)
            with _autocast(use_amp):
                out = model(pixel_values=pv, labels=lab, interpolate_pos_encoding=True)
            loss = out.loss
            if not torch.isfinite(loss):
                bad_streak += 1
                opt.zero_grad(set_to_none=True)
                log(f"  [WARN] non-finite loss at step {st['step']} (streak {bad_streak})")
                if bad_streak >= 25:
                    raise RuntimeError("25 consecutive non-finite losses - aborting (try precision: fp32)")
                st["batch_in_epoch"] += 1; micro += 1
                continue
            bad_streak = 0
            scaler.scale(loss / accum).backward()
            with torch.no_grad():
                m = lab != -100
                acc_ok += ((out.logits.argmax(-1) == lab) & m).sum()
                acc_tok += m.sum()
                acc_loss += loss.detach(); acc_n += 1
            st["batch_in_epoch"] += 1; micro += 1

            if micro % accum != 0:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=tcfg["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            opt.zero_grad(set_to_none=True)
            st["step"] += 1
            step = st["step"]

            if step % tcfg["log_every_steps"] == 0 or step == 1:
                n = max(1, acc_n)
                tl, ta = float(acc_loss) / n, float(acc_ok) / max(1.0, float(acc_tok))
                now = time.time()
                sps = (now - t_log) / max(1, step - steps_at_log)
                lr = sched.get_last_lr()[0]
                history.add("train", step, train_loss=tl, train_token_acc=ta, lr=lr, sec_per_step=sps,
                            loss_scale=float(scaler.get_scale()) if use_amp else 1.0)
                eta = (max_steps - step) * sps / 3600
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                log(f"  step {step:>6}/{max_steps} | loss {tl:.4f} | tok-acc {100 * ta:5.1f}% | lr {lr:.2e} | "
                    f"{sps:.2f}s/step | gpu-mem {mem:.1f}G | ETA {eta:.1f}h")
                acc_loss.zero_(); acc_ok.zero_(); acc_tok.zero_(); acc_n = 0
                t_log, steps_at_log = now, step

            if step % tcfg["val_every_steps"] == 0:
                t0 = time.time()
                r = evaluate(model, tok, val_ds, device, cfg, use_amp, num_beams=tcfg["val_num_beams"],
                             max_gen_lines=tcfg["val_max_lines"])
                history.add("val", step, **{k: r[k] for k in ("val_loss", "val_token_acc", "val_cer",
                                                              "val_char_acc", "val_exact_acc", "empty_pred_frac")})
                log(f"  [val] step {step} | loss {r['val_loss']:.4f} | CER {100 * r['val_cer']:.2f}% | "
                    f"char-acc {100 * r['val_char_acc']:.2f}% | exact-line {100 * r['val_exact_acc']:.1f}% | "
                    f"tok-acc {100 * r['val_token_acc']:.1f}% | empty {100 * r['empty_pred_frac']:.0f}% | {time.time() - t0:.0f}s")
                print_sample_block("        ", r["samples"])
                if r["empty_pred_frac"] > 0.9 and step >= 3000:
                    log("  [WARN] >90% of validation predictions are EMPTY - training is not working.")
                if r["val_cer"] < st["best_cer"]:
                    st["best_cer"] = r["val_cer"]
                    if _disk_free_gb(out_dir) > 3:
                        save_weights_dir(model, tok, os.path.join(out_dir, "best"))
                        with open(os.path.join(out_dir, "best", "best_meta.json"), "w") as f:
                            json.dump({"step": step, "val_cer": r["val_cer"]}, f)
                        log(f"  [BEST] new best CER {100 * st['best_cer']:.2f}% -> {out_dir}/best")
                history.save(out_dir); plot_training_curves(history, os.path.join(out_dir, "plots"))
                model.train()

            if step % tcfg["ckpt_every_steps"] == 0:
                save_training_state(out_dir, model, opt, sched, scaler, st, history, cfg, "periodic")
            if step % tcfg["val_every_steps"] == 0 or step % tcfg["ckpt_every_steps"] == 0:
                t_log, steps_at_log = time.time(), step     # keep s/step & ETA free of val/ckpt time

            if deadline and time.time() > deadline and not STOP["flag"]:
                STOP["flag"], STOP["reason"] = True, f"time budget {tcfg['time_budget_hours']} h reached"
            if STOP["flag"]:
                reason = STOP["reason"]
                done = True
                break
            if step >= max_steps:
                reason, done = "finished", True
                break
        else:
            st["epoch"] += 1; st["batch_in_epoch"] = 0; micro = 0   # epoch completed normally

    finished = st["step"] >= max_steps
    history.save(out_dir); plot_training_curves(history, os.path.join(out_dir, "plots"))
    if finished:
        last = history.records[-1] if history.records else None
        if last and last["kind"] == "val" and last["step"] == st["step"]:
            r = last                                   # validation just ran at this very step
        else:
            r = evaluate(model, tok, val_ds, device, cfg, use_amp, num_beams=tcfg["val_num_beams"],
                         max_gen_lines=tcfg["val_max_lines"])
            history.add("val", st["step"], **{k: r[k] for k in ("val_loss", "val_token_acc", "val_cer",
                                                                "val_char_acc", "val_exact_acc", "empty_pred_frac")})
        log(f"== DONE | final val loss {r['val_loss']:.4f} | CER {100 * r['val_cer']:.2f}% | "
            f"best CER {100 * min(st['best_cer'], r['val_cer']):.2f}% ==")
        if r["val_cer"] < st["best_cer"]:
            st["best_cer"] = r["val_cer"]
            save_weights_dir(model, tok, os.path.join(out_dir, "best"))
        save_weights_dir(model, tok, os.path.join(out_dir, "last"))
        history.save(out_dir); plot_training_curves(history, os.path.join(out_dir, "plots"))
    save_training_state(out_dir, model, opt, sched, scaler, st, history, cfg, reason or "stop")
    write_status(out_dir, st, cfg, finished, reason)
    if finished:
        log("[STATUS] TRAINING COMPLETE.")
    else:
        log(f"[STATUS] STOPPED EARLY ({reason}) at step {st['step']}/{max_steps}. "
            f"Start a new session with this notebook's output attached as input to resume.")
    return finished


def export_prediction_csvs(df, ds, preds, gen_indices, save_preds_dir, data_dir,
                           split="test", num_beams=1, ckpt="", trainer_results=None):
    """Export per-book prediction CSVs in exact 1-to-1 schema compatibility with Kraken EDA.
    Outputs:
      - <save_preds_dir>/hatformer_preds_book_03.csv (Maghrebi)
      - <save_preds_dir>/hatformer_preds_book_05.csv (Ruq'ah)
      - <save_preds_dir>/hatformer_preds_book_09.csv (Naskh)
      - <save_preds_dir>/eval_summary.json
    """
    os.makedirs(save_preds_dir, exist_ok=True)

    # 1. Build stem-to-global-index mapping for exact line_uid alignment with test_manifest / Kraken
    manifest_file = os.path.join(data_dir, "test_manifest.txt")
    stem_to_idx = {}
    if os.path.isfile(manifest_file):
        with open(manifest_file, encoding="utf-8") as f:
            stems = [Path(line.strip().replace("\\", "/")).stem for line in f if line.strip()]
        stem_to_idx = {stem: i for i, stem in enumerate(stems)}
    else:
        meta_file = os.path.join(data_dir, "metadata.csv")
        if os.path.isfile(meta_file):
            meta_full = pd.read_csv(meta_file)
            test_meta = meta_full[meta_full["split"] == "test"]
            if "line_type" in test_meta.columns:
                test_meta = test_meta[test_meta["line_type"].fillna("main") == "main"]
            stem_to_idx = {Path(str(p).replace("\\", "/")).stem: i for i, p in enumerate(test_meta["img_path"])}

    records = []
    for k, idx_val in enumerate(gen_indices):
        row = df.iloc[idx_val]
        img_p = str(row.get("img_path", ""))
        stem = Path(img_p.replace("\\", "/")).stem
        book_id = str(row.get("book_id", ""))

        global_idx = stem_to_idx.get(stem, idx_val)
        line_uid = f"{book_id}_line_{global_idx:06d}"

        gt_raw = str(row.get("gt_text", ""))
        gt_norm = norm_text(gt_raw)

        pred_val = preds[k]
        pred_norm = norm_text(pred_val) if pred_val is not None else ""
        if not pred_norm:
            pred_norm = ""

        records.append({
            "line_uid": line_uid,
            "book_id": book_id,
            "image_stem": stem,
            "gt_normalized": gt_norm,
            "hatformer_pred": pred_norm,
        })

    res_df = pd.DataFrame(records)

    summary = {
        "checkpoint": ckpt,
        "split": split,
        "num_beams": num_beams,
        "export_dir": os.path.abspath(save_preds_dir),
        "overall": {},
        "per_book": {}
    }

    log("\n" + "=" * 68)
    log(f"  [EXPORT] WRITING PER-BOOK PREDICTION CSVS -> {save_preds_dir}")
    log("=" * 68)

    tot_edits = 0
    tot_chars = 0
    tot_cer_sum = 0.0
    tot_lines = len(res_df)

    books_present = sorted(res_df["book_id"].unique())
    for b_id in books_present:
        b_df = res_df[res_df["book_id"] == b_id].copy()

        b_edits = sum(edit_distance(p, r) for p, r in zip(b_df["hatformer_pred"], b_df["gt_normalized"]))
        b_chars = sum(max(1, len(r)) for r in b_df["gt_normalized"])
        b_line_cers = [edit_distance(p, r) / max(1, len(r)) for p, r in zip(b_df["hatformer_pred"], b_df["gt_normalized"])]

        b_corpus_cer = b_edits / max(1, b_chars) if b_chars > 0 else 0.0
        b_mean_cer = sum(b_line_cers) / max(1, len(b_line_cers)) if b_line_cers else 0.0

        tot_edits += b_edits
        tot_chars += b_chars
        tot_cer_sum += sum(b_line_cers)

        csv_filename = f"hatformer_preds_{b_id}.csv"
        csv_path = os.path.join(save_preds_dir, csv_filename)
        b_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        script_name = BOOK_SCRIPT.get(b_id, b_id)
        log(f"  {b_id} ({script_name:9s}): {len(b_df):5d} lines | "
            f"Corpus CER: {100 * b_corpus_cer:6.2f}% | Mean CER: {100 * b_mean_cer:6.2f}% -> {csv_filename}")

        summary["per_book"][b_id] = {
            "script": script_name,
            "lines": len(b_df),
            "csv_file": csv_filename,
            "corpus_cer": float(b_corpus_cer),
            "corpus_cer_pct": round(float(100 * b_corpus_cer), 4),
            "mean_cer": float(b_mean_cer),
            "mean_cer_pct": round(float(100 * b_mean_cer), 4),
        }
        if trainer_results and "per_book" in trainer_results and b_id in trainer_results["per_book"]:
            summary["per_book"][b_id]["trainer_cer_pct"] = round(float(100 * trainer_results["per_book"][b_id]["cer"]), 4)

    overall_corpus_cer = tot_edits / max(1, tot_chars) if tot_chars > 0 else 0.0
    overall_mean_cer = tot_cer_sum / max(1, tot_lines) if tot_lines > 0 else 0.0
    macro_mean_cer = sum(b["mean_cer"] for b in summary["per_book"].values()) / max(1, len(summary["per_book"])) if summary["per_book"] else 0.0
    macro_corpus_cer = sum(b["corpus_cer"] for b in summary["per_book"].values()) / max(1, len(summary["per_book"])) if summary["per_book"] else 0.0

    log(f"  OVERALL          : {tot_lines:5d} lines | "
        f"Corpus CER: {100 * overall_corpus_cer:6.2f}% | Mean CER: {100 * overall_mean_cer:6.2f}% | "
        f"Macro CER: {100 * macro_corpus_cer:6.2f}%")

    summary["overall"] = {
        "lines": tot_lines,
        "corpus_cer": float(overall_corpus_cer),
        "corpus_cer_pct": round(float(100 * overall_corpus_cer), 4),
        "mean_cer": float(overall_mean_cer),
        "mean_cer_pct": round(float(100 * overall_mean_cer), 4),
        "macro_mean_cer": float(macro_mean_cer),
        "macro_mean_cer_pct": round(float(100 * macro_mean_cer), 4),
        "macro_corpus_cer": float(macro_corpus_cer),
        "macro_corpus_cer_pct": round(float(100 * macro_corpus_cer), 4),
    }
    if trainer_results and "overall_cer" in trainer_results:
        summary["overall"]["trainer_cer_pct"] = round(float(100 * trainer_results["overall_cer"]), 4)

    summary_path = os.path.join(save_preds_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"  [EXPORT] Saved verification summary -> {summary_path}")
    log("=" * 68 + "\n")
    return summary


def eval_only(cfg, args, device):
    use_amp = cfg["train"]["precision"] == "amp_fp16" and device.type == "cuda"
    ckpt = args.checkpoint or os.path.join(args.output_dir, "best")
    model, tok = build_model_and_tokenizer(cfg, ckpt, device)
    d = make_datasets(cfg, tok, args.data_dir, [args.split], limit=args.limit)
    df, ds = d[args.split]
    t0 = time.time()
    want_preds = bool(args.save_preds_dir)
    r = evaluate(model, tok, ds, device, cfg, use_amp, num_beams=args.num_beams, max_gen_lines=None,
                 loss_too=True, books=True, want_samples=5, return_preds=want_preds)
    log(f"\n== {args.split.upper()} results | checkpoint={ckpt} | beams={args.num_beams} | {time.time() - t0:.0f}s ==")
    rows = {}
    for b, v in r["per_book"].items():
        paper = PAPER_TEST_CER.get(b)
        rows[b] = {"script": BOOK_SCRIPT.get(b, "?"), "lines": v["lines"], "cer_pct": 100 * v["cer"], "paper_hatformer_cer_pct": paper}
        log(f"  {b} ({BOOK_SCRIPT.get(b, '?'):9s}) lines={v['lines']:5d}  CER {100 * v['cer']:6.2f}%   paper: {paper}")
    log(f"  OVERALL CER {100 * r['overall_cer']:.2f}%   (paper HATFormer: {PAPER_TEST_CER['overall']}%)   "
        f"exact-line acc {100 * r['val_exact_acc']:.1f}%   loss {r['val_loss']:.4f}")
    print_sample_block("   ", r["samples"])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.split}_results_beams{args.num_beams}.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": ckpt, "split": args.split, "num_beams": args.num_beams, "per_book": rows,
                   "overall_cer_pct": 100 * r["overall_cer"], "exact_line_acc": r["val_exact_acc"],
                   "loss": r["val_loss"], "token_acc": r["val_token_acc"]}, f, indent=2, ensure_ascii=False)
    if args.save_preds_dir:
        export_prediction_csvs(
            df=df, ds=ds, preds=r["preds"], gen_indices=r["idx"],
            save_preds_dir=args.save_preds_dir, data_dir=args.data_dir,
            split=args.split, num_beams=args.num_beams, ckpt=ckpt,
            trainer_results=r
        )


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="muharaf_ours", choices=["muharaf_ours", "synth_ours"])
    ap.add_argument("--config", default=os.path.join(HERE, "..", "..", "..", "configs", "ocr_hatformer.yaml"))
    ap.add_argument("--data-dir", required=True, help="folder containing metadata.csv and images/")
    ap.add_argument("--output-dir", required=True, help="where checkpoints/plots/logs go (use /kaggle/working/...)")
    ap.add_argument("--init-path", help="override: HF folder of the pretrained HATFormer checkpoint")
    ap.add_argument("--tokenizer", help="override: path to tokenizer.json")
    ap.add_argument("--resume", default="auto", help="'auto' | 'no' | path to ckpt_last.pt")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--time-budget-hours", type=float)
    ap.add_argument("--ckpt-every", type=int)
    ap.add_argument("--val-every", type=int)
    ap.add_argument("--val-max-lines", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--warmup-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--precision", choices=["amp_fp16", "fp32"])
    ap.add_argument("--limit", type=int, help="use only N lines per split (smoke tests)")
    ap.add_argument("--preflight", action="store_true", help="sanity-check everything, then exit")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", help="eval-only: HF folder to evaluate (default <output-dir>/best)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--save-preds-dir", help="export per-book prediction CSVs and eval_summary.json to this folder")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    t = cfg["train"]
    for key, val in (("max_steps", args.max_steps), ("time_budget_hours", args.time_budget_hours),
                     ("ckpt_every_steps", args.ckpt_every), ("val_every_steps", args.val_every),
                     ("val_max_lines", args.val_max_lines), ("lr", args.lr), ("precision", args.precision),
                     ("warmup_steps", args.warmup_steps), ("batch_size", args.batch_size),
                     ("grad_accum", args.grad_accum)):
        if val is not None:
            t[key] = val
    if args.tokenizer:
        cfg["tokenizer_path"] = args.tokenizer
    exp = cfg["experiments"][args.experiment]
    cfg["_init_path"] = args.init_path or cfg["checkpoints"][exp["init"]]

    device = pick_device(args.allow_cpu)
    log(f"[CFG] lr={t['lr']} betas={t['betas']} eps={t['eps']} wd={t['weight_decay']} bs={t['batch_size']} "
        f"warmup={t['warmup_steps']} clip={t['grad_clip']} max_steps={t['max_steps']} sched={t['lr_scheduler']} "
        f"precision={t['precision']}")
    if args.preflight:
        preflight(cfg, args, device)
    elif args.eval_only:
        eval_only(cfg, args, device)
    else:
        train(cfg, args, device)


if __name__ == "__main__":
    main()
