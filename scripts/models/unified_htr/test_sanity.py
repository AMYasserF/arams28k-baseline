"""Local CPU Sanity Verification Suite for scripts/models/unified_htr/."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, TrOCRConfig, TrOCRForCausalLM

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[2]
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from ctc_vocab import ARAMS28K_TRAIN_CHARS, CTCCharVocab  # noqa: E402
from data_utils import (  # noqa: E402
    BOOK_TO_SCRIPT_ID,
    SCRIPT_ID_TO_NAME,
    load_split_dataframe,
)
from dataset import UnifiedHTRDataset, unified_collate_fn  # noqa: E402
from model import (  # noqa: E402
    TROCR_BOS_ID,
    TROCR_EOS_ID,
    TROCR_PAD_ID,
    UnifiedHTRModel,
    default_hatformer_decoder_config,
    load_trocr_decoder,
)


def build_test_tokenizer(data_dir: Path) -> PreTrainedTokenizerFast:
    """Use local HATFormer tokenizer.json if present; else build deterministic in-memory BBPE tokenizer."""
    for cand in (
        REPO_ROOT / "scripts" / "models" / "hatformer" / "tokenizer" / "tokenizer.json",
        REPO_ROOT / "checkpoints" / "hatformer" / "original" / "hatformer-muharaf" / "tokenizer.json",
    ):
        if cand.is_file() and cand.stat().st_size > 1000:
            tok = PreTrainedTokenizerFast(tokenizer_file=str(cand))
            tok.add_special_tokens({"pad_token": "<pad>", "eos_token": "</s>", "cls_token": "<s>", "bos_token": "<s>"})
            return tok

    # Offline byte-level fallback tokenizer with exact special IDs: <s>=0, <pad>=1, </s>=2, <unk>=3, <mask>=4
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
    pre = pre_tokenizers.ByteLevel(add_prefix_space=False)
    for b_idx in range(256):
        ch_tok = pre_tokenizers.ByteLevel.alphabet()[b_idx]
        if ch_tok not in vocab:
            vocab[ch_tok] = len(vocab)
    # Also ensure every character token is encodable via ByteLevel BPE
    bpe = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    bpe.pre_tokenizer = pre
    bpe.post_processor = processors.RobertaProcessing(sep=("</s>", 2), cls=("<s>", 0), add_prefix_space=False)
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe)
    tok.add_special_tokens({
        "bos_token": "<s>",
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "mask_token": "<mask>",
        "cls_token": "<s>",
    })
    assert tok.bos_token_id == 0 and tok.pad_token_id == 1 and tok.eos_token_id == 2
    return tok


def tiny_trocr_config(vocab_size: int = 512) -> TrOCRConfig:
    return TrOCRConfig(
        vocab_size=vocab_size,
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


def main() -> int:
    data_dir = REPO_ROOT / "data" / "AraMS-28k-HTR"
    assert data_dir.is_dir(), f"Dataset dir not found: {data_dir}"
    print("=" * 72)
    print("Phase 1 Local CPU Sanity Verification — scripts/models/unified_htr/")
    print("=" * 72)

    # ── 1. Official splits & script mapping ───────────────────────────────────
    df_tr = load_split_dataframe(data_dir, "train", exclude_corrupt_train=False)
    df_va = load_split_dataframe(data_dir, "val", exclude_corrupt_train=False)
    df_te = load_split_dataframe(data_dir, "test", exclude_corrupt_train=False)
    assert len(df_tr) == 19739, f"Expected 19739 train lines, got {len(df_tr)}"
    assert len(df_va) == 1486, f"Expected 1486 val lines, got {len(df_va)}"
    assert len(df_te) == 6744, f"Expected 6744 test lines, got {len(df_te)}"

    for split_name, d in (("train", df_tr), ("val", df_va), ("test", df_te)):
        assert set(d["script_id"].unique()).issubset(set(SCRIPT_ID_TO_NAME.keys()))
        assert all(bk in BOOK_TO_SCRIPT_ID for bk in d["book_id"].unique())
    print(f"[PASS 1] Official splits intact: train={len(df_tr)}, val={len(df_va)}, test={len(df_te)} | scripts OK")

    # ── 2. CTC Vocab (38 classes, 0 OOV) ──────────────────────────────────────
    vocab_from_meta = CTCCharVocab.from_metadata(data_dir)
    vocab = CTCCharVocab()
    assert vocab.chars == vocab_from_meta.chars == ARAMS28K_TRAIN_CHARS
    assert vocab.num_classes == 38 and vocab.blank_id == 0
    for split_name, d in (("train", df_tr), ("val", df_va), ("test", df_te)):
        oov = vocab.find_oov_chars(d["gt_normalized"].tolist())
        assert len(oov) == 0, f"Found OOV chars in {split_name}: {oov}"
    sample_txt = df_tr.iloc[0]["gt_normalized"]
    enc_ids = vocab.encode(sample_txt)
    ctc_path: list[int] = [0]
    for tid in enc_ids:
        if ctc_path[-1] == tid:
            ctc_path.append(0)
        ctc_path.extend([tid, tid])  # duplicate frames to test collapsing + blank separation
    ctc_path.append(0)
    assert vocab.decode_greedy(ctc_path) == sample_txt
    print("[PASS 2] CTCCharVocab: 38 classes (blank=0, 1..37 NFC), 0 OOV across train/val/test, round-trip OK")

    # ── 3. Dataset + 3-Channel Ink + Dynamic Collate + 2x2 degenerate guard ──
    tok = build_test_tokenizer(data_dir)
    ds = UnifiedHTRDataset(data_dir=data_dir, split="train", tokenizer=tok, ctc_vocab=vocab, p_blank=0.0)
    assert len(ds) == 19739 and ds.p_blank == 0.0

    idx_maghrebi = int(df_tr.index[df_tr["book_id"] == "book_20"][0])
    idx_ruqah = int(df_tr.index[df_tr["book_id"] == "book_21"][0])
    idx_litho = int(df_tr.index[df_tr["book_id"] == "book_10"][0])
    idx_corrupt_2x2 = int(df_tr.index[df_tr["image_stem"] == "book_27__book_27_page_011_line000"][0])
    picked = [idx_maghrebi, idx_ruqah, idx_litho, idx_corrupt_2x2]

    batch = unified_collate_fn([ds[i] for i in picked])
    pv = batch["pixel_values"]
    mask = batch["encoder_attention_mask"]
    enc_lens = batch["enc_lengths"]
    assert pv.ndim == 4 and pv.shape[0] == 4 and pv.shape[1] == 3 and pv.shape[2] == 64
    assert 16 <= pv.shape[3] <= 1280 and pv.shape[3] % 4 == 0
    assert mask.shape == (4, pv.shape[3] // 4)
    assert torch.isfinite(pv).all(), "Non-finite pixel_values!"
    assert float(pv.min()) >= 0.0 and float(pv.max()) <= 1.0
    for b in range(4):
        w_valid = int(enc_lens[b].item()) * 4
        if w_valid < pv.shape[3]:
            assert float(pv[b, :, :, w_valid:].abs().max()) == 0.0, "Right padding not strictly 0.0!"
        t_v = int(enc_lens[b].item())
        assert bool((mask[b, :t_v] == 1).all()) and (t_v == mask.shape[1] or bool((mask[b, t_v:] == 0).all()))
    print(f"[PASS 3] Dataset & Collate OK: pixel_values={tuple(pv.shape)}, enc_lengths={enc_lens.tolist()}, "
          f"script_ids={batch['script_ids'].tolist()} (incl. 2x2 dummy crop)")

    # ── 4A. Standalone load_trocr_decoder mock verification (strips decoder. & forces start=0) ──
    tiny_cfg = tiny_trocr_config(vocab_size=max(len(tok), 512))
    tiny_dec_ref = TrOCRForCausalLM(tiny_cfg)
    with tempfile.TemporaryDirectory() as tmpd:
        tpath = Path(tmpd)
        nested_cfg = {
            "architectures": ["VisionEncoderDecoderModel"],
            "decoder_start_token_id": 0,
            "decoder": tiny_cfg.to_dict(),
        }
        nested_cfg["decoder"]["decoder_start_token_id"] = 2  # Intentional conflict matching real HF config.json
        (tpath / "config.json").write_text(json.dumps(nested_cfg), encoding="utf-8")
        # Save state dict under "decoder." prefix omitting tied output_projection.weight
        prefixed_sd = {
            f"decoder.{k}": v for k, v in tiny_dec_ref.state_dict().items()
            if k != "output_projection.weight"
        }
        from safetensors.torch import save_file
        save_file(prefixed_sd, str(tpath / "model.safetensors"))
        loaded_dec = load_trocr_decoder(str(tpath))
        assert loaded_dec.config.decoder_start_token_id == 0
        assert loaded_dec.generation_config.decoder_start_token_id == 0
        assert loaded_dec.output_projection.weight is loaded_dec.model.decoder.embed_tokens.weight
    print("[PASS 4A] Standalone load_trocr_decoder: stripped 'decoder.', tied output_projection, forced start_id=0")

    # ── 4B. Full-size TrOCR architecture shape + parameter counts across 4 ablation modes ──
    full_cfg = default_hatformer_decoder_config()
    ablation_configs = [
        ("Full", dict(use_multiscale=True, use_film=True, use_ctc=True, use_script_loss=True)),
        ("No-FiLM", dict(use_multiscale=True, use_film=False, use_ctc=True, use_script_loss=False)),
        ("No-MultiScale", dict(use_multiscale=False, use_film=True, use_ctc=True, use_script_loss=True)),
        ("No-CTC", dict(use_multiscale=True, use_film=True, use_ctc=False, use_script_loss=True)),
    ]
    for name, flags in ablation_configs:
        m_full = UnifiedHTRModel(
            decoder_config=full_cfg,
            ctc_vocab_size=vocab.num_classes,
            pretrained_backbone=False,
            **flags,
        )
        m_full.eval()
        tot_p = sum(p.numel() for p in m_full.parameters())
        enc_p = tot_p - sum(p.numel() for p in m_full.decoder.parameters())
        with torch.no_grad():
            out_ab = m_full(
                pixel_values=pv,
                encoder_attention_mask=mask,
                enc_lengths=enc_lens,
                decoder_labels=batch["decoder_labels"],
                ctc_targets=batch["ctc_targets"],
                ctc_target_lengths=batch["ctc_target_lengths"],
                script_ids=batch["script_ids"],
            )
        assert tuple(out_ab["enc_features"].shape) == (4, pv.shape[3] // 4, 768)
        assert torch.isfinite(out_ab["loss"])
        print(f"  [PASS 4B-{name:13s}] total={tot_p/1e6:.2f}M (encoder={enc_p/1e6:.2f}M) | "
              f"enc_features={tuple(out_ab['enc_features'].shape)} | loss={float(out_ab['loss']):.4f}")
        del m_full

    # ── 4C. FiLM zero-init identity + decoder freeze/unfreeze + generate() ─────
    tiny_model = UnifiedHTRModel(
        decoder_config=tiny_cfg,
        ctc_vocab_size=vocab.num_classes,
        pretrained_backbone=False,
        use_multiscale=True,
        use_film=True,
        use_ctc=True,
        use_script_loss=True,
    )
    # Check exact FiLM identity at init
    tiny_model.eval()
    with torch.no_grad():
        s2_t = torch.randn(2, 128, 8, 16)
        s4_t = torch.randn(2, 512, 2, 16)
        f_t = torch.randn(2, 512, 2, 16)
        m_t = torch.ones(2, 16, dtype=torch.long)
        mod_t, _ = tiny_model.film(s2_t, s4_t, f_t, m_t)
        max_id_diff = float((mod_t - f_t).abs().max())
        assert max_id_diff < 1e-6, f"FiLM not identity at init: {max_id_diff}"

    # Check Stage 1 freeze & Stage 2 unfreeze
    tiny_model.set_decoder_frozen(True)
    assert all(not p.requires_grad for p in tiny_model.decoder.parameters())
    assert any(p.requires_grad for p in tiny_model.backbone.parameters())
    tiny_model.train()
    tiny_model.zero_grad()
    out_frz = tiny_model(
        pixel_values=pv,
        encoder_attention_mask=mask,
        enc_lengths=enc_lens,
        decoder_labels=batch["decoder_labels"],
        ctc_targets=batch["ctc_targets"],
        ctc_target_lengths=batch["ctc_target_lengths"],
        script_ids=batch["script_ids"],
    )
    out_frz["loss"].backward()
    assert all(p.grad is None for p in tiny_model.decoder.parameters())
    assert tiny_model.ctc_head.weight.grad is not None
    assert tiny_model.film.film_mlp[-1].weight.grad is not None

    tiny_model.set_decoder_frozen(False)
    assert all(p.requires_grad for p in tiny_model.decoder.parameters())

    # Check generate() beams=1 and beams=3 + CTC
    dec_g, ctc_g = tiny_model.generate(
        pv, mask, enc_lens, tokenizer=tok, ctc_vocab=vocab, num_beams=3, max_new_tokens=10, return_ctc=True
    )
    assert len(dec_g) == 4 and len(ctc_g) == 4 and all(isinstance(s, str) for s in dec_g)
    print(f"[PASS 4C] FiLM zero-init identity diff={max_id_diff:.2e} | freeze/unfreeze OK | generate(beams=3) OK")

    # ── 4D. 15-step CPU gradient-descent smoke test on 4 real images (3 legible + 1 2x2) ──
    tiny_model.train()
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-4, weight_decay=0.01)
    ce_hist, ctc_hist = [], []
    for step in range(1, 16):
        opt.zero_grad()
        res = tiny_model(
            pixel_values=pv,
            encoder_attention_mask=mask,
            enc_lengths=enc_lens,
            decoder_labels=batch["decoder_labels"],
            ctc_targets=batch["ctc_targets"],
            ctc_target_lengths=batch["ctc_target_lengths"],
            script_ids=batch["script_ids"],
        )
        loss = res["loss"]
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tiny_model.parameters(), 5.0)
        opt.step()
        ce_hist.append(float(res["loss_ce"].item()))
        ctc_hist.append(float(res["loss_ctc"].item()))

    assert ce_hist[-1] < ce_hist[0], f"CE loss did not drop: {ce_hist[0]:.4f} -> {ce_hist[-1]:.4f}"
    assert ctc_hist[-1] < ctc_hist[0], f"CTC loss did not drop: {ctc_hist[0]:.4f} -> {ctc_hist[-1]:.4f}"
    print(f"[PASS 4D] 15-step CPU convergence: CE {ce_hist[0]:.3f} -> {ce_hist[-1]:.3f} | "
          f"CTC {ctc_hist[0]:.3f} -> {ctc_hist[-1]:.3f} | 0 NaN/Inf")
    print("=" * 72)
    print("ALL PHASE 1 SANITY CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
