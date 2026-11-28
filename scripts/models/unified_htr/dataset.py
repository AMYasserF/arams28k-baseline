"""Variable-width 3-channel ink dataset and dynamic batch collator for UnifiedHTR.

No BlockProcessor and no 384x384 row wrapping:
  - Resizes aspect-preserving to fixed height H = 64 and width W in [16, 1280],
    rounded up to a multiple of 4 (so enc_length = W // 4).
  - Applies RTL horizontal flip by default so column 0 aligns with the right start
    of an Arabic line (matching LTR CTC + autoregressive decoder progression).
  - Produces a 3-channel ink-focused float32 tensor in [0.0, 1.0]:
      Ch 0: Inverted grayscale ink intensity ((255 - gray) / 255.0)
      Ch 1: Local contrast-enhanced ink (CLAHE via cv2 if installed, else SciPy/NumPy LCN)
      Ch 2: Soft background-subtracted ink suppressing parchment yellowing/stains
  - Right-side batch padding is strictly 0.0 (matching empty background).
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from .ctc_vocab import CTCCharVocab
    from .data_utils import load_split_dataframe, norm_text
except ImportError:
    from ctc_vocab import CTCCharVocab
    from data_utils import load_split_dataframe, norm_text

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from scipy.ndimage import gaussian_filter, uniform_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None
    uniform_filter = None


def _compute_target_width(orig_w: int, orig_h: int, height: int = 64, min_width: int = 16, max_width: int = 1280) -> int:
    """Compute aspect-preserving width clamped to [min_width, max_width] and rounded up to multiple of 4."""
    safe_w = max(1, int(orig_w))
    safe_h = max(1, int(orig_h))
    raw_w = int(round(height * (safe_w / float(safe_h))))
    clamped = max(min_width, min(max_width, raw_w))
    rounded = ((clamped + 3) // 4) * 4
    return min(max_width, max(min_width, rounded))


def _local_contrast_numpy_scipy(inv_u8: np.ndarray) -> np.ndarray:
    """Pure NumPy/SciPy local contrast fallback when OpenCV is unavailable."""
    arr = inv_u8.astype(np.float32) / 255.0
    if float(arr.max() - arr.min()) < 1e-4:
        return np.zeros_like(arr, dtype=np.float32)
    if uniform_filter is not None:
        mean = uniform_filter(arr, size=(15, 31), mode="nearest")
        sq_mean = uniform_filter(arr * arr, size=(15, 31), mode="nearest")
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 1e-6))
    else:
        mean = np.full_like(arr, float(arr.mean()))
        std = np.full_like(arr, max(float(arr.std()), 1e-3))
    normed = (arr - mean) / (2.5 * std + 0.15)
    return np.clip(0.5 * (normed + 1.0) * (arr > 0.02), 0.0, 1.0).astype(np.float32)


def _soft_background_sub(gray_f: np.ndarray) -> np.ndarray:
    """Estimate smooth local paper background and extract dark ink residual in [0, 1]."""
    # Mask pure white exterior polygon padding (>=253) before blurring when interior pixels exist
    if cv2 is not None:
        bg = cv2.GaussianBlur(gray_f, (31, 31), sigmaX=12.0, borderType=cv2.BORDER_REPLICATE)
    elif gaussian_filter is not None:
        bg = gaussian_filter(gray_f, sigma=12.0, mode="nearest")
    else:
        bg = np.full_like(gray_f, float(np.percentile(gray_f, 90)))
    sub = np.clip((bg - gray_f) / 255.0, 0.0, 1.0)
    # Suppress tiny sensor noise on flat uniform patches (e.g. 2x2 dummy crops)
    if float(gray_f.max() - gray_f.min()) < 1.0:
        return np.zeros_like(sub, dtype=np.float32)
    return sub.astype(np.float32)


def preprocess_ink_3ch(
    img: Image.Image,
    height: int = 64,
    min_width: int = 16,
    max_width: int = 1280,
    rtl_flip: bool = True,
) -> torch.Tensor:
    """Convert PIL line crop into a variable-width (3, H=64, W) float32 tensor in [0, 1]."""
    gray_pil = img.convert("L")
    orig_w, orig_h = gray_pil.size
    target_w = _compute_target_width(orig_w, orig_h, height=height, min_width=min_width, max_width=max_width)

    if orig_w <= 4 and orig_h <= 4:
        # Degenerate dummy crop (e.g., official book_27_page_011 2x2): keep minimal W=min_width safely
        target_w = min_width

    resized = gray_pil.resize((target_w, height), Image.Resampling.BILINEAR)
    if rtl_flip:
        resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    gray_u8 = np.asarray(resized, dtype=np.uint8)
    gray_f = gray_u8.astype(np.float32)

    # Channel 0: Inverted grayscale ink (255 -> 0.0 background, 0 -> 1.0 dark ink)
    inv_u8 = (255 - gray_u8).astype(np.uint8)
    ch0 = inv_u8.astype(np.float32) / 255.0

    # Channel 1: Local contrast enhanced ink (CLAHE with flat-tile protection)
    if float(inv_u8.max() - inv_u8.min()) <= 2.0:
        ch1 = ch0.copy()
    elif cv2 is not None:
        clahe_u8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(inv_u8)
        ch1 = clahe_u8.astype(np.float32) / 255.0
        # Keep pure white exterior polygon padding strictly near zero
        ch1 = np.where(gray_u8 >= 253, 0.0, ch1).astype(np.float32)
    else:
        ch1 = _local_contrast_numpy_scipy(inv_u8)

    # Channel 2: Soft background-subtracted ink suppressing paper yellowing/stains
    ch2 = _soft_background_sub(gray_f)
    ch2 = np.where(gray_u8 >= 253, 0.0, ch2).astype(np.float32)

    stacked = np.stack([ch0, ch1, ch2], axis=0)
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=1.0, neginf=0.0)
    stacked = np.clip(stacked, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(stacked))


def build_decoder_labels(tokenizer: Any, text: str, max_len: int = 64) -> list[int]:
    """Encode text into BBPE token IDs always terminated by </s> and padded with -100."""
    ids = tokenizer(norm_text(text), add_special_tokens=False, truncation=True, max_length=max_len).input_ids
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must have eos_token_id configured before building labels.")
    if not ids or ids[-1] != eos:
        ids = ids[: max_len - 1] + [eos]
    return ids + [-100] * (max_len - len(ids))


class UnifiedHTRDataset(Dataset):
    """Variable-width 3-channel ink + script + dual-target dataset for AraMS-28k."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        tokenizer: Any,
        ctc_vocab: CTCCharVocab | None = None,
        height: int = 64,
        min_width: int = 16,
        max_width: int = 1280,
        rtl_flip: bool = True,
        max_text_length: int = 64,
        main_only: bool = True,
        exclude_corrupt_train: bool = False,
        p_blank: float = 0.0,
        limit: int = 0,
        seed: int = 1337,
    ) -> None:
        self.data_dir = str(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.ctc_vocab = ctc_vocab or CTCCharVocab()
        self.height = int(height)
        self.min_width = int(min_width)
        self.max_width = int(max_width)
        self.rtl_flip = bool(rtl_flip)
        self.max_text_length = int(max_text_length)
        self.p_blank = float(p_blank)  # Disabled (0.0) by default for strict benchmark fairness
        self._rng = random.Random(seed)

        self.df = load_split_dataframe(
            data_dir=data_dir,
            split=split,
            main_only=main_only,
            exclude_corrupt_train=exclude_corrupt_train,
            limit=limit,
        )
        self.paths: list[str] = self.df["resolved_path"].tolist()
        self.texts: list[str] = self.df["gt_normalized"].tolist()
        self.script_ids: list[int] = self.df["script_id"].astype(int).tolist()
        self.line_uids: list[str] = self.df["line_uid"].tolist()
        self.book_ids: list[str] = self.df["book_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Optional disabled blank-drop augmentation hook for future ablations only
        if self.split == "train" and self.p_blank > 0.0 and self._rng.random() < self.p_blank:
            pixel_values = torch.zeros((3, self.height, self.min_width), dtype=torch.float32)
            text = ""
        else:
            with Image.open(self.paths[idx]) as im:
                pixel_values = preprocess_ink_3ch(
                    im,
                    height=self.height,
                    min_width=self.min_width,
                    max_width=self.max_width,
                    rtl_flip=self.rtl_flip,
                )
            text = self.texts[idx]

        dec_labels = build_decoder_labels(self.tokenizer, text, max_len=self.max_text_length)
        ctc_ids = self.ctc_vocab.encode(text)

        return {
            "pixel_values": pixel_values,  # (3, 64, W_i), W_i % 4 == 0
            "orig_width": int(pixel_values.shape[-1]),
            "decoder_labels": torch.tensor(dec_labels, dtype=torch.long),
            "ctc_targets": torch.tensor(ctc_ids, dtype=torch.long),
            "ctc_target_length": len(ctc_ids),
            "script_id": self.script_ids[idx],
            "line_uid": self.line_uids[idx],
            "book_id": self.book_ids[idx],
            "gt_normalized": self.texts[idx],
        }


def unified_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad variable-width batch images with 0.0 and compute exact enc_lengths = W // 4."""
    if not batch:
        raise ValueError("Empty batch passed to unified_collate_fn.")

    bsz = len(batch)
    channels = int(batch[0]["pixel_values"].shape[0])
    height = int(batch[0]["pixel_values"].shape[1])
    widths = [int(item["orig_width"]) for item in batch]
    max_w = max(widths)
    if max_w % 4 != 0:
        max_w = ((max_w + 3) // 4) * 4

    pixel_values = torch.zeros((bsz, channels, height, max_w), dtype=torch.float32)
    enc_lengths = torch.tensor([w // 4 for w in widths], dtype=torch.long)
    t_enc_max = max_w // 4
    encoder_attention_mask = torch.zeros((bsz, t_enc_max), dtype=torch.long)

    max_ctc_len = max((int(item["ctc_target_length"]) for item in batch), default=0)
    max_ctc_len = max(max_ctc_len, 1)
    ctc_targets = torch.zeros((bsz, max_ctc_len), dtype=torch.long)
    ctc_target_lengths = torch.zeros((bsz,), dtype=torch.long)

    decoder_labels = torch.stack([item["decoder_labels"] for item in batch], dim=0)
    script_ids = torch.tensor([int(item["script_id"]) for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        w_i = widths[i]
        pixel_values[i, :, :, :w_i] = item["pixel_values"]
        t_i = int(enc_lengths[i].item())
        encoder_attention_mask[i, :t_i] = 1

        ctc_len = int(item["ctc_target_length"])
        ctc_target_lengths[i] = ctc_len
        if ctc_len > 0:
            ctc_targets[i, :ctc_len] = item["ctc_targets"]

    return {
        "pixel_values": pixel_values,
        "encoder_attention_mask": encoder_attention_mask,
        "enc_lengths": enc_lengths,
        "decoder_labels": decoder_labels,
        "ctc_targets": ctc_targets,
        "ctc_target_lengths": ctc_target_lengths,
        "script_ids": script_ids,
        "line_uid": [item["line_uid"] for item in batch],
        "book_id": [item["book_id"] for item in batch],
        "gt_normalized": [item["gt_normalized"] for item in batch],
    }
