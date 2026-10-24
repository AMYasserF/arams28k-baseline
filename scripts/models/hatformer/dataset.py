"""Line-image dataset for HATFormer, reading the manifest + split directly.

The image pipeline mirrors the original HATFormer preprocessing exactly:
resize to a fixed height, flip horizontally (Arabic is RTL), then paste into a
square `canvas`x`canvas` image, stacking lines wider than the canvas as rows.
Because the canvas is already exactly `canvas` px, the ViT/TrOCR image
normalization (rescale 1/255, then mean/std 0.5) is applied directly here.

Changes vs. the original file (behaviour of the model input is unchanged):
  * image paths are resolved robustly (absolute path, relative to the dataset
    folder, or by file name inside <dataset>/images) and the dataset FAILS
    LOUDLY at construction time if images are missing;
  * labels always end with </s> (added if the tokenizer does not add it), so the
    decoder learns to stop and greedy decoding cannot run on forever;
  * optional `main_only` filter (the paper trains on main-text lines only).
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# trocr-base-stage1 / ViT image processor constants
_IMG_MEAN = 0.5
_IMG_STD = 0.5


def hatformer_canvas(img, height=64, canvas=384, rtl_flip=True):
    """Replicate HATFormer line preprocessing -> a single `canvas`x`canvas` RGB image."""
    img = img.convert("RGB")
    w, h = img.size
    new_w = max(1, int(height * (w / h)))
    resized = img.resize((new_w, height))
    if rtl_flip:
        resized = resized.transpose(Image.FLIP_LEFT_RIGHT)

    out = Image.new("RGB", (canvas, canvas), (0, 0, 0))
    if resized.width <= canvas:
        out.paste(resized, (0, 0))
    else:
        n = (resized.width + canvas - 1) // canvas
        for i in range(n):
            left = i * canvas
            seg = resized.crop((left, 0, min(left + canvas, resized.width), height))
            out.paste(seg, (0, i * height))  # rows beyond canvas//height are clipped (as in the original)
    return out


def to_pixel_values(pil_img):
    """HWC uint8 RGB -> CHW float32 tensor, normalized like the ViT image processor."""
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = (arr - _IMG_MEAN) / _IMG_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(arr))


def load_manifest_split(metadata_csv, split, exclude_books=None, books=None, main_only=True):
    """Filter metadata.csv by split, dropping empty transcriptions (and margin lines)."""
    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == split]
    if main_only and "line_type" in df.columns:
        df = df[df["line_type"].fillna("main") == "main"]
    if exclude_books:
        df = df[~df["book_id"].isin(exclude_books)]
    if books:
        df = df[df["book_id"].isin(books)]
    df = df.dropna(subset=["img_path", "gt_text"])
    df = df[df["gt_text"].astype(str).str.strip().astype(bool)]
    return df.reset_index(drop=True)


def resolve_image_paths(df, root):
    """Return a list with one existing file path per row, or raise with details."""
    root = os.path.abspath(root)
    out, missing = [], []
    for p in df["img_path"].astype(str):
        p = p.replace("\\", "/")
        base = os.path.basename(p)
        cands = []
        if os.path.isabs(p):
            cands.append(p)
        cands += [os.path.join(root, p), os.path.join(root, "images", base),
                  os.path.join(os.path.dirname(root), p)]
        for c in cands:
            if os.path.isfile(c):
                out.append(os.path.normpath(c))
                break
        else:
            out.append(None)
            missing.append(p)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(df)} line images not found under '{root}'.\n"
            f"  e.g. {missing[:3]}\n"
            f"  -> check that the AraMS-28k-HTR folder was unzipped and --data-dir points at "
            f"the folder that contains metadata.csv and images/.")
    return out


def build_labels(tokenizer, text, max_len):
    """token ids, always terminated by </s>, right-padded with -100 to `max_len`."""
    ids = tokenizer(str(text), truncation=True, max_length=max_len).input_ids
    eos = tokenizer.eos_token_id
    if not ids or ids[-1] != eos:
        ids = ids[: max_len - 1] + [eos]
    return ids + [-100] * (max_len - len(ids))


class HatformerLineDataset(Dataset):
    def __init__(self, df, tokenizer, max_text_length, image_cfg, root="."):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.image_cfg = dict(image_cfg)
        self.root = root
        self.paths = resolve_image_paths(self.df, root)  # fail early, not at step 3000
        self.texts = self.df["gt_text"].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            canvas = hatformer_canvas(im, **self.image_cfg)
        pixel_values = to_pixel_values(canvas)
        labels = build_labels(self.tokenizer, self.texts[i], self.max_text_length)
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
