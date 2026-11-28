"""Unified 38-class NFC Arabic CTC character codec shared across all scripts.

Classes:
  - id = 0      : <blank>
  - id = 1..37  : sorted 37 unique Arabic NFC characters observed in official train gt_normalized
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .data_utils import load_split_dataframe, norm_text
except ImportError:
    from data_utils import load_split_dataframe, norm_text


BLANK_ID: int = 0
BLANK_TOKEN: str = "<blank>"

# Exact verified 37-character sorted NFC Arabic inventory from official AraMS-28k train split
ARAMS28K_TRAIN_CHARS: tuple[str, ...] = (
    " ",
    "\u0621", "\u0622", "\u0623", "\u0624", "\u0625", "\u0626",
    "\u0627", "\u0628", "\u0629", "\u062a", "\u062b", "\u062c", "\u062d", "\u062e",
    "\u062f", "\u0630", "\u0631", "\u0632", "\u0633", "\u0634", "\u0635", "\u0636",
    "\u0637", "\u0638", "\u0639", "\u063a",
    "\u0641", "\u0642", "\u0643", "\u0644", "\u0645", "\u0646", "\u0647", "\u0648",
    "\u0649", "\u064a",
)


class CTCCharVocab:
    """Deterministic unified Arabic CTC alphabet codec (blank=0, chars=1..37)."""

    def __init__(self, chars: Sequence[str] = ARAMS28K_TRAIN_CHARS) -> None:
        unique_chars = sorted(set(chars))
        if len(unique_chars) != len(chars):
            raise ValueError("Duplicate characters supplied to CTCCharVocab.")
        self.blank_id: int = BLANK_ID
        self.blank_token: str = BLANK_TOKEN
        self.chars: tuple[str, ...] = tuple(unique_chars)
        self.id_to_token: list[str] = [self.blank_token, *self.chars]
        self.token_to_id: dict[str, int] = {ch: i for i, ch in enumerate(self.chars, start=1)}

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def num_classes(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CTCCharVocab":
        charset: set[str] = set()
        for t in texts:
            charset.update(norm_text(t))
        return cls(sorted(charset))

    @classmethod
    def from_metadata(cls, data_dir: str | Path, main_only: bool = True) -> "CTCCharVocab":
        df_train = load_split_dataframe(
            data_dir=data_dir,
            split="train",
            main_only=main_only,
            exclude_corrupt_train=False,
        )
        return cls.from_texts(df_train["gt_normalized"].tolist())

    def encode(self, text: str, ignore_unknown: bool = False) -> list[int]:
        """Encode normalized string into CTC class IDs in 1..37."""
        cleaned = norm_text(text)
        out: list[int] = []
        for ch in cleaned:
            idx = self.token_to_id.get(ch)
            if idx is None:
                if ignore_unknown:
                    continue
                raise ValueError(
                    f"Out-of-vocabulary character {repr(ch)} (U+{ord(ch):04X}) in text: {cleaned!r}"
                )
            out.append(idx)
        return out

    def decode_greedy(self, ids: Sequence[int]) -> str:
        """Collapse consecutive duplicates and strip <blank>=0 tokens."""
        chars: list[str] = []
        prev: int | None = None
        for raw_idx in ids:
            idx = int(raw_idx)
            if idx == prev:
                continue
            prev = idx
            if idx == self.blank_id:
                continue
            if 0 < idx < len(self.id_to_token):
                chars.append(self.id_to_token[idx])
        return norm_text("".join(chars))

    def decode_batch_greedy(self, batch_ids: Sequence[Sequence[int]]) -> list[str]:
        return [self.decode_greedy(seq) for seq in batch_ids]

    def find_oov_chars(self, texts: Iterable[str]) -> set[str]:
        oov: set[str] = set()
        for t in texts:
            for ch in norm_text(t):
                if ch not in self.token_to_id:
                    oov.add(ch)
        return oov

    def to_json(self, path: str | Path) -> None:
        payload = {"blank_id": self.blank_id, "blank_token": self.blank_token, "chars": list(self.chars)}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "CTCCharVocab":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["chars"])
