"""Unified Script-Aware Hybrid CTC/Attention HTR Package (`unified_htr`)."""

try:
    from .ctc_vocab import ARAMS28K_TRAIN_CHARS, BLANK_ID, CTCCharVocab
    from .data_utils import (
        BOOK_TO_SCRIPT_ID,
        BOOK_TO_SCRIPT_NAME,
        NUM_SCRIPTS,
        SCRIPT_ID_TO_NAME,
        SCRIPT_NAME_TO_ID,
        load_split_dataframe,
        norm_text,
    )
    from .dataset import UnifiedHTRDataset, preprocess_ink_3ch, unified_collate_fn
    from .model import UnifiedHTRModel, load_trocr_decoder
except ImportError:
    from ctc_vocab import ARAMS28K_TRAIN_CHARS, BLANK_ID, CTCCharVocab
    from data_utils import (
        BOOK_TO_SCRIPT_ID,
        BOOK_TO_SCRIPT_NAME,
        NUM_SCRIPTS,
        SCRIPT_ID_TO_NAME,
        SCRIPT_NAME_TO_ID,
        load_split_dataframe,
        norm_text,
    )
    from dataset import UnifiedHTRDataset, preprocess_ink_3ch, unified_collate_fn
    from model import UnifiedHTRModel, load_trocr_decoder

__all__ = [
    "ARAMS28K_TRAIN_CHARS",
    "BLANK_ID",
    "BOOK_TO_SCRIPT_ID",
    "BOOK_TO_SCRIPT_NAME",
    "CTCCharVocab",
    "NUM_SCRIPTS",
    "SCRIPT_ID_TO_NAME",
    "SCRIPT_NAME_TO_ID",
    "UnifiedHTRDataset",
    "UnifiedHTRModel",
    "load_split_dataframe",
    "load_trocr_decoder",
    "norm_text",
    "preprocess_ink_3ch",
    "unified_collate_fn",
]
