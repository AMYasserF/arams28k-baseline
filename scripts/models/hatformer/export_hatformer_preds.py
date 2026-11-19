#!/usr/bin/env python3
"""Standalone prediction exporter for HATFormer on AraMS-28k.

Exports per-book prediction CSVs in exact 1-to-1 schema compatibility with
the Kraken EDA pipeline (line_uid, book_id, image_stem, gt_normalized, hatformer_pred).

Usage:
    python export_hatformer_preds.py --checkpoint /path/to/best \
        --data-dir /path/to/AraMS-28k-HTR --save-preds-dir /path/to/predictions
"""
import argparse
import os
import sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_hatformer import (  # noqa: E402
    build_model_and_tokenizer,
    evaluate,
    export_prediction_csvs,
    make_datasets,
    pick_device,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Export HATFormer predictions to Kraken-compatible CSVs.")
    ap.add_argument("--checkpoint", required=True, help="HF folder of the fine-tuned model (e.g. <run_dir>/best)")
    ap.add_argument("--data-dir", required=True, help="dataset folder containing metadata.csv and images/")
    ap.add_argument("--save-preds-dir", required=True, help="destination folder for prediction CSVs and eval_summary.json")
    ap.add_argument("--config", default=os.path.join(HERE, "..", "..", "..", "configs", "ocr_hatformer.yaml"))
    ap.add_argument("--tokenizer", help="override: path to tokenizer.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--num-beams", type=int, default=1, help="beam search width (default 1 = greedy)")
    ap.add_argument("--limit", type=int, help="limit evaluation to N lines (for testing)")
    ap.add_argument("--allow-cpu", action="store_true", help="allow CPU execution if no GPU")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.tokenizer:
        cfg["tokenizer_path"] = args.tokenizer

    device = pick_device(args.allow_cpu)
    use_amp = cfg["train"]["precision"] == "amp_fp16" and device.type == "cuda"

    print(f"[EXPORT] Loading checkpoint: {args.checkpoint} on {device}")
    model, tok = build_model_and_tokenizer(cfg, args.checkpoint, device)

    d = make_datasets(cfg, tok, args.data_dir, [args.split], limit=args.limit)
    df, ds = d[args.split]

    print(f"[EXPORT] Evaluating {len(ds)} lines on split '{args.split}' (beams={args.num_beams})...")
    r = evaluate(
        model, tok, ds, device, cfg, use_amp,
        num_beams=args.num_beams, max_gen_lines=None,
        loss_too=False, books=True, want_samples=3, return_preds=True
    )

    export_prediction_csvs(
        df=df, ds=ds, preds=r["preds"], gen_indices=r["idx"],
        save_preds_dir=args.save_preds_dir, data_dir=args.data_dir,
        split=args.split, num_beams=args.num_beams, ckpt=args.checkpoint,
        trainer_results=r
    )
    print(f"[EXPORT] Done! All files written to: {args.save_preds_dir}")


if __name__ == "__main__":
    main()
