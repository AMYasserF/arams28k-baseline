# Running HATFormer on Kaggle — step-by-step guide

This guide covers (1) why your first run failed, (2) exactly what to click, (3) how to resume, (4) troubleshooting.

---

## 1. What went wrong in the first run (from your log)

**There was no Python error.** The run "succeeded" in 9,300 s and produced a useless model:

| Evidence in the log | Meaning |
|---|---|
| `CER 1.0000` at *every* validation (step 1000 … 19000) and every `pred:` line is empty | the model emits `</s>` immediately; CER is exactly 1.0 when every prediction is empty |
| train loss goes 3.5 (step 50) → 7.3 (step 200) and never comes back; val loss 8.3 → 12.5 → 20.8 | the weights were being *destroyed* as the LR warmed up |
| all 400 logged loss values lie on the **float16 grid** (e.g. 5.2070 = 1333/256, 7.2734 = 1862/256); only 13 % would fall on a bfloat16 grid (chance level) | the model was computing and training in **pure fp16** |
| progress bars `Loading weights … Materializing param=` | `transformers` v5, which loads a checkpoint in the dtype it was *saved* in (fp16) instead of fp32 |
| peak LR in the log = `9.99e-06` at step 500 | that run used 1e-5, **not the paper's 5e-5** (it was a different config than the one in the zip) |

**Root cause.** v5 silently loaded the HATFormer weights as float16. The old loop then ran `AdamW(eps=1e-6)` directly on fp16 weights with no mixed-precision machinery. In fp16, squared gradients underflow to 0 (values < 6e-8), so Adam divides by ≈ε and takes enormous steps, while genuinely small updates are rounded away. Training diverges exactly when the LR ramps up, and the decoder collapses to "emit end-of-sequence".

**What the new trainer does differently**
* forces **fp32 master weights** (`assert` that no parameter is fp16) and uses fp16 **autocast + GradScaler** for speed — the standard, numerically safe recipe;
* **fails loudly**: GPU self-test, `--preflight` (loss of the pretrained init, sample predictions), special-token-id checks, image-path checks, a warning when > 90 % of validation predictions are empty, and abort on 25 consecutive NaN losses;
* labels always end in `</s>`; generation config ids are set explicitly;
* paper hyper-parameters (below); **LR / loss / accuracy / CER curves** are written during training.

Non-problems you saw: `generation flags … length_penalty` (harmless; greedy decoding ignores it) and the `SyntaxWarning`s from nbconvert at the end.

**Timing from your log:** setup ≈ 17 min (downloads), training ≈ 2.3 h for 20 000 steps (0.41 s/step including validation). So the full paper recipe normally finishes inside **one** 12-hour session. Resume support is insurance (slower GPU, interruptions, extra steps).

---

## 2. Hyper-parameters (paper Appendix B.1 → `configs/ocr_hatformer.yaml`)

| | value |
|---|---|
| optimizer | AdamW, β=(0.9, 0.999), ε=1e-6, weight-decay 0.01 |
| LR | **5e-5**, linear warm-up 500 steps |
| batch / clip / steps | 8 / 5.0 / 20 000 |
| checkpoint + validation | every 1 000 steps, **best checkpoint chosen by validation CER, greedy decoding** |
| init | Muharaf-pretrained HATFormer (`hatformer-muharaf`), 50 265-token Arabic BBPE tokenizer |
| input | height 64, RTL flip, 384×384 canvas, `max_text_length` 64, target = `gt_normalized` |
| *not in the paper* | decay after warm-up = **inverse-sqrt** (same as the repo/original HATFormer code), CER on 256 evenly spaced val lines per validation, fp16 autocast |

---

## 3. One-time preparation (on your PC)

1. Copy these files from the fixed package into your clone of `AMYasserF/arams28k-baseline`, keeping the same paths:
   `scripts/models/hatformer/{train_hatformer.py,hat_utils.py,dataset.py}`, `configs/ocr_hatformer.yaml`, `requirements.txt`, `README.md`, `kaggle/hatformer_kaggle.ipynb`, `docs/KAGGLE_RUN_GUIDE.md`.
2. `git add -A && git commit -m "Resumable Kaggle HATFormer trainer" && git push origin main`
3. **Downloads are pre-filled and verified** (both are public Zenodo records, no account needed): the dataset `https://zenodo.org/records/21499649` (AraMS-28k-HTR.zip, 2.4 GB) and the official HATFormer release `https://zenodo.org/records/14165756` (HATFormer.zip, 1.3 GB; it is the 'Code Link' in the HATFormer paper). The notebook looks inside the HATFormer zip for the folder whose path contains `muharaf`; if the folder names differ it prints every model folder it found, and you set `INIT_PATH_OVERRIDE`. The tokenizer is taken from the zip if present, otherwise from `Archatext/hatformer-arams28k`.
   *Optional, faster:* upload `AraMS-28k-HTR`, `HATFormer` and `tokenizer.json` once as **Kaggle Datasets** and attach them as inputs — the notebook auto-detects them under `/kaggle/input` and skips the downloads.

## 4. Kaggle set-up

1. Kaggle account must be **phone-verified** (needed for GPU + Internet).
2. *Create → New Notebook → File → Import Notebook* → upload `kaggle/hatformer_kaggle.ipynb` (or paste its cells).
3. Right sidebar → **Session options**
   * Accelerator: **GPU T4 x2** (the code uses one GPU; avoid P100 — recent PyTorch builds may lack its kernels)
   * **Internet: On**
4. In cell 1 paste `HATFORMER_ZIP_URL` and `TOKENIZER_URL`. Leave `SMOKE_TEST = True`.

## 5. Smoke test (≈ 5–8 min, interactive)

Run all cells (*Run → Run all*). It will:
* print the GPU name and run a matmul on it (stops with a clear message if there is no GPU);
* clone your repo, download/unzip data (≈ 10 min the first time), find the pretrained folder and tokenizer;
* **preflight** — look for: `fp32 master weights`, `train: 19739 | val: 1486` line counts, a `[TOKENS]` line that ends with the `</s>` id, the pretrained model's loss on 8 lines and a few sample predictions;
* train 30 steps, **stop**, start a *second process* that must print `[RESUME] … continuing from step 30`, and finish at step 60;
* print `SMOKE TEST PASSED` and show the four curves.

If anything fails, fix it here — it costs minutes, not hours.

## 6. The real run (background, survives closing the browser)

1. Set `SMOKE_TEST = False` in cell 1.
2. **Save Version → Save & Run All (Commit)** → give it a name → Save. (If the dialog offers "save output even if the run fails", enable it.)
3. Close the tab. Kaggle runs it in the background for up to 12 h.
4. Watch it: notebook page → **Versions / View Active Events** (bottom-left) → click the running version → **Logs**.

**What healthy logs look like**
```
[GPU] Tesla T4 | 14.7 GiB ...          [MODEL] ... fp32 master weights ...
[PLAN] 19739 train lines | batch 8 x accum 1 | 2468 batches/epoch | steps 0 -> 20000 | lr 5.0e-05 ...
  step     50/20000 | loss 3.9 | tok-acc  .. | lr 5.00e-06 | 0.4s/step | ETA 2.3h
  [val] step 1000 | loss ... | CER xx% | ...   pred: <Arabic text, not blank>
  [BEST] new best CER ...        [CKPT] step 1000 saved (periodic) in 20s
```
Sanity rules of thumb: the train loss at step ~1000 must be **lower** than at step 50 (in the failed run it was higher); validation predictions must be Arabic text; CER must be < 100 % and fall over the first few validations. If at step 3000 the CER is still ≈ 100 %, cancel the run (saves GPU quota) and send me the log.

Steps/time: ≈ 0.3–0.5 s/step → 20 000 steps ≈ 2–3 h, plus ~1 min per validation.

## 7. Resuming (after a stop, or to continue a run)

The trainer writes `ckpt_last.pt` (model + AdamW + LR-schedule + grad-scaler + RNG + exact data position + full metric history) every 1 000 steps **and** when the time budget (11 h of session time) is reached, and sets `status.json` → `"finished": false`. Resume:

1. Wait until the version shows **Complete**; check the last cell's `STATUS:` line.
2. Open the notebook → right sidebar → **Input → + Add Input → Notebooks / Your Work →** pick *this* notebook (latest version output).
   Alternative: Version → **Output** tab → **New Dataset**, then attach that dataset.
3. Keep `SMOKE_TEST = False` → **Save & Run All (Commit)** again.
4. The log must show `[RESUME] loading …/ckpt_last.pt` and `continuing from step N`. Curves continue seamlessly (history is stored in the checkpoint).

The resume is exact: same batches in the same order, same LR, same optimizer state (verified by simulation for every cut point incl. epoch boundaries).
Kaggle GPU quota is ≈ 30 h/week (see your profile); a full run uses ~3 h.

## 8. Results

* `hatformer_run/plots/` → `training_curves.png`, `lr_curve.png`, `loss_curve.png`, `accuracy_curve.png` (also shown in the notebook), `metrics.csv`, `history.json`
* `hatformer_run/best/` → best model by validation CER (HF format, fp32) — download from the version's **Output** tab
* last cell prints per-book test CER next to the paper's HATFormer numbers: Maghrebi 37.88 · Ruq'ah 13.26 · Naskh 25.37 · overall 26.74 (greedy and beam-3; results saved to `test_results_beams*.json`)

Accuracy curves: *char accuracy* = 1 − CER, *exact-line accuracy* = whole line identical, *token accuracy* = teacher-forced next-token accuracy (train and validation).

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `NO GPU!` assertion | Session options → Accelerator → GPU T4 x2 → restart session |
| `no kernel image is available` | you are on a P100 with a new PyTorch → switch to T4 x2 |
| `CUDA out of memory` | `EXTRA_ARGS = "--batch-size 4 --grad-accum 2"` (same effective batch 8) |
| `git clone` / `wget` fails | Internet toggle is off, or phone not verified |
| `FileNotFoundError: N line images not found` | zip not fully extracted, or wrong folder; delete `/kaggle/temp/data` and rerun cell 4 |
| `tokenizer special ids … do not fit` | wrong `tokenizer.json` (must be the 50 265-token HATFormer BBPE) |
| `Pretrained HATFormer not found` | set `HATFORMER_ZIP_URL`, attach it as a Dataset, or set `INIT_PATH_OVERRIDE` |
| `non-finite loss` warnings | harmless if rare (GradScaler skips the step); persistent → `EXTRA_ARGS = "--precision fp32"` |
| output too large (> 20 GB) | delete `ckpt_last.pt` from finished runs, keep `best/` |

`scripts/train_hatformer.py` (the HF-datasets script) is legacy and not used by the notebook.
