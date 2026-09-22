import os
import subprocess
from datasets import load_dataset
from PIL import Image
import glob
import shutil

# Phase 1: Environment Initialization and Resource Routing
os.makedirs("/kaggle/tmp/hf_cache", exist_ok=True)
os.makedirs("/kaggle/tmp/models", exist_ok=True)
os.makedirs("/kaggle/tmp/data", exist_ok=True)

os.environ["HF_HOME"] = "/kaggle/tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/tmp/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

print("Environment initialized and HF routed to /kaggle/tmp")

# Phase 3.1 & 3.2: Fetch and Stage Pre-trained Weights
print("Fetching pretrained Muharaf model directly from Zenodo...")
model_path = "/kaggle/tmp/models/muharaf_rec_best.mlmodel"
subprocess.run(["wget", "-O", model_path, "https://zenodo.org/records/14295489/files/muharaf_rec_best.mlmodel?download=1"], check=True)
print(f"Model staged at {model_path}")

# Phase 2: Data Ingestion and Normalization Protocol
print("Downloading AraMS-28k dataset...")
dataset = load_dataset("SubSpring/arams28k-htr-lines", cache_dir="/kaggle/tmp/hf_cache")

# Prepare data for ketos compile -f path
train_dir = "/kaggle/tmp/data/train"
val_dir = "/kaggle/tmp/data/val"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def prepare_split(split_name, output_dir):
    print(f"Preparing {split_name} split...")
    split_data = dataset[split_name]
    image_paths = []
    for i, item in enumerate(split_data):
        # Target Transcription Isolation: use text column which contains the normalized text
        text = item["text"]
        image = item["image"]
        
        base_name = f"line_{i:06d}"
        img_path = os.path.join(output_dir, f"{base_name}.png")
        gt_path = os.path.join(output_dir, f"{base_name}.gt.txt")
        
        image.save(img_path)
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        image_paths.append(img_path)
    return image_paths

print("Writing images and text to disk for Kraken compilation...")
train_images = prepare_split("train", train_dir)
val_images = prepare_split("val", val_dir)

# Phase 3.3: Binary Dataset Compilation
print("Compiling Arrow binaries...")
subprocess.run(["ketos", "compile", "-f", "path", "-o", "/kaggle/tmp/data/arams_train.arrow"] + train_images, check=True)
subprocess.run(["ketos", "compile", "-f", "path", "-o", "/kaggle/tmp/data/arams_val.arrow"] + val_images, check=True)

# Phase 3.4: Execute CRNN Fine-tuning
print("Launching ketos train...")
train_cmd = [
    "ketos", "train",
    "--device", "cuda:0",
    "--precision", "bf16-mixed",
    "-B", "32",
    "--resize", "new",
    "-i", model_path,
    "-t", "/kaggle/tmp/data/arams_train.arrow",
    "-e", "/kaggle/tmp/data/arams_val.arrow"
]
print("Running command:", " ".join(train_cmd))
subprocess.run(train_cmd, check=True)
print("Kraken training completed.")
