import os
import subprocess
import glob
from datasets import load_dataset

# Configuration
model_path = "/kaggle/working/kraken_best.mlmodel" # Assumes kraken_best.mlmodel exists after training
cache_dir = "/kaggle/tmp/hf_cache"
test_dir = "/kaggle/tmp/data/test"

if not os.path.exists(model_path):
    print(f"Warning: Model not found at {model_path}. You may need to provide the correct path.")

os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir

print("Downloading AraMS-28k test dataset...")
dataset = load_dataset("SubSpring/arams28k-htr-lines", split="test", cache_dir=cache_dir)
dataset = dataset.filter(lambda x: x.get("line_type", "main") != "margin")

os.makedirs(test_dir, exist_ok=True)

books = {
    "Maghrebi (book_03)": "book_03",
    "Ruq'ah (book_05)": "book_05",
    "Naskh (book_09)": "book_09"
}

# Write images to disk per book
manifests = {book_id: [] for book_id in books.values()}

print("Extracting images and transcripts for evaluation...")
for i, item in enumerate(dataset):
    book_id = item["book_id"]
    if book_id not in manifests:
        continue
        
    text = item["text"]
    image = item["image"]
    
    base_name = f"{book_id}_line_{i:06d}"
    img_path = os.path.join(test_dir, f"{base_name}.png")
    gt_path = os.path.join(test_dir, f"{base_name}.gt.txt")
    
    image.save(img_path)
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    manifests[book_id].append(img_path)

print("\nStarting Kraken evaluation...")

for book_name, book_prefix in books.items():
    images = manifests[book_prefix]
    
    temp_manifest = f"temp_manifest_{book_prefix}.txt"
    with open(temp_manifest, "w", encoding="utf-8") as f:
        for img in images:
            f.write(img + "\n")
            
    print(f"\nEvaluating {book_name} ({len(images)} lines)...")
    
    # Run ketos test
    cmd = [
        "ketos", "test",
        "-m", model_path,
        "-e", temp_manifest,
        "-f", "path",
        "-u", "NFD"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout + "\n" + result.stderr)
    
    if os.path.exists(temp_manifest):
        os.remove(temp_manifest)

print("Evaluation complete.")
