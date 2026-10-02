import os
import subprocess

manifest_path = "data/AraMS-28k-HTR/test_manifest.txt"
model_path = "kraken.mlmodel"

with open(manifest_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

books = {
    "Maghrebi (book_03)": "book_03",
    "Ruq'ah (book_05)": "book_05",
    "Naskh (book_09)": "book_09"
}

print(f"Starting Kraken evaluation...")

for book_name, book_prefix in books.items():
    # Filter lines for this book
    book_lines = [line for line in lines if f"images/{book_prefix}_" in line]
    
    # Adjust paths to be correct when running from the root directory
    adjusted_lines = [line.replace("./AraMs-28k-HTR", "data/AraMS-28k-HTR").replace("./AraMS-28k-HTR", "data/AraMS-28k-HTR") for line in book_lines]
    
    temp_manifest = f"temp_manifest_{book_prefix}.txt"
    with open(temp_manifest, "w", encoding="utf-8") as f:
        f.writelines(adjusted_lines)
    
    print(f"\nEvaluating {book_name} ({len(adjusted_lines)} lines)...")
    
    # Run ketos test
    cmd = [
        "ketos", "test",
        "-m", model_path,
        "-e", temp_manifest,
        "-f", "path",
        "-u", "NFD"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Output parsing
    output = result.stdout + "\n" + result.stderr
    print(output)
    
    # Clean up
    if os.path.exists(temp_manifest):
        os.remove(temp_manifest)

print("Evaluation complete.")
