import os
import jiwer
import unicodedata
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, AutoTokenizer
import torch
import numpy as np
from PIL import Image, ImageOps

# Phase 1: Environment Initialization
os.environ["HF_HOME"] = "/kaggle/tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/tmp/hf_cache"

# Load the dataset
dataset = load_dataset("SubSpring/arams28k-htr-lines", split="test", cache_dir="/kaggle/tmp/hf_cache")

# Initialize HATFormer Model for Evaluation
print("Loading HATFormer model and tokenizer for evaluation...")
# For evaluation, we load from the saved final model (or Hugging Face if just testing zero-shot)
model_path = "/kaggle/working/hatformer_final"
if not os.path.exists(model_path):
    print(f"Warning: {model_path} not found. Falling back to base pretrained model.")
    model_path = "Archatext/hatformer-arams28k"

model = VisionEncoderDecoderModel.from_pretrained(model_path, cache_dir="/kaggle/tmp/hf_cache")
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/kaggle/tmp/hf_cache")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Phase 4.2: BlockProcessor for Evaluation
def custom_block_processor(image: Image.Image):
    w, h = image.size
    new_w = int(w * (64 / h))
    image = image.resize((new_w, 64), Image.LANCZOS)
    image = ImageOps.mirror(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_array = np.array(image)
    
    padded_img = np.zeros((384, 384, 3), dtype=np.uint8)
    current_row, src_x, width_remaining = 0, 0, new_w
    while width_remaining > 0 and current_row < 6:
        chunk_w = min(width_remaining, 384)
        padded_img[current_row*64:(current_row+1)*64, 0:chunk_w, :] = img_array[:, src_x:src_x+chunk_w, :]
        width_remaining -= chunk_w
        src_x += chunk_w
        current_row += 1

    tensor_img = torch.tensor(padded_img).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    tensor_img = (tensor_img - mean) / std
    return tensor_img.unsqueeze(0).to(device)

# Phase 5.2: Normalized JiWER Pipeline
preprocess_text = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

def normalize_text(text):
    text = preprocess_text(text)
    return unicodedata.normalize('NFD', text)

# Phase 5.3: Disaggregate Metrics by Manuscript Script
results = {
    "Maghrebi (book_03)": {"refs": [], "hyps": []},
    "Ruq'ah (book_05)": {"refs": [], "hyps": []},
    "Naskh (book_09)": {"refs": [], "hyps": []},
    "Other": {"refs": [], "hyps": []}
}

print("Running inference and evaluation...")
for item in dataset:
    image = item["image"]
    ref_raw = item["gt_normalized"]
    book_id = item.get("book", "")
    
    pixel_values = custom_block_processor(image)
    
    # Phase 5.1: Configure Autoregressive Generation Parameters
    with torch.no_grad():
        outputs = model.generate(
            pixel_values, 
            num_beams=3, 
            length_penalty=0.5, # Between 0.2 and 0.8
            max_length=128
        )
    
    hyp_raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    ref_norm = normalize_text(ref_raw)
    hyp_norm = normalize_text(hyp_raw)
    
    if "book_03" in book_id:
        script_key = "Maghrebi (book_03)"
    elif "book_05" in book_id:
        script_key = "Ruq'ah (book_05)"
    elif "book_09" in book_id:
        script_key = "Naskh (book_09)"
    else:
        script_key = "Other"
        
    results[script_key]["refs"].append(ref_norm)
    results[script_key]["hyps"].append(hyp_norm)

print("Evaluation Results:")
for script, data in results.items():
    if len(data["refs"]) > 0:
        cer = jiwer.cer(data["refs"], data["hyps"])
        print(f"[{script}] CER: {cer:.2%}")
