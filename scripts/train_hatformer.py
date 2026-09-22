import os
import torch
import gc
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image, ImageOps
import numpy as np

# Phase 1: Environment Initialization and Resource Routing
os.makedirs("/kaggle/tmp/hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/kaggle/tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/kaggle/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/tmp/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Phase 1.3: Patch PyTorch Multiprocessing Semantics
torch.multiprocessing.set_sharing_strategy('file_system')

print("Environment initialized, HF routed to /kaggle/tmp, and PyTorch multiprocessing patched.")

# Phase 2: Data Ingestion and Normalization Protocol
print("Downloading AraMS-28k dataset...")
dataset = load_dataset("SubSpring/arams28k-htr-lines", cache_dir="/kaggle/tmp/hf_cache")

# Phase 4.1: Instantiate the Topology
print("Loading HATFormer model and tokenizer...")
model = VisionEncoderDecoderModel.from_pretrained(
    "Archatext/hatformer-arams28k", 
    cache_dir="/kaggle/tmp/hf_cache"
)
# Ensure tie_encoder_decoder is managed if needed, though usually loaded from config
if hasattr(model.config, "tie_encoder_decoder") and model.config.tie_encoder_decoder:
    print("Encoder-decoder tie is enabled.")

tokenizer = AutoTokenizer.from_pretrained(
    "Archatext/hatformer-arams28k", 
    cache_dir="/kaggle/tmp/hf_cache"
)

# Phase 4.2: Implement the Aspect-Ratio Preserving BlockProcessor
def custom_block_processor(image: Image.Image):
    """
    Standardize height to 64 pixels, flip horizontally, pad width to fit 384x384.
    """
    # 1. Resize height to 64 pixels, preserving aspect ratio
    w, h = image.size
    new_w = int(w * (64 / h))
    image = image.resize((new_w, 64), Image.LANCZOS)
    
    # 2. Horizontal flip to align right-to-left Arabic with left-to-right Transformer
    image = ImageOps.mirror(image)
    
    # Convert to RGB if not already
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_array = np.array(image)
    
    # 3. Pad to 384x384 without squashing
    padded_img = np.zeros((384, 384, 3), dtype=np.uint8)
    # The image height is exactly 64, we will chunk/pad width up to 384.
    # To strictly fit the 384x384 container "from left to right and top to bottom", 
    # we can break the 64-height image into 384-width chunks and stack them vertically.
    # A single chunk max width is 384. We have 6 rows of 64px (6 * 64 = 384).
    # This means we can pack an image up to 384 * 6 = 2304 pixels wide!
    
    current_row = 0
    current_col = 0
    width_remaining = new_w
    src_x = 0
    
    while width_remaining > 0 and current_row < 6:
        chunk_w = min(width_remaining, 384)
        padded_img[current_row*64:(current_row+1)*64, 0:chunk_w, :] = img_array[:, src_x:src_x+chunk_w, :]
        width_remaining -= chunk_w
        src_x += chunk_w
        current_row += 1

    # Normalize image for ViT (typically standard ImageNet mean/std, or just 0-1 range depending on model)
    # Since we don't have the exact mean/std from the paper, we'll normalize to [0, 1] then typical 
    tensor_img = torch.tensor(padded_img).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    tensor_img = (tensor_img - mean) / std
    
    return tensor_img

def preprocess_function(examples):
    pixel_values = []
    for img in examples["image"]:
        pixel_values.append(custom_block_processor(img))
    
    labels = tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True).input_ids
    
    return {"pixel_values": pixel_values, "labels": labels}

print("Preprocessing dataset...")
# Due to memory constraints, process on the fly or in small batches
train_dataset = dataset["train"].map(preprocess_function, batched=True, batch_size=8, remove_columns=dataset["train"].column_names)
val_dataset = dataset["val"].map(preprocess_function, batched=True, batch_size=8, remove_columns=dataset["val"].column_names)
# test_dataset = dataset["test"].map(...)

# Phase 4.3: Configure the Overtraining Loop
training_args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/hatformer_outputs", # Save only final to /kaggle/working
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_steps=500,
    predict_with_generate=True,
    generation_num_beams=3,
    generation_config=model.generation_config,
    save_strategy="epoch",
    eval_strategy="epoch",

    save_total_limit=1, # Keep only the best
    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,
    num_train_epochs=50, # Overtraining strategy
    dataloader_num_workers=2, # Using file_system sharing strategy
)

# Custom metrics for CER overtraining strategy
import jiwer
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # JiWER CER with preprocessing
    transform = jiwer.Compose([jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    
    # Need to apply transform to strings
    pred_str = [transform(s) for s in pred_str]
    label_str = [transform(s) for s in label_str]

    cer = jiwer.cer(label_str, pred_str)
    
    # Manual garbage collection to prevent memory footprint creep
    gc.collect()

    return {"cer": cer}

from transformers import default_data_collator
trainer = Seq2SeqTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
)

print("Starting HATFormer training...")
trainer.train()
print("HATFormer training completed.")
trainer.save_model("/kaggle/working/hatformer_final")
