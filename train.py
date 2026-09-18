import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import AraMSDataset
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

def collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    
    labels = processor.tokenizer(text=texts, padding=True, return_tensors="pt").input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    return {"pixel_values": pixel_values, "labels": labels}

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data" / "AraMS-28k-HTR"
    
    train_dataset = AraMSDataset(data_dir=data_dir, split="train")
    
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    batch = next(iter(dataloader))
    
    print("pixel_values shape:", batch["pixel_values"].shape)
    print("labels shape:", batch["labels"].shape)
