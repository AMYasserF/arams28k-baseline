from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class AraMSDataset(Dataset):
    """PyTorch Dataset for the AraMS-28k-HTR handwritten text recognition data."""

    SPLIT_TO_MANIFEST = {
        "train": "train_manifest.txt",
        "val": "val_manifest.txt",
        "test": "test_manifest.txt",
    }

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        if split not in self.SPLIT_TO_MANIFEST:
            raise ValueError(f"Unsupported split '{split}'. Expected one of: {sorted(self.SPLIT_TO_MANIFEST)}")

        self.images_dir = self.data_dir / "images"
        manifest_path = self.data_dir / self.SPLIT_TO_MANIFEST[split]

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            self.sample_names = [self._normalize_sample_name(line) for line in handle if line.strip()]

    @staticmethod
    def _normalize_sample_name(line: str) -> str:
        """Convert a manifest entry to the base sample name."""
        sample_path = Path(line.strip())
        sample_name = sample_path.name

        if sample_name.endswith(".gt.txt"):
            return sample_name[: -len(".gt.txt")]
        if sample_name.endswith(".png"):
            return sample_name[: -len(".png")]
        return sample_name

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int) -> Tuple[object, str]:
        sample_name = self.sample_names[index]

        image_path = self.images_dir / f"{sample_name}.png"
        text_path = self.images_dir / f"{sample_name}.gt.txt"

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        with text_path.open("r", encoding="utf-8") as handle:
            text = handle.read().strip()

        return image, text


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data" / "AraMS-28k-HTR"
    dataset = AraMSDataset(data_dir=data_dir, split="train")
    print(f"Dataset length: {len(dataset)}")

    image, text = dataset[0]
    print(f"First image size: {image.size}")
    print(f"First text: {text}")
