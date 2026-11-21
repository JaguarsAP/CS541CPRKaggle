
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import CPRVideoDataset

train_csv = PROJECT_ROOT / "data" / "train_split.csv"

dataset = CPRVideoDataset(
    csv_path=train_csv,
    split="train",
    num_frames=32,
    image_size=224,
    stride=4
)

print("Dataset size:", len(dataset))

video, labels = dataset[0]

print("Video shape:", video.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

videos, labels = next(iter(loader))

print("\nBatch video shape:", videos.shape)
print("Batch labels shape:", labels.shape)
