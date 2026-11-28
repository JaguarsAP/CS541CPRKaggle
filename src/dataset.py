
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports from the instructor's starter folder
sys.path.insert(0, str(PROJECT_ROOT / "starter"))

from cpr_data import load_video, video_file


class CPRVideoDataset(Dataset):
    """
    PyTorch Dataset for CPR technique error detection.

    Each example contains:
        video: Tensor [T, 3, H, W]
        labels: Tensor [13]
    """

    def __init__(
        self,
        csv_path,
        split="train",
        num_frames=32,
        image_size=224,
        stride=4
    ):
        self.df = pd.read_csv(csv_path, dtype={"video_id": str})
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.stride = stride

        # Identify the 13 error columns
        self.error_cols = [
            col for col in self.df.columns
            if col.startswith("e") and col[1:3].isdigit()
        ]

        # ImageNet normalization statistics
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row["video_id"]

        # Load sampled RGB frames
        path = video_file(video_id, self.split)

        frames = load_video(
            path,
            stride=self.stride,
            max_frames=self.num_frames,
            size=(self.image_size, self.image_size)
        )

        # If a video is too short, repeat its final frame
        if len(frames) < self.num_frames:
            missing = self.num_frames - len(frames)

            padding = np.repeat(
                frames[-1:],
                missing,
                axis=0
            )

            frames = np.concatenate(
                [frames, padding],
                axis=0
            )

        # Convert RGB uint8 frames to float tensors
        frames = torch.from_numpy(frames.copy()).float() / 255.0

        # Convert [T, H, W, C] to [T, C, H, W]
        frames = frames.permute(0, 3, 1, 2)

        # Normalize for pretrained ImageNet models
        frames = (frames - self.mean) / self.std

        # Test videos have no ground-truth labels.
        if self.split == "test":
            return frames, str(video_id)

        # Get the 13 binary labels
        labels = torch.tensor(
            row[self.error_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32
        )

        return frames, labels
