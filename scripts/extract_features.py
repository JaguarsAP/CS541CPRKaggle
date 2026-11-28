
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import CPRVideoDataset
from src.model import CPRResNet18


def extract_features(split, batch_size, num_workers):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "Feature extraction requires a GPU. "
            "Run this through an SCC GPU job."
        )

    print("Device:", device, flush=True)

    csv_path = ROOT / "data" / f"{split}_split.csv"

    dataset = CPRVideoDataset(
        csv_path=csv_path,
        split="train",
        num_frames=32,
        image_size=224,
        stride=4,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = CPRResNet18(
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    model.eval()

    features = []
    labels = []

    with torch.inference_mode():
        for batch_idx, (videos, targets) in enumerate(loader):
            videos = videos.to(device, non_blocking=True)

            # videos: [B, T, 3, H, W]
            b, t, c, h, w = videos.shape

            # Process frames in small chunks to limit GPU memory.
            flat = videos.reshape(b * t, c, h, w)

            frame_features = []

            for chunk in flat.split(16):
                frame_features.append(model.backbone(chunk))

            frame_features = torch.cat(frame_features, dim=0)

            # Preserve temporal features for each video.
            video_features = frame_features.reshape(
                b, t, 512
            )

            features.append(video_features.cpu())
            labels.append(targets.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"{split}: processed "
                    f"{min((batch_idx + 1) * batch_size, len(dataset))}"
                    f"/{len(dataset)} videos",
                    flush=True,
                )

    output_dir = ROOT / "features"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{split}_features.pt"

    torch.save(
        {
            "features": torch.cat(features),
            "labels": torch.cat(labels),
        },
        output_path,
    )

    print("Saved:", output_path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["train", "val"], required=True
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    extract_features(
        args.split,
        args.batch_size,
        args.num_workers,
    )
