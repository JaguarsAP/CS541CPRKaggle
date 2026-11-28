import argparse
import sys
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from cpr_data import data_file
from src.dataset import CPRVideoDataset
from src.model import CPRResNet18


def main(args):

    device = torch.device("cuda")

    assert torch.cuda.is_available(), "GPU required!"

    print("GPU:", torch.cuda.get_device_name(0))

    sample_path = data_file("sample_submission.csv")

    sample = pd.read_csv(
        sample_path,
        dtype={"video_id": str}
    )

    dataset = CPRVideoDataset(
        csv_path=sample_path,
        split="test",
        num_frames=32,
        image_size=224,
        stride=4
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = CPRResNet18(
        pretrained=True,
        freeze_backbone=True
    ).to(device)

    model.eval()

    features = []
    video_ids = []

    print("Test videos:", len(dataset))

    with torch.inference_mode():

        for batch_idx, (videos, ids) in enumerate(loader):

            videos = videos.to(device, non_blocking=True)

            b, t, c, h, w = videos.shape

            flat = videos.reshape(b * t, c, h, w)

            frame_features = []

            for chunk in flat.split(16):

                frame_features.append(
                    model.backbone(chunk)
                )

            frame_features = torch.cat(
                frame_features, dim=0
            )

            video_features = frame_features.reshape(
                b, t, 512
            )

            features.append(video_features.cpu())
            video_ids.extend(ids)

            if (batch_idx + 1) % 25 == 0:

                processed = min(
                    (batch_idx + 1) * args.batch_size,
                    len(dataset)
                )

                print(
                    f"Processed {processed}/{len(dataset)}",
                    flush=True
                )

    features = torch.cat(features, dim=0)

    assert features.shape == (len(sample), 32, 512)
    assert torch.isfinite(features).all()
    assert video_ids == sample["video_id"].tolist()

    output_dir = ROOT / "features"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "test_features.pt"

    torch.save({
        "features": features,
        "video_ids": video_ids
    }, output_path)

    print("Feature shape:", features.shape)
    print("Saved:", output_path)
    print("TEST FEATURE EXTRACTION COMPLETE!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size", type=int, default=2
    )

    parser.add_argument(
        "--num-workers", type=int, default=2
    )

    main(parser.parse_args())
