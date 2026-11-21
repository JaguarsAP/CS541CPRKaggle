
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import CPRResNet18


def main():
    # No pretrained weights needed for this shape test.
    model = CPRResNet18(pretrained=False)
    model.eval()

    # Use 2 frames and 64x64 images for a quick CPU test.
    # The model supports our full 32-frame, 224x224 inputs too.
    dummy_video = torch.randn(1, 2, 3, 64, 64)

    with torch.no_grad():
        logits = model(dummy_video)
        probabilities = torch.sigmoid(logits)

    print("Input shape:", dummy_video.shape)
    print("Output shape:", logits.shape)
    print("Probabilities:", probabilities)

    assert logits.shape == (1, 13)
    print("Model test passed!")


if __name__ == "__main__":
    main()
