
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CPRResNet18(nn.Module):
    def __init__(self, num_classes=13, pretrained=True):
        super().__init__()

        # Load a ResNet-18 pretrained on ImageNet.
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Replace the original ImageNet classifier.
        # ResNet-18 normally outputs 1000 ImageNet classes.
        self.backbone.fc = nn.Identity()

        # ResNet-18 produces 512 features per frame.
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        x: [batch_size, num_frames, 3, height, width]

        returns: [batch_size, 13] raw logits
        """

        batch_size, num_frames, channels, height, width = x.shape

        # Combine batch and time dimensions so ResNet
        # can process all frames as ordinary images.
        x = x.reshape(
            batch_size * num_frames,
            channels,
            height,
            width
        )

        # Extract visual features from each frame.
        features = self.backbone(x)

        # Restore the time dimension.
        features = features.reshape(batch_size, num_frames, 512)

        # Average features across the video.
        video_features = features.mean(dim=1)

        # Predict one logit for each CPR error.
        logits = self.classifier(video_features)

        return logits
