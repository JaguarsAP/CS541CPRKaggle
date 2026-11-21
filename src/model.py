
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CPRResNet18(nn.Module):

    def __init__(
        self,
        num_classes=13,
        pretrained=True,
        freeze_backbone=True
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

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
