import torch
import torch.nn as nn


class LinearClassifier(nn.Module):

    def __init__(self, num_classes=13):
        super().__init__()

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)
        return self.classifier(x)


class MLPClassifier(nn.Module):

    def __init__(self, num_classes=13):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.mean(dim=1)
        return self.classifier(x)


class StatisticalClassifier(nn.Module):

    def __init__(self, num_classes=13):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        mean = x.mean(dim=1)
        maximum = x.max(dim=1).values
        std = x.std(dim=1, unbiased=False)

        x = torch.cat([mean, maximum, std], dim=1)

        return self.classifier(x)


class LSTMClassifier(nn.Module):

    def __init__(self, num_classes=13):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        x = output.mean(dim=1)

        return self.classifier(x)


class TemporalCNNClassifier(nn.Module):

    def __init__(self, num_classes=13):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        # [B, T, 512] -> [B, 512, T]
        x = x.transpose(1, 2)

        x = self.temporal(x)

        mean = x.mean(dim=2)
        maximum = x.max(dim=2).values

        x = torch.cat([mean, maximum], dim=1)

        return self.classifier(x)


def build_classifier(name):

    models = {
        "linear": LinearClassifier,
        "mlp": MLPClassifier,
        "statistical": StatisticalClassifier,
        "lstm": LSTMClassifier,
        "temporal_cnn": TemporalCNNClassifier
    }

    return models[name]()
