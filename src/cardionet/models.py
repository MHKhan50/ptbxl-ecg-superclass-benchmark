"""Compact 1D neural network architectures for ECG sequences."""
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.block(x) + self.skip(x))

class ECGResNet(nn.Module):
    def __init__(self, num_labels=5, channels=(32, 64, 128), dropout=0.2):
        super().__init__()
        layers = []
        in_ch = 12
        for i, ch in enumerate(channels):
            layers.append(ConvBlock(in_ch, ch, stride=1 if i == 0 else 2))
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_labels))
    def forward(self, x):
        return self.head(self.pool(self.features(x)))

class ECGCNN(nn.Module):
    def __init__(self, num_labels=5, channels=(32, 64, 128), dropout=0.2):
        super().__init__()
        blocks = []
        in_ch = 12
        for ch in channels:
            blocks += [nn.Conv1d(in_ch, ch, kernel_size=7, padding=3), nn.BatchNorm1d(ch),
                       nn.ReLU(inplace=True), nn.MaxPool1d(2)]
            in_ch = ch
        self.net = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                 nn.Dropout(dropout), nn.Linear(in_ch, num_labels))
    def forward(self, x):
        return self.net(x)

def build_model(name: str, **kwargs):
    if name == "cnn":
        return ECGCNN(**kwargs)
    if name == "resnet":
        return ECGResNet(**kwargs)
    raise ValueError(f"Unknown model: {name}")
