"""Trivial models that can be used for testing purposes."""
from torch import nn


class TestClassification(nn.Module):
    """Classification model for testing purposes."""

    def __init__(self, num_channels=1, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_classes, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        return self.pool(self.conv(x))

class TestSegmentation(nn.Module):
    """Segmentation model for testing purposes."""

    def __init__(self, num_channels=1, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_classes, 3, padding=1)
    
    def forward(self, x):
        return self.conv(x)