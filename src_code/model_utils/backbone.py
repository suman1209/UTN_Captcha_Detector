import torch
import torch.nn as nn
from torchvision import models

class VGG16Backbone(nn.Module):
    """VGG16 backbone for SSD model (modified for grayscale input)."""
    def __init__(self, pretrained=True, in_channels=1):
        super().__init__()

        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Modify first conv layer for grayscale (1 channel)
        self.features = vgg16.features[:30]
        if in_channels == 1:
            self.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        # Additional SSD layers (conv6, conv7)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # VGG16 base features
        conv4_3 = self.features(x)
        
        # SSD-specific additional layers
        conv6_out = self.relu(self.conv6(conv4_3))
        conv7_out = self.relu(self.conv7(conv6_out))
        
        return [conv4_3, conv7_out]
