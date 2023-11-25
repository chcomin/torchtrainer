'''U-Net architecture with residual blocks'''

import torch
import torch.nn.functional as F
from torch import nn
from .layers import BasicBlockOld, conv3x3, conv1x1

class ResNetSeg(nn.Module):
    """ResNetModel for segmentation. Model adapted from Pytorch."""

    def __init__(self, layers, inplanes, in_channels=1, num_classes=2, zero_init_residual=False):
        super().__init__()

        self.norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, inplanes[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = self.norm_layer(inplanes[0], momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        stages = [self._make_layer(inplanes[0], inplanes[0], layers[0])]
        for idx in range(len(layers)-1):
            stages.append(self._make_layer(inplanes[idx], inplanes[idx+1], layers[idx+1]))

        self.stages = nn.Sequential(*stages)
        self.conv_output = conv3x3(inplanes[-1], num_classes)

        self._init_parameters(zero_init_residual)

    def _make_layer(self, inplanes, planes, blocks):

        downsample = None
        norm_layer = self.norm_layer
        block = BasicBlockOld

        if inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes),
                norm_layer(planes, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)
    
    def _init_parameters(self, zero_init_residual):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockOld):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stages(x)
        x = self.conv_output(x)

        return x