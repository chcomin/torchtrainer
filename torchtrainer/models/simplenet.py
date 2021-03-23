'''U-Net architecture with residual blocks'''

import torch
import torch.nn.functional as F
from torch import nn
from torch import tensor

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.stride = stride

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        if (inplanes!=planes) or (stride>1):
            # If in and out planes are different, we also need to change the planes of the input
            # If stride is not 1, we need to change the size of the input
            reshape_input = nn.Sequential(
                                    conv1x1(inplanes, planes, stride),
                                    norm_layer(planes),
                            )
            self.reshape_input = reshape_input
        else:
            self.reshape_input = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.reshape_input is not None:
            identity = self.reshape_input(x)

        out += identity
        out = self.relu(out)

        return out


class SimpleNet(nn.Module):
    '''Encoder part of U-Net'''

    def __init__(self, num_channels, num_classes):
        super(SimpleNet, self).__init__()

        #self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU(inplace=True)

        self.resblock1 = ResBlock(num_channels, 64, stride=1)
        self.resblock2 = ResBlock(64, 128, stride=1)
        self.resblock3 = ResBlock(128, 256, stride=1)
        self.resblock4 = ResBlock(256, 512, stride=1)
        self.final = nn.Conv2d(512, num_classes, kernel_size=1)
        self.reset_parameters()

    def forward(self, x):
        for layer in self.children(): x = layer(x)
        return F.log_softmax(x, 1)

    def reset_parameters(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def get_shapes(self, img_shape):

        input_img = torch.zeros(img_shape)[None, None]
        input_img = input_img.to(next(model.parameters()).device)
        output = self(input_img)
        return output[0, 0].shape

class SimpleNet2(nn.Module):
    '''Encoder part of U-Net'''

    def __init__(self, num_channels, num_classes):
        super(SimpleNet2, self).__init__()

        #self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU(inplace=True)

        self.resblock1 = ResBlock(num_channels, 8, stride=1)
        self.resblock2 = ResBlock(8, 16, stride=1)
        self.resblock3 = ResBlock(16, 32, stride=1)
        self.resblock4 = ResBlock(32, 64, stride=1)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.reset_parameters()

    def forward(self, x):
        for layer in self.children(): x = layer(x)
        return F.log_softmax(x, 1)

    def reset_parameters(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def get_shapes(self, img_shape):

        input_img = torch.zeros(img_shape)[None, None]
        input_img = input_img.to(next(model.parameters()).device)
        output = self(input_img)
        return output[0, 0].shape