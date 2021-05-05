'''U-Net architecture with residual blocks'''

import torch
import torch.nn.functional as F
from torch import nn
from torch import tensor
from .layers import ResBlock, conv3x3, conv1x1


class SimpleNet(nn.Module):

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
        input_img = input_img.to(next(self.parameters()).device)
        output = self(input_img)
        return output[0, 0].shape

class FlexibleSimpleNet(nn.Module):

    def __init__(self, num_channels, num_classes, layers=[8, 16, 32, 64], input_conv=True):
        super().__init__()

        if input_conv:
            channels_first_layer = layers[0]
            self.input_conv = nn.Conv2d(num_channels, layers[0], kernel_size=7, stride=1, padding=3, bias=False)
            self.input_conv_bn = nn.BatchNorm2d(layers[0])
            self.input_conv_relu = nn.ReLU(inplace=True)
        else:
            channels_first_layer = num_channels

        channels_prev_layer = channels_first_layer
        for idx, channels_curr_layer in enumerate(layers):
            setattr(self, f'resblock{idx+1}', ResBlock(channels_prev_layer, channels_curr_layer, stride=1))
            channels_prev_layer = channels_curr_layer
            
        self.output_conv = nn.Conv2d(layers[-1], num_classes, kernel_size=1)
        self.reset_parameters()

    def forward(self, x):
        for layer in self.children(): x = layer(x)
        return x

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
        input_img = input_img.to(next(self.parameters()).device)
        output = self(input_img)
        return output[0, 0].shape