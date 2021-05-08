"""U-Net architecture with residual blocks"""

import torch
import torch.nn.functional as F
from torch import nn
from torch import tensor
from .layers import ResBlock, Concat, Blur
from ..module_util import ActivationSampler
from collections import OrderedDict
from ..module_util import get_submodule

class Encoder(nn.Module):

    def __init__(self, num_channels, reduce_by=1):
        """Encoder part of U-Net"""

        super(Encoder, self).__init__()

        #num_planes = 64
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.resblock1 = ResBlock(64, 64, stride=1)
        self.resblock2 = ResBlock(64, 128, stride=2)
        self.resblock3 = ResBlock(128, 256, stride=2)
        self.resblock4 = ResBlock(256, 512, stride=2)
        self.resblock_mid = ResBlock(512, 1024, stride=2)

    def forward(self, x):
        for layer in self.children(): x = layer(x)
        return x

class ResUNet(nn.Module):

    def __init__(self, num_channels, num_classes):
        """U-Net with residual blocks."""

        super(ResUNet, self).__init__()

        self.encoder = Encoder(num_channels)

        self.a_mid_up = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.blur_mid_up = Blur()
        self.sample_a4_ = ActivationSampler(self.encoder.resblock4)
        self.concat_a4 = Concat(1)
        self._l4 = ResBlock(1024, 512, stride=1)

        self.a4_up = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.blur_a4_up = Blur()
        self.sample_a3_ = ActivationSampler(self.encoder.resblock3)
        self.concat_a3 = Concat(1)
        self._l3 = ResBlock(512, 256, stride=1)

        self.a3_up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.blur_a3_up = Blur()
        self.sample_a2_ = ActivationSampler(self.encoder.resblock2)
        self.concat_a2 = Concat(1)
        self._l2 = ResBlock(256, 128, stride=1)

        self.a2_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.blur_a2_up = Blur()
        self.sample_a1_ = ActivationSampler(self.encoder.resblock1)
        self.concat_a1 = Concat(1)
        self._l1 = ResBlock(128, 64, stride=1)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.reset_parameters()

    def forward(self, x):
        a_mid = self.encoder(x)

        a_mid_up = self.a_mid_up(a_mid)
        a_mid_up = self.blur_mid_up(a_mid_up)
        a4_ = self.sample_a4_()
        _a4 = self._l4(self.concat_a4(a4_, a_mid_up))

        a4_up = self.a4_up(_a4)
        a4_up = self.blur_a4_up(a4_up)
        a3_ = self.sample_a3_()
        _a3 = self._l3(self.concat_a3(a3_, a4_up))

        a3_up = self.a3_up(_a3)
        a3_up = self.blur_a3_up(a3_up)
        a2_ = self.sample_a2_()
        _a2 = self._l2(self.concat_a2(a2_, a3_up))

        a2_up = self.a2_up(_a2)
        a2_up = self.blur_a2_up(a2_up)
        a1_ = self.sample_a1_()
        _a1 = self._l1(self.concat_a1(a1_, a2_up))

        final = self.final(_a1)

        return final

    def reset_parameters(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class FlexibleEncoder(nn.Module):

    def __init__(self, num_channels, layers, first_kernel_size=7):
        """Encoder part of FlexibleResUNet"""

        super(FlexibleEncoder, self).__init__()

        self.input_conv = nn.Sequential(
                nn.Conv2d(num_channels, layers[0], kernel_size=first_kernel_size, stride=1, padding=(first_kernel_size-1)//2, bias=False),
                nn.BatchNorm2d(layers[0]),
                nn.ReLU(inplace=True)
        )

        if len(layers)>1:
            self.initial_block = ResBlock(layers[0], layers[1], stride=1)
        for layer_idx in range(1, len(layers)-1):
            self.add_module(f'down_{layer_idx}', ResBlock(layers[layer_idx], layers[layer_idx+1], stride=2))

    def forward(self, x):
        for layer in self.children(): x = layer(x)
        return x

class FlexibleResUNet(nn.Module):

    def __init__(self, num_channels, num_classes, layers=(64, 64, 128, 256, 512, 1024), use_blur=True):
        """U-Net with residual blocks that adapts the number of layers and number of filters
        according to parameter `layers`. The first two values of `layers` set the number
        of filters of an initial Conv2d and an initial residual block. The remaining values
        set the number of filters at each downsample operation. Consequently, it also sets
        the number of filters at each upsample operation. Thus, the number of downsamples
        is equal to len(layers)-2."""

        super().__init__()
        self.layers = layers
        self.use_blur = use_blur

        self.encoder = FlexibleEncoder(num_channels, layers)
        for layer_idx in range(len(layers)-1, 1, -1):
            upsample_block = OrderedDict()
            upsample_block[f'upsample'] = nn.ConvTranspose2d(layers[layer_idx], layers[layer_idx-1], kernel_size=2, stride=2)
            if use_blur:
                upsample_block[f'blur'] = Blur()
            if layer_idx==2:
                upsample_block['sample_initial_block_act'] = ActivationSampler(get_submodule(self, "encoder.initial_block"))
            else:
                upsample_block[f'sample_down_{layer_idx-2}_act'] = ActivationSampler(get_submodule(self, f"encoder.down_{layer_idx-2}"))
            upsample_block[f'concat'] = Concat(1)
            upsample_block[f'squeeze_channels'] = ResBlock(layers[layer_idx], layers[layer_idx-1], stride=1)
            self.add_module(f'up_{layer_idx-1}', nn.Sequential(upsample_block))

        self.final = nn.Conv2d(layers[1], num_classes, kernel_size=1)
        self.reset_parameters()

    def forward(self, x):

        layers = self.layers
        
        x = self.encoder(x)
        for layer_idx in range(len(layers)-1, 1, -1):
            upsample_block = get_submodule(self, f'up_{layer_idx-1}')
            x = upsample_block.upsample(x)
            if self.use_blur:
                x = upsample_block.blur(x)
            if layer_idx==2:
                act = upsample_block.sample_initial_block_act()
            else:
                act = get_submodule(upsample_block, f"sample_down_{layer_idx-2}_act")()
            x = upsample_block.concat(act, x)
            x = upsample_block.squeeze_channels(x)
        x = self.final(x)

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