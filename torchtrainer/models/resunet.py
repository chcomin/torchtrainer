'''U-Net architecture with residual blocks'''

import torch
import torch.nn.functional as F
from torch import nn
from torch import tensor
from .layers import ResBlock, Concat, Blur

# For importing in both the notebook and in the .py file
try:
    import ActivationSampler
except ImportError:
    from torchtrainer.module_util import ActivationSampler

class Encoder(nn.Module):
    '''Encoder part of U-Net'''

    def __init__(self, num_channels, reduce_by=1):
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
    # TODO: fix output size being different than input

    def __init__(self, num_channels, num_classes):
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
        # _a4 = F.dropout(_a4, p=0.2)

        a4_up = self.a4_up(_a4)
        a4_up = self.blur_a4_up(a4_up)
        a3_ = self.sample_a3_()
        _a3 = self._l3(self.concat_a3(a3_, a4_up))

        a3_up = self.a3_up(_a3)
        a3_up = self.blur_a3_up(a3_up)
        a2_ = self.sample_a2_()
        _a2 = self._l2(self.concat_a2(a2_, a3_up))
        # _a2 = F.dropout(_a2, p=0.2)

        a2_up = self.a2_up(_a2)
        a2_up = self.blur_a2_up(a2_up)
        a1_ = self.sample_a1_()
        _a1 = self._l1(self.concat_a1(a1_, a2_up))

        final = self.final(_a1)
        #print(f"a_mid:{a_mid.shape}, a_mid_up:{a_mid_up.shape}, _a4:{_a4.shape}, a4_up:{a4_up.shape}, "+
        #      f"_a3:{_a3.shape}, a3_up:{a3_up.shape}, _a2:{_a2.shape}, a2_up:{a2_up.shape}, "+
        #      f"_a1:{_a1.shape}, final:{final.shape}")
        return F.log_softmax(final, 1)

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