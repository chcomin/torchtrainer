'''U-Net architecture'''

import torch
import torch.nn.functional as F
from torch import nn
from torch import tensor
from ..module_util import ActivationSampler

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels, kernel_size=3, p=1):
        super(DoubleConvolution, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=kernel_size, padding=p),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(middle_channel),
            nn.Conv2d(middle_channel, out_channels, kernel_size=kernel_size, padding=p),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        self.dconv = nn.Sequential(*layers)

    def forward(self, x):
        return self.dconv(x)

class ResDoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels, kernel_size=3, p=1):
        super(ResDoubleConvolution, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=kernel_size, padding=p, bias=False),
            nn.BatchNorm2d(middle_channel),
            self.relu,
            nn.Conv2d(middle_channel, out_channels, kernel_size=kernel_size, padding=p, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        self.dconv = nn.Sequential(*layers)

    def forward(self, x):

        return self.relu(self.dconv(x) + x)

class Concat(nn.Module):
    '''Module for concatenating two activations'''

    def __init__(self, concat_dim=1):
        super(Concat, self).__init__()
        self.concat_dim = concat_dim

    def forward(self, x1, x2):
        # Inputs will be padded if not the same size

        x1, x2 = self.pad_inputs(x1, x2)
        return torch.cat((x1, x2), self.concat_dim)

    def pad_inputs(self, x1, x2):

        cd = self.concat_dim
        shape_diff = tensor(x2.shape[cd+1:]) - tensor(x1.shape[cd+1:])
        pad1 = []
        pad2 = []
        for sd in shape_diff.flip(0):
            sd_abs = abs(sd.item())
            if sd%2==0:
                pb = pe = sd_abs//2
            else:
                pb = sd_abs//2
                pe = pb + 1

            if sd>=0:
                pad1 += [pb, pe]
                pad2 += [0, 0]
            else:
                pad1 += [0, 0]
                pad2 += [pb, pe]

        x1 = F.pad(x1, pad1)
        x2 = F.pad(x2, pad2)

        return x1, x2

    def extra_repr(self):
        s = 'concat_dim={concat_dim}'
        return s.format(**self.__dict__)


class Encoder(nn.Module):
    '''Encoder part of U-Net'''

    def __init__(self, num_channels, ConvBlock, reduce_by=1):
        super(Encoder, self).__init__()

        self.l1_ = ConvBlock(num_channels, 64//reduce_by, 64//reduce_by)
        self.a1_dwn = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l2_ = ConvBlock(64//reduce_by, 128//reduce_by, 128//reduce_by)
        self.a2_dwn = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l3_ = ConvBlock(128//reduce_by, 256//reduce_by, 256//reduce_by)
        self.a3_dwn = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l4_ = ConvBlock(256//reduce_by, 512//reduce_by, 512//reduce_by)
        self.a4_dwn = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l_mid = ConvBlock(512//reduce_by, 1024//reduce_by, 1024//reduce_by)

    def forward(self, x):
        for layer in self.children(): x = layer(x)
        return x


class EDUNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(EDUNet, self).__init__()

        reduce_by = 1

        ConvBlock = DoubleConvolution
        self.encoder = Encoder(num_channels, ConvBlock)

        self.a_mid_up = nn.ConvTranspose2d(1024//reduce_by, 512//reduce_by, kernel_size=2, stride=2)
        self.sample_a4_ = ActivationSampler(self.encoder.l4_)
        self.concat_a4 = Concat(1)
        self._l4 = ConvBlock(1024//reduce_by, 512//reduce_by, 512//reduce_by)

        self.a4_up = nn.ConvTranspose2d(512//reduce_by, 256//reduce_by, kernel_size=2, stride=2)
        self.sample_a3_ = ActivationSampler(self.encoder.l3_)
        self.concat_a3 = Concat(1)
        self._l3 = ConvBlock(512//reduce_by, 256//reduce_by, 256//reduce_by)

        self.a3_up = nn.ConvTranspose2d(256//reduce_by, 128//reduce_by, kernel_size=2, stride=2)
        self.sample_a2_ = ActivationSampler(self.encoder.l2_)
        self.concat_a2 = Concat(1)
        self._l2 = ConvBlock(256//reduce_by, 128//reduce_by, 128//reduce_by)

        self.a2_up = nn.ConvTranspose2d(128//reduce_by, 64//reduce_by, kernel_size=2, stride=2)
        self.sample_a1_ = ActivationSampler(self.encoder.l1_)
        self.concat_a1 = Concat(1)
        self._l1 = ConvBlock(128//reduce_by, 64//reduce_by, 64//reduce_by)

        self.final = nn.Conv2d(64//reduce_by, num_classes, kernel_size=1)
        self.reset_parameters()

    def forward(self, x):
        a_mid = self.encoder(x)

        a_mid_up = self.a_mid_up(a_mid)
        a4_ = self.sample_a4_()
        _a4 = self._l4(self.concat_a4(a4_, a_mid_up))
        # _a4 = F.dropout(_a4, p=0.2)

        a4_up = self.a4_up(_a4)
        a3_ = self.sample_a3_()
        _a3 = self._l3(self.concat_a3(a3_, a4_up))

        a3_up = self.a3_up(_a3)
        a2_ = self.sample_a2_()
        _a2 = self._l2(self.concat_a2(a2_, a3_up))
        # _a2 = F.dropout(_a2, p=0.2)

        a2_up = self.a2_up(_a2)
        a1_ = self.sample_a1_()
        _a1 = self._l1(self.concat_a1(a1_, a2_up))

        final = self.final(_a1)
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

class ActivationSampler(nn.Module):
    '''Generates a hook for sampling a layer activation'''

    def __init__(self, model):
        super(ActivationSampler, self).__init__()
        self.model_name = model.__class__.__name__
        self.activation = None
        model.register_forward_hook(self.get_hook())

    def forward(self, x=None):
        return self.activation

    def get_hook(self):
        def hook(model, input, output):
            self.activation = output
        return hook

    def extra_repr(self):
        return f'{self.model_name}'
