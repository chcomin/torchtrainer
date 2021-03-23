'''U-Net architecture with residual blocks'''

import torch
import torch.nn.functional as F
from torch import nn
from torch import tensor

# For importing in both the notebook and in the .py file
try:
    import ActivationSampler
except ImportError:
    from torchtrainer.module_util import ActivationSampler

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

class Blur(nn.Module):

    def __init__(self):
        super(Blur, self).__init__()

        self.pad = nn.ReplicationPad2d((0,1,0,1))
        self.blur = nn.AvgPool2d(2, stride=1)

    def forward(self, x):

        return self.blur(self.pad(x))