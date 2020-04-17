'''U-Net architecture'''

import torch
import torch.nn.functional as F
from torch import nn

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels, kernel_size=3, p=1):
        super(DoubleConvolution, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=kernel_size, padding=p),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, out_channels, kernel_size=kernel_size, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.dconv = nn.Sequential(*layers)

    def forward(self, x):
        return self.dconv(x)

class UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()

        reduce_by = 1

        self.l1_ = DoubleConvolution(num_channels, 64//reduce_by, 64//reduce_by)
        self.a1_dwn = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l2_ = DoubleConvolution(64//reduce_by, 128//reduce_by, 128//reduce_by)
        self.a2_dwn = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l3_ = DoubleConvolution(128//reduce_by, 256//reduce_by, 256//reduce_by)
        self.a3_dwn = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l4_ = DoubleConvolution(256//reduce_by, 512//reduce_by, 512//reduce_by)
        self.a4_dwn = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l_mid = DoubleConvolution(512//reduce_by, 1024//reduce_by, 1024//reduce_by)

        self.a_mid_up = nn.ConvTranspose2d(1024//reduce_by, 512//reduce_by, kernel_size=2, stride=2)
        self._l4 = DoubleConvolution(1024//reduce_by, 512//reduce_by, 512//reduce_by)

        self.a4_up = nn.ConvTranspose2d(512//reduce_by, 256//reduce_by, kernel_size=2, stride=2)
        self._l3 = DoubleConvolution(512//reduce_by, 256//reduce_by, 256//reduce_by)

        self.a3_up = nn.ConvTranspose2d(256//reduce_by, 128//reduce_by, kernel_size=2, stride=2)
        self._l2 = DoubleConvolution(256//reduce_by, 128//reduce_by, 128//reduce_by)

        self.a2_up = nn.ConvTranspose2d(128//reduce_by, 64//reduce_by, kernel_size=2, stride=2)
        self._l1 = DoubleConvolution(128//reduce_by, 64//reduce_by, 64//reduce_by)

        self.final = nn.Conv2d(64//reduce_by, num_classes, kernel_size=1)
        self.reset_parameters()

    def forward(self, x):

        a1_ = self.l1_(x)
        a1_dwn = self.a1_dwn(a1_)

        a2_ = self.l2_(a1_dwn)
        a2_dwn = self.a2_dwn(a2_)

        a3_ = self.l3_(a2_dwn)
        a3_dwn = self.a3_dwn(a3_)

        a4_ = self.l4_(a3_dwn)
        # a4_ = F.dropout(a4_, p=0.2)
        a4_dwn = self.a4_dwn(a4_)

        a_mid = self.l_mid(a4_dwn)

        a_mid_up = self.a_mid_up(a_mid)
        _a4 = self._l4(UNet.match_and_concat(a4_, a_mid_up))
        # _a4 = F.dropout(_a4, p=0.2)

        a4_up = self.a4_up(_a4)
        _a3 = self._l3(UNet.match_and_concat(a3_, a4_up))

        a3_up = self.a3_up(_a3)
        _a2 = self._l2(UNet.match_and_concat(a2_, a3_up))
        # _a2 = F.dropout(_a2, p=0.2)

        a2_up = self.a2_up(_a2)
        _a1 = self._l1(UNet.match_and_concat(a1_, a2_up))

        final = self.final(_a1)
        return F.log_softmax(final, 1)

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):

        if crop:
            c_h = (bypass.shape[2] - upsampled.shape[2])
            c_w = (bypass.shape[3] - upsampled.shape[3])
            if c_h%2==0:
                c_hu = c_hd = c_h//2
            else:
                c_hu = c_h//2
                c_hd = c_h//2+1
            if c_w%2==0:
                c_wl = c_wr = c_w//2
            else:
                c_wl = c_w//2
                c_wr = c_w//2+1

            bypass = F.pad(bypass, (-c_wl, -c_wr, -c_hu, -c_hd))
        return torch.cat((upsampled, bypass), 1)

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