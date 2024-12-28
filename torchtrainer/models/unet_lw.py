""" Model from the Little wnet paper."""
import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):  # noqa: D103
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock(torch.nn.Module):
    """Residual block"""

    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
        """pool_mode can be False (no pooling) or True ('maxpool')"""
        super().__init__()
        if shortcut: 
            self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))
        else: 
            self.shortcut=False
        pad = (k_sz - 1) // 2

        block = []
        if pool: 
            self.pool = nn.MaxPool2d(kernel_size=2)
        else: 
            self.pool = False

        block.append(nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        block.append(nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*block)
    def forward(self, x):
        if self.pool: 
            x = self.pool(x)
        out = self.block(x)
        if self.shortcut: 
            return out + self.shortcut(x)
        else: 
            return out

class UpsampleBlock(torch.nn.Module):
    """Upsample input using transpose convolution or interpolation."""

    def __init__(self, in_c, out_c, up_mode="transp_conv"):
        super().__init__()
        block = []
        if up_mode == "transp_conv":
            block.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
        elif up_mode == "up_conv":
            block.append(nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False))
            block.append(nn.Conv2d(in_c, out_c, kernel_size=1))
        else:
            raise Exception("Upsampling mode not supported")

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class ConvBridgeBlock(torch.nn.Module):
    """Conv-ReLU-Batch Norm block"""

    def __init__(self, channels, k_sz=3):
        super().__init__()
        pad = (k_sz - 1) // 2
        block=[]

        block.append(nn.Conv2d(channels, channels, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(channels))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock(torch.nn.Module):
    """Upsampling block with a convolutional bridge."""

    def __init__(self, in_c, out_c, k_sz=3, up_mode="up_conv", conv_bridge=False, shortcut=False):
        super().__init__()
        self.conv_bridge = conv_bridge

        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode)
        self.conv_layer = ConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

    def forward(self, x, skip):
        up = self.up_layer(x)
        if self.conv_bridge:
            out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
        else:
            out = torch.cat([up, skip], dim=1)
        out = self.conv_layer(out)
        return out

class UNetLW(nn.Module):
    """U-Net model from the paper
    State-of-the-art retinal vessel segmentation with minimalistic models
    https://www.nature.com/articles/s41598-022-09675-y
    """

    def __init__(
            self, 
            in_c, 
            n_classes, 
            layers, 
            k_sz=3, 
            up_mode="transp_conv", 
            conv_bridge=True, 
            shortcut=True
            ):
        super().__init__()
        self.n_classes = n_classes
        self.first = ConvBlock(in_c=in_c, out_c=layers[0], k_sz=k_sz,
                               shortcut=shortcut, pool=False)

        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            block = ConvBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                              shortcut=shortcut, pool=True)
            self.down_path.append(block)

        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
                                up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut)
            self.up_path.append(block)

        # init, shamelessly lifted from torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.final = nn.Conv2d(layers[0], n_classes, kernel_size=1)

    def forward(self, x):
        x = self.first(x)
        down_activations = []
        for down in self.down_path:
            down_activations.append(x)
            x = down(x)
        down_activations.reverse()
        for i, up in enumerate(self.up_path):
            x = up(x, down_activations[i])
        return self.final(x)
    
def get_model(layers=(8, 16, 32), k_sz=3, num_channels=1, num_classes=2):
    """Get the model."""

    model = UNetLW(in_c=num_channels, n_classes=num_classes, layers=layers, k_sz=k_sz)

    return model