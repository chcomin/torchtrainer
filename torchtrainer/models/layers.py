"Some useful neural net layers"

from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)

def ntuple(x, n):
    '''Verify if x is iterable. If not, create tuple containing x repeated n times'''

    if isinstance(x, Iterable):
        return x
    return tuple([x]*n)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class BasicUpsampleBlock_old(nn.Module):

    def __init__(self, inplanes, planes, stride=2, residual_adj=None, use_conv_transpose=False):
        super().__init__()

        if use_conv_transpose:
            #kernel_size=4 because the input tensor will be filled with zeros as [a, 0., b, 0., c, 0.,...] and
            #kernel_size=3 would only reach a single value in some positions. Also, with kernel_size=4 the size
            #of the output is always exactly double of the input for stride=2.
            self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=4, stride=stride, padding=1, bias=False)
        else:
            self.upsample = Upsample()
            self.conv1 = conv3x3(inplanes, planes)    
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.residual_adj = residual_adj
        self.use_conv_transpose = use_conv_transpose

    def forward(self, x, output_shape):
        identity = x

        if self.use_conv_transpose:
            out = self.conv1(x)
            if out[-2:]!=output_shape:
                out = F.interpolate(out, output_shape, mode='nearest')
        else:
            out = self.upsample(x, output_shape)
            out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual_adj is not None:
            identity = self.residual_adj(x, out.shape[-2:])

        out += identity
        out = self.relu(out)

        return out
    
class Upsample(nn.Module):

    def __init__(self, inplanes=None, planes=None, stride=2, use_conv=False, use_conv_transpose=False, mode='nearest'):
        super().__init__()

        if use_conv or use_conv_transpose:
            if inplanes is None or planes is None:
                raise ValueError('When use_conv or use_conv_transpose is True, inplanes and planes must be defined.')
        if use_conv and use_conv_transpose:
            raise ValueError('use_conv and use_conv_transpose cannot be both True.')

        if use_conv_transpose:
            #kernel_size=4 because the input tensor will be filled with zeros as [a, 0., b, 0., c, 0.,...] and
            #kernel_size=3 would only reach a single value in some positions. Also, with kernel_size=4 the size
            #of the output is always exactly double of the input for stride=2.
            self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=4, stride=stride, padding=1, bias=False)
        elif use_conv:
            self.conv1 = conv1x1(inplanes, planes)    
            self.interpolate = Interpolate(mode)
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.mode = mode

    def forward(self, x, output_shape):
  
        if self.use_conv_transpose:
            x = self.conv1(x)
            if x.shape[-2:]!=output_shape:
                x = F.interpolate(x, output_shape, mode='nearest')
        else:
            if self.use_conv:
                x = self.conv1(x)
            x = self.interpolate(x, output_shape)
 
        return x
    
class Interpolate(nn.Module):

    def __init__(self, mode='nearest'):
        super().__init__()
        self.mode = mode

    def forward(self, x, output_shape):
        return F.interpolate(x, output_shape, mode=self.mode)    
    
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class Concat(nn.Module):
    '''Module for concatenating two activations using interpolation'''

    def __init__(self, concat_dim=1):
        super(Concat, self).__init__()
        self.concat_dim = concat_dim

    def forward(self, x1, x2):
        # Inputs will be padded if not the same size

        if x1.shape[self.concat_dim+1:]!=x2.shape[self.concat_dim+1:]:
            x1, x2 = self.fix_shape(x1, x2)
        return torch.cat((x1, x2), self.concat_dim)

    def fix_shape(self, x1, x2):

        x2 = F.interpolate(x2, x1.shape[self.concat_dim+1:], mode='nearest')

        return x1, x2

    def extra_repr(self):
        s = 'concat_dim={concat_dim}'
        return s.format(**self.__dict__)

class Blur(nn.Module):

    def __init__(self):
        super(Blur, self).__init__()

        self.pad = nn.ReplicationPad2d((0,1,0,1))
        self.blur = nn.AvgPool2d(2, stride=1)

    def forward(self, x):

        return self.blur(self.pad(x))

class Conv2dCH(nn.Module):
    """Create 2D cross-hair convolution filter. Parameters are the same as torch.nn.Conv2d, with the exception
    that padding must be larger than or equal to (kernel_size-1)//2 (otherwise the filter would need negative padding
    to properly work) and dilation is not supported. Also, if padding is not provided it will be equal to (kernel_size-1)//2.
    That is, by default the result of the convolution has the same shape as the input tensor.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Size of the kernel. Even sizes are not supported.
    stride : int or tuple
        Stride of the convolution.
    padding : int or tuple
        Padding of the input. Must be larger than or equal to (kernel_size-1)//2.
    groups : int
        Controls the connections between inputs and outputs.
    bias : bool
        If True, adds a learnable bias to the output.
    padding_mode : string
        Padding mode to use.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=True,
                 padding_mode='zeros'):
        super(Conv2dCH, self).__init__()

        kernel_size = ntuple(kernel_size, 2)
        stride = ntuple(stride, 2)
        padding = ntuple(padding, 2)

        padding = list(padding)
        if padding[0] is None:
            padding[0] = (kernel_size[0]-1)//2
        if padding[1] is None:
            padding[1] = (kernel_size[1]-1)//2

        if padding[0]<(kernel_size[0]-1)//2:
            raise ValueError("Padding must be padding[0]>=(kernel_size[0]-1)//2")
        if padding[1]<(kernel_size[1]-1)//2:
            raise ValueError("Padding must be padding[1]>=(kernel_size[1]-1)//2")

        pad_conv1d_v_h = padding[1]-(kernel_size[1]-1)//2
        pad_conv1d_h_v = padding[0]-(kernel_size[0]-1)//2
        self.conv1d_v = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=(kernel_size[0], 1), stride=stride, padding=(padding[0], pad_conv1d_v_h),
                                 groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv1d_h = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=(1, kernel_size[1]), stride=stride, padding=(pad_conv1d_h_v, padding[1]),
                                 groups=groups, bias=bias, padding_mode=padding_mode)

        self.reset_parameters()

    def forward(self, x):
        return self.conv1d_v(x) + self.conv1d_h(x)

    def reset_parameters(self):

        self.conv1d_v.reset_parameters()
        self.conv1d_h.reset_parameters()

        '''for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()'''

class Conv3dCH(nn.Module):
    """Create 3D cross-hair convolution filter. Parameters are the same as torch.nn.Conv3d, with the exception
    that padding must be larger than or equal to (kernel_size-1)//2 (otherwise the filter would need negative padding
    to properly work) and dilation is not supported. Also, if padding is not provided it will be equal to (kernel_size-1)//2.
    That is, by default the result of the convolution has the same shape as the input tensor.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Size of the kernel. Even sizes are not supported.
    stride : int or tuple
        Stride of the convolution.
    padding : int or tuple
        Padding of the input. Must be larger than or equal to (kernel_size-1)//2.
    groups : int
        Controls the connections between inputs and outputs.
    bias : bool
        If True, adds a learnable bias to the output.
    padding_mode : string
        Padding mode to use.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=True,
                 padding_mode='zeros'):
        super(Conv3dCH, self).__init__()

        kernel_size = ntuple(kernel_size, 3)
        stride = ntuple(stride, 3)
        padding = ntuple(padding, 3)

        padding = list(padding)
        if padding[0] is None:
            padding[0] = (kernel_size[0]-1)//2
        if padding[1] is None:
            padding[1] = (kernel_size[1]-1)//2
        if padding[2] is None:
            padding[2] = (kernel_size[2]-1)//2

        if padding[0]<(kernel_size[0]-1)//2:
            raise ValueError("Padding must be padding[0]>=(kernel_size[0]-1)//2")
        if padding[1]<(kernel_size[1]-1)//2:
            raise ValueError("Padding must be padding[1]>=(kernel_size[1]-1)//2")
        if padding[2]<(kernel_size[2]-1)//2:
            raise ValueError("Padding must be padding[2]>=(kernel_size[2]-1)//2")

        pad_conv2d_p_v = pad_conv2d_h_v = padding[1]-(kernel_size[1]-1)//2
        pad_conv2d_p_h = pad_conv2d_v_h = padding[2]-(kernel_size[2]-1)//2
        pad_conv2d_v_p = pad_conv2d_h_p = padding[0]-(kernel_size[0]-1)//2

        self.conv2d_p = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=(kernel_size[0], 1, 1), stride=stride,
                                 padding=(padding[0], pad_conv2d_p_v, pad_conv2d_p_h),
                                 groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv2d_v = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=(1, kernel_size[1], 1), stride=stride,
                                 padding=(pad_conv2d_v_p, padding[1], pad_conv2d_v_h),
                                 groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv2d_h = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=(1, 1, kernel_size[2]), stride=stride,
                                 padding=(pad_conv2d_h_p, pad_conv2d_h_v, padding[2]),
                                 groups=groups, bias=bias, padding_mode=padding_mode)

        self.reset_parameters()

    def forward(self, x):
        return self.conv2d_p(x) + self.conv2d_v(x) + self.conv2d_h(x)

    def reset_parameters(self):

        self.conv2d_p.reset_parameters()
        self.conv2d_v.reset_parameters()
        self.conv2d_h.reset_parameters()

        '''for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()'''