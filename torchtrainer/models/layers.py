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
    '''Residual block.'''

    def __init__(self, in_channels, out_channels, stride=1, residual_adj=None):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu2 = nn.ReLU(inplace=True)
        self.residual_adj = residual_adj

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual_adj is not None:
            identity = self.residual_adj(x)

        out += identity
        out = self.relu2(out)

        return out
    
class BasicBlockOld(nn.Module):
    '''Residual block. Some attributes in this class have nonintuitive names. These names were changed in class BasicBlock.'''

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out
    
class Upsample(nn.Module):
    '''Upsamples an input and optionally adjusts the number of channels.
    upsample_strategy can be 'interpolation', 'conv_transpose' or 'grid'.
    '''

    def __init__(self, in_channels=None, out_channels=None, stride=2, upsample_strategy='interpolation', mode='nearest'):
        super().__init__()

        if upsample_strategy=='conv_transpose':
            #kernel_size=4 because the input tensor will be filled with zeros as [a, 0., b, 0., c, 0.,...] and
            #kernel_size=3 would only reach a single value in some positions. Also, with kernel_size=4 the size
            #of the output is always exactly double of the input for stride=2.
            self.conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True)
            )
        else:
            if in_channels!=out_channels:
                self.channel_adj = nn.Sequential(conv1x1(in_channels, out_channels),
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU(inplace=True)
                )
            else:
                self.channel_adj = None 
            if upsample_strategy=='interpolation':
                self.interpolate = Interpolate(mode)
            elif upsample_strategy=='grid':    
                self.interpolate = InterpolateGrid(mode)

        self.upsample_strategy = upsample_strategy
        self.mode = mode

    def forward(self, x, output_shape):
  
        if self.upsample_strategy=='conv_transpose':
            x = self.conv(x)
            if x.shape[-2:]!=output_shape:
                x = F.interpolate(x, output_shape, mode='nearest')
        else:
            if self.channel_adj is not None:
                x = self.channel_adj(x)
            x = self.interpolate(x, output_shape)            
 
        return x
    
class UpsampleOld(nn.Module):
    '''Upsamples an input and optionally adjusts the number of channels. Some attributes in this class have
    nonintuitive names. These names were changed in class Upsample.'''

    def __init__(self, inplanes=None, planes=None, stride=2, use_conv_transpose=False, mode='nearest'):
        super().__init__()

        if use_conv_transpose:
            #kernel_size=4 because the input tensor will be filled with zeros as [a, 0., b, 0., c, 0.,...] and
            #kernel_size=3 would only reach a single value in some positions. Also, with kernel_size=4 the size
            #of the output is always exactly double of the input for stride=2.
            self.conv = nn.Sequential(nn.ConvTranspose2d(inplanes, planes, kernel_size=4, stride=stride, padding=1, bias=False),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU(inplace=True)
            )
        else:
            if inplanes!=planes:
                self.conv = nn.Sequential(conv1x1(inplanes, planes),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(inplace=True)
                )
            else:
                self.conv = None 
            self.interpolate = Interpolate(mode)

        self.use_conv_transpose = use_conv_transpose
        self.mode = mode

    def forward(self, x, output_shape):
  
        if self.use_conv_transpose:
            x = self.conv(x)
            if x.shape[-2:]!=output_shape:
                x = F.interpolate(x, output_shape, mode='nearest')
        else:
            if self.conv is not None:
                x = self.conv(x)
            x = self.interpolate(x, output_shape)
 
        return x

class Interpolate(nn.Module):
    '''Layer that receives two inputs when called: x and output_shape. The layer interpolates the input
    to the desired output.'''

    def __init__(self, mode='nearest'):
        super().__init__()
        self.mode = mode

    def forward(self, x, output_shape):
        return F.interpolate(x, output_shape, mode=self.mode)    
    
class InterpolateConv(nn.Module):
    """Linearly interpolate tensor to x2 the size using a fixed filter."""

    def __init__(self):
        super().__init__()

        self.filter = torch.tensor([
            [0.25, 0.5, 0.25],
            [0.5 , 1. ,  0.5],
            [0.25, 0.5, 0.25]
        ])

    def forward(self, x):
        return self.interpolate_conv(x)

    def interpolate_conv(self, x):

        channels = x.shape[1]
        weight = torch.tile(self.filter, (channels,1,1,1)).to(x.device)
        res = F.conv_transpose2d(x, weight, stride=2, padding=1, output_padding=1, groups=channels)
        
        return res

class InterpolateGrid(nn.Module):

    def __init__(self, mode='nearest'):
        super().__init__()
        self.mode = mode

    def forward(self, x, output_shape):
        return interpolate_grid_sample(x, output_shape, mode=self.mode)

def interpolate_grid_sample(input, size=None, scale_factor=None, mode='nearest', half_r=True, half_c=True):
    '''Interpolate tensor using half pixel shifts, which preserves the position of the receptive field.'''

    if size is None and scale_factor is None:
        raise ValueError("Either size os scale_factor must be given")
    if isinstance(size, int):
        size = (size, size)
    if isinstance(scale_factor, (int,float)):
        scale_factor = (scale_factor, scale_factor)

    nr_i, nc_i = input.shape[-2:]
    if size is None:
        nr_o = int(scale_factor[0]*nr_i)
        nc_o = int(scale_factor[1]*nc_i)
    else:
        nr_o, nc_o = size

    if half_r:
        delta_r = 0.5
    else:
        delta_r = 0.
    if half_c:
        delta_c = 0.5
    else:
        delta_c = 0.

    r = nr_i*(torch.arange(0, nr_o)+0.5)/nr_o + delta_r
    rn = 2*(r/nr_i - 0.5)
    c = nc_i*(torch.arange(0, nc_o)+0.5)/nc_o + delta_c
    cn = 2*(c/nc_i - 0.5)

    rr, cc = torch.meshgrid(rn, cn, indexing='ij')
    grid = torch.stack((cc,rr), dim=2)
    grid = grid.tile((input.shape[0],1,1,1))
    grid = grid.to(input.device)

    res = torch.nn.functional.grid_sample(input, grid, mode=mode, align_corners=False)

    return res

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
        
