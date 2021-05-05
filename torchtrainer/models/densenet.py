import torch
import torch.nn.functional as F
from torch import nn
from torch import tensor
from .layers import ResBlock, conv3x3, conv1x1

class DenseNet(nn.Module):

    def __init__(self, num_channels, num_classes, layers=[8, 16, 32, 64]):
        super().__init__()

        sum_channels = num_channels
        for idx, channels_curr_layer in enumerate(layers):
            setattr(self, f'resblock{idx+1}', ResBlock(sum_channels, channels_curr_layer, stride=1))
            sum_channels += channels_curr_layer
            
        self.output_conv = nn.Conv2d(sum_channels, num_classes, kernel_size=1)
        self.reset_parameters()

    def forward(self, x):
        blocks = list(self.children())

        layers_outputs = [x]
        for layer in self.children(): 
            x = torch.cat(layers_outputs, dim=1)
            x = layer(x)
            layers_outputs.append(x)
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
