"""U-Net architecture with residual blocks"""

import torch
from torch import nn
from .layers import BasicBlock, Upsample, Concat, conv3x3, conv1x1

class ResNetPyramid(nn.Module):

    def __init__(self, blocks_per_encoder_stage, channels_per_stage, decoder_blocks, decoder_channels, in_channels=1, num_classes=2, zero_init_residual=False):
        """U-Net with residual blocks."""

        super().__init__()

        num_stages = len(blocks_per_encoder_stage)

        if num_stages!=len(channels_per_stage):
            raise ValueError("Length of `blocks_per_encoder_stage` and `channels_per_stage` must be equal.")

        self.norm_layer = nn.BatchNorm2d
        self.residual_block = BasicBlock
        
        self.stage_input = nn.Sequential(
            nn.Conv2d(in_channels, channels_per_stage[0], kernel_size=7, stride=1, padding=3, bias=False),
            self.norm_layer(channels_per_stage[0], momentum=0.1),
            nn.ReLU(inplace=True),
        )

        #Encoder stages. Each stage involves a downsample and a change to the number of channels at the beggining,
        #followed by blocks_per_encoder_stage[i] residual blocks. The only exception is the first stage that downsamples but does not
        #change the number of channels.
        stages = [('stage_0', self._make_down_stage(channels_per_stage[0], channels_per_stage[0], blocks_per_encoder_stage[0], stride=2))]
        for idx in range(num_stages-1):
            stages.append((f'stage_{idx+1}', self._make_down_stage(channels_per_stage[idx], channels_per_stage[idx+1], blocks_per_encoder_stage[idx+1], stride=2)))

        self.encoder = nn.ModuleDict(stages)

        #Decoder stages. Each stage involves an upsample and a change to the number of channels at the beggining. The
        #upsampled activation is concatenated with the respective activation of the encoder and the number of channels
        # is halved. The last stage upsamples but do not changes the number of channels.
        upsamples, layers = self._make_decoder(channels_per_stage, decoder_channels, decoder_blocks)
        self.upsamples = upsamples
        self.decoder = layers

        self.conv_output = conv3x3(decoder_channels, num_classes)

        self._init_parameters(zero_init_residual)

    def _make_down_stage(self, in_channels, out_channels, num_blocks, stride):

        residual_adj = None      # For adjusting number of channels and size of the residual connection
        norm_layer = self.norm_layer
        block = self.residual_block

        if stride != 1 or in_channels != out_channels:
            residual_adj = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels, momentum=0.1),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, residual_adj))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        stage = nn.Sequential(*layers)

        return stage
    
    def _make_decoder(self, channels_per_stage, decoder_channels, decoder_blocks):

        residual_adj = None
        norm_layer = self.norm_layer
        block = self.residual_block
        num_inputs = len(channels_per_stage) + 1

        residual_adj = nn.Sequential(
            conv1x1(num_inputs*decoder_channels, decoder_channels),
            norm_layer(decoder_channels, momentum=0.1),
        )

        upsamples = [Upsample(channels_per_stage[0], decoder_channels)]
        for in_channels in channels_per_stage:
            upsamples.append(Upsample(in_channels, decoder_channels))
        upsamples = nn.ModuleList(upsamples)

        layers = [block(num_inputs*decoder_channels, decoder_channels, residual_adj=residual_adj)]
        for _ in range(decoder_blocks):
            layers.append(block(decoder_channels, decoder_channels))
        layers = nn.ModuleList(layers)

        return upsamples, layers

    def _init_parameters(self, zero_init_residual):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, return_acts=False):

        shape = x.shape[-2:]
        x = self.stage_input(x)

        down_samples = (x,)
        for _, stage in self.encoder.items():
            x = stage(x)
            down_samples += (x,)

        activations = []
        for upsample, sample in zip(self.upsamples, down_samples):
            activations.append(upsample(sample, shape))
        x = torch.cat(activations, dim=1)

        for block in self.decoder:
            x = block(x)

        x = self.conv_output(x)

        return x  
 