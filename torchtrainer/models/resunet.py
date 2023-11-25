"""U-Net architecture with residual blocks"""

import torch
from torch import nn
from .layers import BasicBlock, BasicBlockOld, Upsample, UpsampleOld, Concat, conv3x3, conv1x1

class ResUNetV2(nn.Module):

    def __init__(self, blocks_per_encoder_stage, blocks_per_decoder_stage, channels_per_stage, in_channels=1, num_classes=2, zero_init_residual=False):
        """U-Net with residual blocks."""

        super().__init__()

        num_stages = len(blocks_per_encoder_stage)
        if num_stages!=len(blocks_per_decoder_stage):
            raise ValueError("Length of `blocks_per_encoder_stage` and `blocks_per_decoder_stage` must be equal.")
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
        stages = []
        for idx in range(num_stages-1, 0, -1):
            stages.append((f'stage_{idx}',self._make_up_stage(channels_per_stage[idx], channels_per_stage[idx-1], blocks_per_decoder_stage[idx])))
        stages.append((f'stage_{0}', self._make_up_stage(channels_per_stage[0], channels_per_stage[0], blocks_per_decoder_stage[0])))

        self.decoder = nn.ModuleDict(stages)
        self.conv_output = conv3x3(channels_per_stage[0], num_classes)

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
    
    def _make_up_stage(self, in_channels, out_channels, num_blocks):

        residual_adj = None
        norm_layer = self.norm_layer
        block = self.residual_block

        residual_adj = nn.Sequential(
            conv1x1(2*out_channels, out_channels),
            norm_layer(out_channels, momentum=0.1),
        )

        upsample = Upsample(in_channels, out_channels)
        layers = []
        # 2*out_channels is used here because the first upsample block concatenates the output of a downsample block
        layers.append(block(2*out_channels, out_channels, 1, residual_adj)) 
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        stage = nn.ModuleList([upsample, Concat(), nn.Sequential(*layers)])

        return stage

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

        x = self.stage_input(x)

        down_samples = ()
        for _, stage in self.encoder.items():
            down_samples += (x,)
            x = stage(x)

        if return_acts:
            acts = down_samples + (x,)

        down_samples = down_samples[::-1]
        for (_, stage), sample in zip(self.decoder.items(), down_samples):
            upsample, concat, blocks = stage
            x = upsample(x, sample.shape[-2:])
            x = concat(sample, x)
            x = blocks(x)

            if return_acts:
                acts += (x,)            

        x = self.conv_output(x)

        if return_acts: 
            return x, acts
        else:
            return x  
        
class ResUNet(nn.Module):

    def __init__(self, layers, inplanes, in_channels=1, num_classes=2, zero_init_residual=False):
        """U-Net with residual blocks."""

        super().__init__()

        self.norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, inplanes[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = self.norm_layer(inplanes[0], momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        #Encoder stages. Each stage involves a downsample and a doubling of the number of channels at the beggining,
        #followed by layers[i] residual blocks. The only exception is the first stage that downsamples but do not
        #double the number of channels.
        stages = [('stage_0', self._make_down_layer(inplanes[0], inplanes[0], layers[0], stride=2))]
        for idx in range(len(layers)-1):
            stages.append((f'stage_{idx+1}', self._make_down_layer(inplanes[idx], inplanes[idx+1], layers[idx+1], stride=2)))

        self.encoder = nn.ModuleDict(stages)

        #Middle block
        self.mid_block = BasicBlockOld(inplanes[-1], inplanes[-1])

        #Decoder stages. Each stage involves an upsample and a halving of the number of channels at the beggining. The
        #upsampled activation is concatenated with the respective activation of the encoder and the number of channels
        # is halved again. The last stage upsamples but do not halves the number of channels.
        stages = []
        for idx in range(len(layers)-1, 0, -1):
            upsample, blocks = self._make_up_layer(inplanes[idx], inplanes[idx-1], layers[idx-1], stride=2)
            stages.append([f'stage_{idx}', nn.ModuleList([upsample, Concat(), blocks])])
        upsample, blocks = self._make_up_layer(inplanes[0], inplanes[0], layers[0], stride=2)
        stages.append([f'stage_{0}', nn.ModuleList([upsample, Concat(), blocks])])

        self.decoder = nn.ModuleDict(stages)
        self.conv_output = conv3x3(inplanes[0], num_classes)

        self._init_parameters(zero_init_residual)

    def _init_parameters(self, zero_init_residual):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockOld):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_down_layer(self, inplanes, planes, blocks, stride):

        residual_adj = None
        norm_layer = self.norm_layer
        block = BasicBlockOld

        if stride != 1 or inplanes != planes:
            residual_adj = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, residual_adj))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)
    
    def _make_up_layer(self, inplanes, planes, blocks, stride):

        residual_adj = None
        norm_layer = self.norm_layer
        block = BasicBlockOld

        residual_adj = nn.Sequential(
            conv1x1(2*planes, planes),
            norm_layer(planes, momentum=0.1),
        )

        upsample = UpsampleOld(inplanes, planes, stride=stride)
        layers = []
        layers.append(block(2*planes, planes, 1, residual_adj))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return upsample, nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        down_samples = ()
        for _, stage in self.encoder.items():
            down_samples += (x,)
            x = stage(x)

        x = self.mid_block(x)

        down_samples = down_samples[::-1]
        for (_, stage), sample in zip(self.decoder.items(), down_samples):
            upsample, concat, blocks = stage
            x = upsample(x, sample.shape[-2:])
            x = concat(sample, x)
            x = blocks(x)

        x = self.conv_output(x)     

        return x  
        
