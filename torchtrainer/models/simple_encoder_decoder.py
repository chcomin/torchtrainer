"""A simple encoder decoder architecture that accepts any Pytorch resnet model"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def conv_norm(in_channels, out_channels, kernel_size=3, act=True):
    """"conv-batchnorm-reul block."""

    layer = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                padding=kernel_size//2, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if act:
        layer += [nn.ReLU()]
    
    return nn.Sequential(*layer)

class DecoderBlock(nn.Module):
    """Receives the activation from the previous level of the decoder `x_dec` and the activation from 
    the encoder `x_enc`. It is assumed that `x_dec` has a smaller spatial resolution than `x_enc` 
    and that `x_enc` has a different number of channels than `x_dec`. The module adjusts the 
    resolution of `x_dec` to be equal to `x_enc` and the number of channels of `x_enc` to be equal 
    to `x_dec`.

    Args:
        enc_channels: Number of channels of the encoder features.
        dec_channels: Number of channels to use for the decoder.

    """

    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        self.channel_adjust = conv_norm(enc_channels, dec_channels, kernel_size=1,
                                        act=False)
        self.mix = conv_norm(dec_channels, dec_channels)

    def forward(self, x_enc, x_dec):
        x_dec_int = F.interpolate(x_dec, size=x_enc.shape[-2:], mode="nearest")
        x_enc_ad = self.channel_adjust(x_enc)
        y = x_dec_int + x_enc_ad
        return self.mix(y)

class Decoder(nn.Module):
    """Decode a list of features having distinct resolutions and number of channels."""

    def __init__(self, encoder_channels_list, decoder_channels):
        super().__init__()

        encoder_channels_list = encoder_channels_list[::-1]

        self.middle = conv_norm(encoder_channels_list[0], decoder_channels)
        blocks = []
        for channels in encoder_channels_list[1:]:
            blocks.append(DecoderBlock(channels, decoder_channels))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):

        features = features[::-1]

        x = self.middle(features[0])
        for idx in range(1, len(features)):
            x = self.blocks[idx-1](features[idx], x)

        return x

class SimpleEncoderDecoder(nn.Module):
    """Sample the activations of a Pytorch ResNet model and create a decoder."""

    def __init__(self, resnet_encoder, decoder_channels, num_classes):
        super().__init__()

        self.resnet_encoder = resnet_encoder
        encoder_channels_list = self.get_channels()
        self.decoder = Decoder(encoder_channels_list, decoder_channels)
        self.classification = nn.Conv2d(decoder_channels, num_classes, 3, padding=1)
        
    def get_features(self, x):
        """Extract the features from a ResNet encoder."""
        
        features = []
        re = self.resnet_encoder
        x = re.conv1(x)
        x = re.bn1(x)
        x = re.relu(x)
        features.append(x)
        x = re.maxpool(x)

        x = re.layer1(x)
        features.append(x)
        x = re.layer2(x)
        features.append(x)
        x = re.layer3(x)
        features.append(x)
        x = re.layer4(x)
        features.append(x)

        return features

    def get_channels(self):
        """Get the number of channels of the encoder features."""

        re = self.resnet_encoder
        training = re.training
        re.eval()

        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            features = self.get_features(x)
        encoder_channels_list = [f.shape[1] for f in features]

        if training:
            re.train()

        return encoder_channels_list
        
    def forward(self, x):
        in_shape = x.shape[-2:]
        features = self.get_features(x)
        x = self.decoder(features)

        if x.shape[-2:]!=in_shape:
            x = F.interpolate(x, size=in_shape, mode="nearest")

        x = self.classification(x)

        return x

def get_model(encoder_name="resnet18", decoder_channels=64, num_classes=2, weights_strategy=None):
    """Get a SimpleEncoderDecoder model.
    
    Parameters
    ----------
    encoder_name
        Name of the ResNet encoder to use. The encoder must be available in torch.hub.
    decoder_channels
        Number of channels to use in the decoder.
    num_classes
        Number of classes to predict.
    weights_strategy
        Strategy to load the weights. If "encoder" is passed, only the encoder weights will be 
        loaded. If a string is passed, it is assumed to be a path to a checkpoint file.
    """

    if weights_strategy=="encoder":
        # Load only encoder weights
        weights = "DEFAULT"
    else:
        weights = None

    encoder = torch.hub.load("pytorch/vision", encoder_name, weights=weights, verbose=False)

    model = SimpleEncoderDecoder(encoder, decoder_channels=decoder_channels, 
                                 num_classes=num_classes)  

    # Check if weights_strategy is a path to a checkpoint file
    if weights_strategy is not None:
        weights_path = Path(weights_strategy)
        if weights_path.is_file():
            model.load_state_dict(torch.load(weights_path))

    return model