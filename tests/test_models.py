import unittest
from torchtrainer.models import densenet, simplenet, resunet, layers
import torch

class TestModels(unittest.TestCase):

    def test_unet(self):

        img = torch.rand(1, 1, 256, 256)
        runet = resunet.ResUNet(1, 2)
        runet(img)

    def test_flexible_unet_default(self):

        img = torch.rand(1, 1, 256, 256)
        runet = resunet.FlexibleResUNet(1, 2)
        runet(img)

    def test_flexible_unet_layers(self):

        img = torch.rand(1, 1, 256, 256)
        runet = resunet.FlexibleResUNet(1, 2, layers=[8, 16])
        runet(img)

    def test_flexible_unet_blur(self):

        img = torch.rand(1, 1, 256, 256)
        runet = resunet.FlexibleResUNet(1, 2, use_blur=False)
        runet(img)

    def test_densenet(self):

        img = torch.rand(1, 1, 256, 256)
        runet = densenet.DenseNet(1, 2, layers=[8, 16, 32, 64])
        runet(img)

    def test_flexible_simplenet(self):

        img = torch.rand(1, 1, 256, 256)
        runet = simplenet.FlexibleSimpleNet(1, 2, layers=[4, 8, 64])
        runet(img)

    def test_flexible_simplenet_no_input_conv(self):

        img = torch.rand(1, 1, 256, 256)
        runet = simplenet.FlexibleSimpleNet(1, 2, layers=[16, 32, 64], input_conv=False)
        runet(img)

if __name__ == '__main__':
    # Run with python -m unittest discover -s tests
    unittest.main()