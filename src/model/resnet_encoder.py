import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.utils.model_zoo as model_zoo


class MultiImageResNet(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(MultiImageResNet, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResNetEncoder(nn.Module):
    """ResNet encoder for feature extraction."""

    def __init__(self, num_layers, pretrained=False, num_input_images=1):
        """
        Args:
            num_layers (int): Number of ResNet layers (18, 34, 50, 101, 152).
            pretrained (bool): If True, load pretrained weights.
            num_input_images (int): Number of stacked input images.
        """
        super(ResNetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnet_configs = {
            18: (models.resnet.BasicBlock, [2, 2, 2, 2]),
            34: (models.resnet.BasicBlock, [3, 4, 6, 3]),
            50: (models.resnet.Bottleneck, [3, 4, 6, 3]),
            101: (models.resnet.Bottleneck, [3, 4, 23, 3]),
            152: (models.resnet.Bottleneck, [3, 8, 36, 3]),
        }
        if num_layers not in resnet_configs:
            raise ValueError(f"{num_layers} is not a valid number of ResNet layers")

        block_type, layer_config = resnet_configs[num_layers]
        self.encoder = MultiImageResNet(block_type, layer_config, num_input_images)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if pretrained and num_input_images == 1:
            state_dict = model_zoo.load_url(
                models.resnet.model_urls[f"resnet{num_layers}"]
            )
            self.encoder.load_state_dict(state_dict)

    def forward(self, input_image):
        """
        Forward pass for the encoder.

        Args:
            input_image (torch.Tensor): Input image tensor.

        Returns:
            list: List of feature maps from each ResNet stage.
        """
        self.x = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.x.append(self.encoder.relu(x))
        self.x.append(self.encoder.layer1(self.encoder.maxpool(self.x[-1])))
        self.x.append(self.encoder.layer2(self.x[-1]))
        self.x.append(self.encoder.layer3(self.x[-1]))
        self.x.append(self.encoder.layer4(self.x[-1]))

        return self.x
