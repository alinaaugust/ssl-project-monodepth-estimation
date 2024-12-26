import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.utils.model_zoo as model_zoo


class MultiImageResNet(models.ResNet):
    """
    ResNet variant that supports a custom number of input image frames.

    Adapted from torchvision's ResNet implementation.
    """
    def __init__(self, block_type, layer_config, num_classes=1000, num_image_frames=1):
        """
        Args:
            block_type (nn.Module): BasicBlock or Bottleneck block.
            layer_config (list): Number of layers for each ResNet stage.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            num_image_frames (int, optional): Number of stacked input image frames. Defaults to 1.
        """
        super(MultiImageResNet, self).__init__(block_type, layer_config)
        self.input_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=num_image_frames * 3, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList()
        output_channels = [64, 128, 256, 512]
        for idx, num_blocks in enumerate(layer_config):
            stride = 2 if idx > 0 else 1
            self.stages.append(self._make_layer(block_type, output_channels[idx], num_blocks, stride=stride))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            list: List of feature maps from each ResNet stage.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        features = [x]

        x = self.pool(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


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
            152: (models.resnet.Bottleneck, [3, 8, 36, 3])
        }

        if num_layers not in resnet_configs:
            raise ValueError(f"{num_layers} is not a valid number of ResNet layers")

        block_type, layer_config = resnet_configs[num_layers]
        self.encoder = MultiImageResNet(
            block_type, layer_config, num_image_frames=num_input_images
        )

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if pretrained and num_input_images == 1:
            state_dict = model_zoo.load_url(models.resnet.model_urls[f'resnet{num_layers}'])
            self.encoder.load_state_dict(state_dict)

    def forward(self, input_image):
        """
        Forward pass for the encoder.

        Args:
            input_image (torch.Tensor): Input image tensor.

        Returns:
            list: List of feature maps from each ResNet stage.
        """
        input_image = (input_image - 0.45) / 0.225
        features = self.encoder(input_image)
        return features
