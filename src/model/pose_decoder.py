import torch
import torch.nn as nn

class PoseEstimationDecoder(nn.Module):
    """
    Decoder for predicting pose transformations (rotations and translations) 
    from input feature maps.
    """
    def __init__(self, encoder_channels, input_feature_count, frames_to_predict=None, kernel_stride=1):
        """
        Args:
            encoder_channels (list): Channels from the encoder layers.
            input_feature_count (int): Number of input feature maps.
            frames_to_predict (int, optional): Frames to predict. Defaults None.
            kernel_stride (int, optional): Stride for convolutions. Defaults to 1.
        """
        super(PoseEstimationDecoder, self).__init__()

        self.encoder_channels = encoder_channels
        self.input_feature_count = input_feature_count
        self.frames_to_predict = frames_to_predict if frames_to_predict is not None else input_feature_count - 1

        self.layers = nn.ModuleDict({
            "compress": nn.Conv2d(self.encoder_channels[-1], 256, kernel_size=1),
            "pconv_0": nn.Conv2d(input_feature_count * 256, 256, kernel_size=3, stride=kernel_stride, padding=1),
            "pconv_1": nn.Conv2d(256, 256, kernel_size=3, stride=kernel_stride, padding=1),
            "pconv_2": nn.Conv2d(256, 6 * self.frames_to_predict, kernel_size=1),
        })

        self.activation = nn.ReLU()

    def forward(self, feature_inputs):
        """
        Forward pass to compute pose transformations.

        Args:
            feature_inputs (list of torch.Tensor): Input feature maps.

        Returns:
            tuple: Predicted rotations and translations.
        """
        last_layer_features = [features[-1] for features in feature_inputs]
        compressed_features = [self.activation(self.layers["compress"](feature)) for feature in last_layer_features]
        concatenated = torch.cat(compressed_features, dim=1)

        x = concatenated
        for idx in range(3):
            x = self.layers[f"p_{idx}"](x)
            if idx != 2:
                x = self.activation(x)

        x = x.mean(dim=(2, 3))
        x = 0.01 * x.view(-1, self.frames_to_predict, 1, 6)

        rotations = x[..., :3]
        translations = x[..., 3:]

        return rotations, translations
