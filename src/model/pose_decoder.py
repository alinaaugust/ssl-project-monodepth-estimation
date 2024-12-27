# import torch
# import torch.nn as nn


# class PoseEstimationDecoder(nn.Module):
#     """
#     Decoder for predicting pose transformations (rotations and translations)
#     from input feature maps.
#     """

#     def __init__(
#         self,
#         encoder_channels,
#         input_feature_count,
#         frames_to_predict=None,
#         kernel_stride=1,
#     ):
#         """
#         Args:
#             encoder_channels (list): Channels from the encoder layers.
#             input_feature_count (int): Number of input feature maps.
#             frames_to_predict (int, optional): Frames to predict. Defaults None.
#             kernel_stride (int, optional): Stride for convolutions. Defaults to 1.
#         """
#         super(PoseEstimationDecoder, self).__init__()

#         self.encoder_channels = encoder_channels
#         self.input_feature_count = input_feature_count
#         self.frames_to_predict = (
#             frames_to_predict
#             if frames_to_predict is not None
#             else input_feature_count - 1
#         )

#         self.layers = nn.ModuleDict(
#             {
#                 "compress": nn.Conv2d(self.encoder_channels[-1], 256, kernel_size=1),
#                 "pconv_0": nn.Conv2d(
#                     input_feature_count * 256,
#                     256,
#                     kernel_size=3,
#                     stride=kernel_stride,
#                     padding=1,
#                 ),
#                 "pconv_1": nn.Conv2d(
#                     256, 256, kernel_size=3, stride=kernel_stride, padding=1
#                 ),
#                 "pconv_2": nn.Conv2d(256, 6 * self.frames_to_predict, kernel_size=1),
#             }
#         )

#         self.activation = nn.ReLU()

#     def forward(self, feature_inputs):
#         """
#         Forward pass to compute pose transformations.

#         Args:
#             feature_inputs (list of torch.Tensor): Input feature maps.

#         Returns:
#             tuple: Predicted rotations and translations.
#         """
#         last_layer_features = [features[-1] for features in feature_inputs]
#         compressed_features = [
#             self.activation(self.layers["compress"](feature))
#             for feature in last_layer_features
#         ]
#         concatenated = torch.cat(compressed_features, dim=1)

#         x = concatenated
#         for idx in range(3):
#             x = self.layers[f"p_{idx}"](x)
#             if idx != 2:
#                 x = self.activation(x)

#         x = x.mean(dim=(2, 3))
#         x = 0.01 * x.view(-1, self.frames_to_predict, 1, 6)

#         rotations = x[..., :3]
#         translations = x[..., 3:]

#         return rotations, translations


from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseEstimationDecoder(nn.Module):
    def __init__(self, num_ch_enc, input_feature_count, frames_to_predict=None, stride=1):
        super(PoseEstimationDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = input_feature_count

        if frames_to_predict is None:
            frames_to_predict = input_feature_count - 1
        self.num_frames_to_predict_for = frames_to_predict

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(input_feature_count * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * frames_to_predict, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation