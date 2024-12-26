import torch
import torch.nn as nn
import torch.nn.functional as F



class DepthDecoder2(nn.Module):
    def __init__(self, num_ch_enc, scales=[0, 1, 2, 3], enable_skips=True):
        """
        num_ch_enc: list of the number of output channels at each encoder level.
        scales: which resolution levels to compute depth maps.
        """
        super(DepthDecoder2, self).__init__()

        self.scales = scales
        self.num_ch_enc = num_ch_enc

        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.enable_skips = enable_skips

        self.upconvs = nn.ModuleDict()
        self.iconvs = nn.ModuleDict()
        self.disps = nn.ModuleDict()
        self.sigmoid = nn.Sigmoid()

        for i in range(4, -1, -1):
            self.upconvs[f"upconv_{i}_0"] = self._conv_block(self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1], self.num_ch_dec[i])
            input_channels = self.num_ch_dec[i]
            if self.enable_skips and i > 0:
                input_channels += self.num_ch_enc[i - 1]
            self.iconvs[f"iconv_{i}_1"] = self._conv_block(input_channels, self.num_ch_dec[i])
            if i in self.scales:
                self.disps[f"disp_{i}"] = self._conv3_disp(self.num_ch_dec[i], 1)

    def forward(self, input_features):
        """
        input_features: list of features from the encoder
        """
        outputs = {}

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.upconvs[f"upconv_{i}_0"](x)
            x = [self._upsample(x)]
            if self.enable_skips and i > 0:
                x.append(input_features[i - 1])

            x = torch.cat(x,1)
            x = self.iconvs[f"iconv_{i}_1"](x)

            if i in self.scales:
                outputs[f"disp_{i}"] = self.sigmoid(self.disps[f"disp_{i}"](x))

        return outputs

    def _conv_block(self, in_channels, out_channels):
        """Convolution block + ELU activation."""
        conv_layer = self._conv3_disp(in_channels, out_channels)
        return nn.Sequential(
            conv_layer,
            nn.ELU(inplace=True)
        )

    def _conv3_disp(self, in_channels, out_channels, use_refl=True):
        """3x3 convolution block with padding and optional reflection padding."""
        if use_refl:
            pad = nn.ReflectionPad2d(1)
        else:
            pad = nn.ZeroPad2d(1)
        conv = nn.Conv2d(in_channels, out_channels, 3)
        
        return nn.Sequential(pad, conv)

    def _upsample(self, x):
        """Upsampling by a factor of 2."""
        return F.interpolate(x, scale_factor=2, mode="nearest")
