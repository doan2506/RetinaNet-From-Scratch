import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) generating multi-scale feature maps (P3, P4, P5, P6, P7).
    """

    def __init__(self, in_channels_list: dict, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        # 1x1 Conv Lateral connections
        self.lat_c5 = nn.Conv2d(in_channels_list["c5"], out_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(in_channels_list["c4"], out_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(in_channels_list["c3"], out_channels, kernel_size=1)

        # 3x3 Conv Smooth layers for P3, P4, P5
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # P6 from C5 (backbone output), P7 from relu(P6) — per RetinaNet paper
        self.p6 = nn.Conv2d(in_channels_list["c5"], out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs: dict) -> list:
        c3, c4, c5 = inputs["c3"], inputs["c4"], inputs["c5"]

        # Top-down pathway
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        # Smooth layers
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        # P6: Conv 3x3 stride 2 on C5 (backbone output, NOT smoothed P5)
        p6 = self.p6(c5)

        # P7: Conv 3x3 stride 2 on relu(P6)
        p7 = self.p7(self.relu(p6))

        return [p3, p4, p5, p6, p7]
