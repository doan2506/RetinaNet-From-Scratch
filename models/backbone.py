import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    ResNet Feature Extractor returning feature maps C3, C4, C5.
    Supports ResNet-34 and ResNet-50 backbones.
    """

    def __init__(self, architecture="resnet50", pretrained=True):
        super().__init__()
        self.architecture = architecture

        if architecture == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            self.out_channels = {"c3": 128, "c4": 256, "c5": 512}
        elif architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.out_channels = {"c3": 512, "c4": 1024, "c5": 2048}
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Stem layers (conv1 -> bn1 -> relu -> maxpool)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # ResNet Residual Blocks
        self.layer1 = resnet.layer1  # C2 (stride 4)
        self.layer2 = resnet.layer2  # C3 (stride 8)
        self.layer3 = resnet.layer3  # C4 (stride 16)
        self.layer4 = resnet.layer4  # C5 (stride 32)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return {"c3": c3, "c4": c4, "c5": c5}
