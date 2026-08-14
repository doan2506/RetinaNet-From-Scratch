import math
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    RetinaNet Classification Subnet.
    Predicts probability of K classes for each anchor box.
    """

    def __init__(self, in_channels=256, num_anchors=9, num_classes=5, prior_prob=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        convs = []
        for _ in range(4):
            convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            convs.append(nn.ReLU(inplace=True))
        self.conv_subnet = nn.Sequential(*convs)

        self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob):
        for m in self.conv_subnet.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

        # Special initialization for focal loss stability
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(self, feature_maps: list) -> torch.Tensor:
        """
        feature_maps: List of feature map Tensors [P3, P4, P5, P6, P7]
        Returns: Tensor of classification logits (B, N_total, num_classes)
        """
        all_cls_logits = []
        for feat in feature_maps:
            batch_size = feat.shape[0]
            cls_out = self.conv_subnet(feat)
            cls_out = self.cls_pred(cls_out)  # (B, A*K, H, W)

            # Permute and reshape to (B, H*W*A, K)
            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = cls_out.view(batch_size, -1, self.num_classes)
            all_cls_logits.append(cls_out)

        return torch.cat(all_cls_logits, dim=1)


class RegressionHead(nn.Module):
    """
    RetinaNet Regression Subnet.
    Predicts (dx, dy, dw, dh) deltas for each anchor box.
    """

    def __init__(self, in_channels=256, num_anchors=9):
        super().__init__()
        self.num_anchors = num_anchors

        convs = []
        for _ in range(4):
            convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            convs.append(nn.ReLU(inplace=True))
        self.conv_subnet = nn.Sequential(*convs)

        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature_maps: list) -> torch.Tensor:
        """
        feature_maps: List of feature map Tensors [P3, P4, P5, P6, P7]
        Returns: Tensor of predicted deltas (B, N_total, 4)
        """
        all_reg_deltas = []
        for feat in feature_maps:
            batch_size = feat.shape[0]
            reg_out = self.conv_subnet(feat)
            reg_out = self.reg_pred(reg_out)  # (B, A*4, H, W)

            # Permute and reshape to (B, H*W*A, 4)
            reg_out = reg_out.permute(0, 2, 3, 1).contiguous()
            reg_out = reg_out.view(batch_size, -1, 4)
            all_reg_deltas.append(reg_out)

        return torch.cat(all_reg_deltas, dim=1)
