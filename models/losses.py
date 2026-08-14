import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for One-Stage Object Detection.
    FL(pt) = - alpha_t * (1 - pt)^gamma * log(pt)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="sum"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        logits: Tensor (N, num_classes) - raw predicted logits
        targets: Tensor (N,) - class index for positive (>=0), -1 for background, -2 for ignore
        valid_mask: Tensor (N,) - boolean mask where target != -2 (not ignored)
        """
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)

        # Filter out ignored anchors
        logits = logits[valid_mask]  # (N_valid, num_classes)
        targets = targets[valid_mask]  # (N_valid,)

        probs = torch.sigmoid(logits)

        # Create one-hot target tensor
        one_hot = torch.zeros_like(logits)
        pos_indices = targets >= 0
        if pos_indices.any():
            one_hot[pos_indices, targets[pos_indices]] = 1.0

        # Compute pt and alpha_t
        pt = torch.where(one_hot == 1, probs, 1 - probs)
        alpha_t = torch.where(one_hot == 1, self.alpha, 1 - self.alpha)

        # Focal Loss computation
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        bce_loss = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
        loss = focal_weight * bce_loss

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss


class BBoxRegressionLoss(nn.Module):
    """
    Smooth L1 Bounding Box Regression Loss calculated on Positive Anchors.
    """

    def __init__(self, beta=1.0 / 9.0, reduction="sum"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred_deltas: torch.Tensor, target_deltas: torch.Tensor, pos_mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred_deltas: Tensor (N, 4) or (N_pos, 4)
        target_deltas: Tensor (N, 4) or (N_pos, 4)
        pos_mask: Optional boolean mask for positive anchors
        """
        if pos_mask is not None:
            if not pos_mask.any():
                return torch.tensor(0.0, device=pred_deltas.device)
            pred_pos = pred_deltas[pos_mask]
            target_pos = target_deltas[pos_mask]
        else:
            if pred_deltas.numel() == 0:
                return torch.tensor(0.0, device=pred_deltas.device)
            pred_pos = pred_deltas
            target_pos = target_deltas

        loss = F.smooth_l1_loss(pred_pos, target_pos, beta=self.beta, reduction="none")

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        return loss
